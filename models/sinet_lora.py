import torch
import torch.nn as nn
import copy
import math

from models.vit import VisionTransformer, PatchEmbed, Block, resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from models.sinet import ViT_Prompts, _create_vision_transformer


class LoRASet(nn.Module):
    """Holds the LoRA A/B matrices for one domain across all transformer layers.

    For each layer i (0..num_layers-1):
      - lora_A_q[i]: shape [rank, dim]  (Q down-projection, Gaussian init)
      - lora_B_q[i]: shape [dim,  rank] (Q up-projection,   zero init)
      - lora_A_v[i]: shape [rank, dim]  (V down-projection, Gaussian init)
      - lora_B_v[i]: shape [dim,  rank] (V up-projection,   zero init)

    LoRA delta: delta = scale * (x @ A.t()) @ B.t()
    scale = lora_alpha / lora_rank
    """

    def __init__(self, num_layers: int, dim: int, lora_rank: int, lora_alpha: float):
        super().__init__()
        self.scale = lora_alpha / lora_rank

        self.lora_A_q = nn.ParameterList()
        self.lora_B_q = nn.ParameterList()
        self.lora_A_v = nn.ParameterList()
        self.lora_B_v = nn.ParameterList()

        for _ in range(num_layers):
            self.lora_A_q.append(nn.Parameter(torch.empty(lora_rank, dim)))
            self.lora_B_q.append(nn.Parameter(torch.zeros(dim, lora_rank)))
            self.lora_A_v.append(nn.Parameter(torch.empty(lora_rank, dim)))
            self.lora_B_v.append(nn.Parameter(torch.zeros(dim, lora_rank)))

        # Kaiming uniform init for A matrices (same as nn.Linear default)
        for param in list(self.lora_A_q) + list(self.lora_A_v):
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))


class ViT_LoRA(ViT_Prompts):
    """ViT that supports domain-specific LoRA adapters.

    When lora_set is None the forward pass is identical to the standard ViT
    (used for extract_vector / K-means clustering).
    When lora_set is provided, each transformer block is run through
    forward_with_lora, which injects per-layer Q/V LoRA deltas.
    """

    def forward(self, x, lora_set=None, **kwargs):
        if lora_set is None:
            # Standard ViT forward (also handles instance_tokens via super)
            return super().forward(x, **kwargs)

        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed.to(x.dtype)
        x = self.pos_drop(x)

        for i, block in enumerate(self.blocks):
            x = block.forward_with_lora(x, lora_set, i)

        x = self.norm(x)
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x


def _create_vit_lora(variant='vit_base_patch16_224', pretrained=True):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    pretrained_cfg = resolve_pretrained_cfg(variant)
    if hasattr(pretrained_cfg, 'to_dict'):
        pretrained_cfg = pretrained_cfg.to_dict()
    if 'url' in pretrained_cfg and 'npz' in pretrained_cfg['url']:
        pretrained_cfg['custom_load'] = True
    repr_size = None

    model = build_model_with_cfg(
        ViT_LoRA, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        **model_kwargs,
    )
    return model


class SiNet_LoRA(nn.Module):
    """Domain-incremental learner using per-domain LoRA adapters instead of prompts.

    Architecture:
      - Shared frozen ViT-B/16 backbone
      - lora_pool[t]: LoRASet for domain t (trainable during task t)
      - classifier_pool[t]: Linear head for domain t (trainable during task t)

    K-means domain detection uses extract_vector() (plain ViT, no LoRA) for
    consistency with the original S-Prompts design.
    """

    def __init__(self, args):
        super().__init__()

        self.lora_rank = args.get('lora_rank', 4)
        self.lora_alpha = args.get('lora_alpha', 8.0)
        num_layers = 12  # ViT-B/16
        dim = args['embd_dim']  # 768

        self.image_encoder = _create_vit_lora('vit_base_patch16_224', pretrained=True)

        self.class_num = 1
        if args['dataset'] == 'cddb':
            self.class_num = 2
        elif args['dataset'] == 'domainnet':
            self.class_num = 345
        elif args['dataset'] == 'core50':
            self.class_num = 50
        else:
            raise ValueError('Unknown dataset: {}'.format(args['dataset']))

        total = args['total_sessions']
        self.classifier_pool = nn.ModuleList([
            nn.Linear(dim, self.class_num, bias=True)
            for _ in range(total)
        ])
        self.lora_pool = nn.ModuleList([
            LoRASet(num_layers, dim, self.lora_rank, self.lora_alpha)
            for _ in range(total)
        ])

        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.embed_dim

    def extract_vector(self, image):
        """Plain ViT features (no LoRA) used for K-means clustering."""
        features = self.image_encoder(image)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def forward(self, image):
        """Training forward: use LoRA adapter for the current task."""
        image_features = self.image_encoder(image, lora_set=self.lora_pool[self.numtask - 1])
        logits = self.classifier_pool[self.numtask - 1](image_features)
        return {
            'logits': logits,
            'features': image_features,
        }

    def interface(self, image, selection):
        """Inference forward: route each sample to its detected domain.

        Args:
            image:     [B, C, H, W]
            selection: [B] integer tensor with domain index per sample
        Returns:
            [B, class_num] logits for the selected domain's classifier
        """
        unique_tasks = selection.unique()
        features = torch.zeros(image.size(0), self.image_encoder.embed_dim,
                               device=image.device, dtype=image.dtype)

        for task_id in unique_tasks:
            mask = (selection == task_id).nonzero(as_tuple=True)[0]
            feat = self.image_encoder(image[mask], lora_set=self.lora_pool[task_id.item()])
            features[mask] = feat

        # Classify each sample with its domain's classifier
        selectedlogit = []
        for idx, task_id in enumerate(selection):
            logit = self.classifier_pool[task_id.item()](features[idx].unsqueeze(0))
            selectedlogit.append(logit)
        return torch.cat(selectedlogit, dim=0)

    def update_fc(self, nb_classes):
        self.numtask += 1

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self
