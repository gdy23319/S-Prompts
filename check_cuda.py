import torch

# 1. PyTorch版本
print(f"PyTorch版本: {torch.__version__}")

# 2. PyTorch编译使用的CUDA版本（不是系统安装的CUDA版本）
print(f"PyTorch绑定的CUDA版本: {torch.version.cuda}")

# 3. CUDA是否可用
print(f"CUDA是否可用: {torch.cuda.is_available()}")

# 4. GPU数量
print(f"可用GPU数量: {torch.cuda.device_count()}")

# 5. 当前GPU名称和计算能力（核心！确认是否支持sm_120）
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(f"当前GPU: {torch.cuda.get_device_name(device)}")
    major, minor = torch.cuda.get_device_capability(device)
    print(f"GPU计算能力: sm_{major}{minor}")