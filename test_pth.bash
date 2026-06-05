PATH_LOCATION=/home/lenovo/S-Prompts/logs/reproduce_1993_sprompts_slip_cddb_2_2_2026-06-04-00:52:16

python /home/lenovo/eval-S-Prompts/SPrompts_eval/eval.py --resume $PATH_LOCATION/task_0.pth --dataroot /scratch/el3g21/CDDB/ --datatype deepfake &>> eval.txt
python /home/lenovo/eval-S-Prompts/SPrompts_eval/eval.py --resume $PATH_LOCATION/task_1.pth --dataroot /scratch/el3g21/CDDB/ --datatype deepfake &>> eval.txt
python /home/lenovo/eval-S-Prompts/SPrompts_eval/eval.py --resume $PATH_LOCATION/task_2.pth --dataroot /scratch/el3g21/CDDB/ --datatype deepfake &>> eval.txt
python /home/lenovo/eval-S-Prompts/SPrompts_eval/eval.py --resume $PATH_LOCATION/task_3.pth --dataroot /scratch/el3g21/CDDB/ --datatype deepfake &>> eval.txt
python /home/lenovo/eval-S-Prompts/SPrompts_eval/eval.py --resume $PATH_LOCATION/task_4.pth --dataroot /scratch/el3g21/CDDB/ --datatype deepfake &>> eval.txt
python /home/lenovo/eval-S-Prompts/SPrompts_eval/eval.py --resume $PATH_LOCATION/task_5.pth --dataroot /scratch/el3g21/CDDB/ --datatype deepfake &>> eval.txt
python /home/lenovo/eval-S-Prompts/SPrompts_eval/eval.py --resume $PATH_LOCATION/task_6.pth --dataroot /scratch/el3g21/CDDB/ --datatype deepfake &>> eval.txt