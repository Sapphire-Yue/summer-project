import os

train_path = "E:/Shiro/OneDrive - 中原大學/1131/專題研究/暑期專題/summer-project/only_dataset/images/train"
val_path = "E:/Shiro/OneDrive - 中原大學/1131/專題研究/暑期專題/summer-project/only_dataset/images/val"

print("Train path exists:", os.path.exists(train_path))
print("Validation path exists:", os.path.exists(val_path))

yaml_path = "E:/Shiro/OneDrive - 中原大學/1131/專題研究/暑期專題/summer-project/only_dataset/dataset.yaml"

if os.path.exists(yaml_path):
    print("YAML file found!")
else:
    print("YAML file not found!")

import torch
print(torch.__version__)  # 檢查 PyTorch 版本
print(torch.cuda.is_available())  # 檢查 CUDA 是否可用
print(torch.cuda.current_device())  # 當前設備
print(torch.cuda.get_device_name(0))  # 顯示當前 GPU 名稱
