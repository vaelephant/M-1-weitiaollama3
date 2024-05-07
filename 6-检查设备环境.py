import datetime
import torch


# 第一步检查系统环境

print('环境检查如下：')
# 打印 PyTorch 的版本
print("PyTorch Version:", torch.__version__)

# 打印 CUDA 的版本，如果 CUDA 可用的话
print("CUDA Version:", torch.version.cuda)

# 检查 CUDA 是否可用
print("CUDA is available:", torch.cuda.is_available())

# 如果 CUDA 可用，打印出当前 CUDA 设备的数量和名称
if torch.cuda.is_available():
    print("Number of CUDA Devices:", torch.cuda.device_count())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
