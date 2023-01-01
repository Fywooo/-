import torch

print("Total GPU Count:{}".format(torch.cuda.device_count()))   #查看所有可用GPU个数
print("Total CPU Count:{}".format(torch.cuda.os.cpu_count()))   #获取系统CPU数量
print(torch.cuda.get_device_name(torch.device("cuda:0")))       #获取GPU设备名称   NVIDIA GeForce GT 1030
print("GPU Is Available:{}".format(torch.cuda.is_available()))  #GPU设备是否可用  True
