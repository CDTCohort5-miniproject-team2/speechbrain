import torch

print("CUDA is available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Default device ref:", torch.cuda.device(0))
print("No. of GPUs:", torch.cuda.device_count())
print("Name of default device:", torch.cuda.get_device_name(0))