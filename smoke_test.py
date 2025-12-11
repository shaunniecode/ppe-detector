import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# Simple tensor test
x = torch.rand(3, 3).cuda() if torch.cuda.is_available() else torch.rand(3, 3)
y = torch.mm(x, x)
print("Matrix multiply result:\n", y)

