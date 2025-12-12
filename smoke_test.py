import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Always force CPU
device = torch.device("cpu")
print("Selected device:", device)

# Simple tensor test on CPU
x = torch.rand(3, 3, device=device)
y = torch.mm(x, x)
print("Matrix multiply result:\n", y)

