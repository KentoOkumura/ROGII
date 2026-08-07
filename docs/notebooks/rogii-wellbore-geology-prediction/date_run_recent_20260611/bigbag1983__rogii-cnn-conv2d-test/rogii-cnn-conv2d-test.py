import torch
import torch.nn as nn

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

# Test 1: Basic conv2d
print("\nTest 1: Basic Conv2d...")
try:
    m = nn.Conv2d(5, 32, 3, padding=1).cuda()
    x = torch.randn(2, 5, 256, 512).cuda()
    y = m(x)
    print("Basic conv2d OK:", y.shape)
except Exception as e:
    print("Basic conv2d FAILED:", e)

# Test 2: BatchNorm + Conv
print("\nTest 2: Conv + BatchNorm...")
try:
    m = nn.Sequential(
        nn.Conv2d(5, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
    ).cuda()
    y = m(x)
    print("Conv+BN OK:", y.shape)
except Exception as e:
    print("Conv+BN FAILED:", e)

# Test 3: torchvision resnet
print("\nTest 3: torchvision resnet18...")
try:
    import torchvision
    m = torchvision.models.resnet18(pretrained=False).cuda()
    x2 = torch.randn(2, 3, 224, 224).cuda()
    y = m(x2)
    print("ResNet18 OK:", y.shape)
except Exception as e:
    print("ResNet18 FAILED:", e)

print("\nDone.")
