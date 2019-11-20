import torch


def AP2(x):
    return torch.sign(x) * 2 ** torch.round(torch.log2(torch.abs(x)))


if __name__ == "__main__":
    print(AP2(torch.randn(3,3)))
