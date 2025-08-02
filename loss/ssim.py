import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    # gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    gauss = []
    for x in range(window_size):
        value = exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        gauss.append(value)

    gauss = torch.Tensor(gauss)
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss




# def PSNR(img1, img2):
# 	# pdb.set_trace()
# 	img1 = np.float64(img1.cpu())
# 	img2 = np.float64(img2.cpu())
# 	mse = np.mean( (img1 - img2) ** 2 )
# 	if mse == 0: 8
# 		return 100
# 	PIXEL_MAX = 255.0
# 	psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
# 	print(psnr)
# 	return psnr

# import torch
# import torch.nn.functional as F
# import math
#
# def PSNR(img1, img2):
#     # Ensure img1 and img2 are PyTorch tensors
#     if not isinstance(img1, torch.Tensor) or not isinstance(img2, torch.Tensor):
#         raise ValueError("Both inputs must be PyTorch tensors")
#
#     # Convert tensors to NumPy arrays and then to images
#     img1_np = img1.cpu().detach().numpy()
#     img2_np = img2.cpu().detach().numpy()
#
#     # Convert from [0, 1] range to [0, 255]
#     img1_np = np.clip(img1_np * 255.0, 0, 255).astype(np.uint8)
#     img2_np = np.clip(img2_np * 255.0, 0, 255).astype(np.uint8)
#
#     # Calculate PSNR
#     mse = np.mean((img1_np - img2_np) ** 2)
#     if mse == 0:
#         return 100
#     PIXEL_MAX = 255.0
#     psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
#     return psnr


# def PSNR(img1, img2):
#     """
#     Compute PSNR between two images.
#
#     Parameters:
#     img1 (torch.Tensor): First image tensor.
#     img2 (torch.Tensor): Second image tensor.
#
#     Returns:
#     float: PSNR value.
#     """
#     # Ensure the images have the same shape
#     if img1.shape != img2.shape:
#         raise ValueError("Input images must have the same dimensions.")
#
#     # Flatten the tensors to (N, C, -1)
#     img1 = img1.view(img1.size(0), -1)
#     img2 = img2.view(img2.size(0), -1)
#
#     # Calculate MSE
#     mse = F.mse_loss(img1, img2, reduction='mean')
#
#     # If MSE is zero, PSNR is infinity (perfect match)
#     if mse == 0:
#         return float('inf')
#
#     # Calculate PSNR
#     pixel_max = 1.0  # Assuming normalized images (pixel values in [0, 1])
#     psnr = 20 * torch.log10(pixel_max / torch.sqrt(mse))
#
#     return psnr.item()

import numpy as np
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio

def PSNR(img1, img2):
    """
    Calculate PSNR between two PyTorch tensors.

    Parameters:
    img1 (torch.Tensor): First image tensor.
    img2 (torch.Tensor): Second image tensor.

    Returns:
    float: PSNR value.
    """
    # Ensure the images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Convert PyTorch tensors to NumPy arrays
    img1_np = img1.squeeze().cpu().detach().numpy()
    img2_np = img2.squeeze().cpu().detach().numpy()

    # Rescale the images to the range [0, 1]
    img1_np = (img1_np - np.min(img1_np)) / (np.max(img1_np) - np.min(img1_np))
    img2_np = (img2_np - np.min(img2_np)) / (np.max(img2_np) - np.min(img2_np))

    # Convert to uint8 images (assuming 8-bit images)
    img1_np = img_as_ubyte(img1_np)
    img2_np = img_as_ubyte(img2_np)

    # Compute PSNR
    psnr_value = peak_signal_noise_ratio(img1_np, img2_np)

    return psnr_value


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice