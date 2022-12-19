import torch
import numpy as np
import math
from matplotlib import cm

xyz_from_rgb = np.array(
    [[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]]
)
rgb_from_xyz = np.array(
    [[3.24048134, -0.96925495, 0.05564664], [-1.53715152, 1.87599, -0.20404134], [-0.49853633, 0.04155593, 1.05731107]]
)


def tensor_lab2rgb(input):
    """
    n * 3* h *w
    """
    input_trans = input.transpose(1, 2).transpose(2, 3)  # n * h * w * 3
    L, a, b = input_trans[:, :, :, 0:1], input_trans[:, :, :, 1:2], input_trans[:, :, :, 2:]
    y = (L + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)

    neg_mask = z.data < 0
    z[neg_mask] = 0 
    xyz = torch.cat((x, y, z), dim=3)

    mask = xyz.data > 0.2068966
    mask_xyz = xyz.clone()
    mask_xyz[mask] = torch.pow(xyz[mask], 3.0)
    mask_xyz[~mask] = (xyz[~mask] - 16.0 / 116.0) / 7.787
    mask_xyz[:, :, :, 0] = mask_xyz[:, :, :, 0] * 0.95047
    mask_xyz[:, :, :, 2] = mask_xyz[:, :, :, 2] * 1.08883

    rgb_trans = torch.mm(mask_xyz.view(-1, 3), torch.from_numpy(rgb_from_xyz).type_as(xyz)).view(
        input.size(0), input.size(2), input.size(3), 3
    )
    rgb = rgb_trans.transpose(2, 3).transpose(1, 2)

    mask = rgb > 0.0031308
    mask_rgb = rgb.clone()
    mask_rgb[mask] = 1.055 * torch.pow(rgb[mask], 1 / 2.4) - 0.055
    mask_rgb[~mask] = rgb[~mask] * 12.92

    neg_mask = mask_rgb.data < 0
    large_mask = mask_rgb.data > 1
    mask_rgb[neg_mask] = 0
    mask_rgb[large_mask] = 1
    return mask_rgb

def cal_normal_acc(gt_n, pred_n, thres=180.0, mask=None):
    """Tensor Dim: NxCxHxW"""
    dot_product = (gt_n * pred_n).sum(1).clamp(-1,1)
    error_map = torch.acos(dot_product) # [-pi, pi]
    angular_map = error_map * thres / math.pi
    angular_map = angular_map * mask.narrow(1, 0, 1).squeeze(1)

    valid = mask.narrow(1, 0, 1).sum()
    ang_valid = angular_map[mask.narrow(1, 0, 1).squeeze(1).byte()]
    n_err_mean = ang_valid.sum() / valid
    n_err_med = ang_valid.median()
    n_acc_11 = (ang_valid < 11.25).sum().float() / valid
    n_acc_30 = (ang_valid < 30).sum().float() / valid
    n_acc_45 = (ang_valid < 45).sum().float() / valid

    angular_map = colormap(angular_map.cpu().squeeze(1))
    value = {'n_err_mean': n_err_mean.item(), #'n_err_med': n_err_med.item(), 
    'n_acc_11': n_acc_11.item(), 'n_acc_30': n_acc_30.item(), 'n_acc_45': n_acc_45.item()}
    angular_error_map = {'angular_map': angular_map}
    return value, angular_error_map


# map 1 channel gray image (error) to 3 channel fake color image 
# output is three channel uint8 image
def colormap(error,thres=90):
    error_norm = np.clip(error, 0, thres) / thres
    error_map = cm.jet(error_norm)[:,:,0:3]
    return np.uint8(255*error_map)