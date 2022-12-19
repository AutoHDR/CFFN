# from scipy.misc import face
import torch, shutil
from torch.functional import norm
import torch.nn as nn
import numpy as np
# from torchvision.transforms.transforms import GaussianBlur
# from GuidedFilter import *
from torch.autograd import Variable
import time
from torch.utils.tensorboard import SummaryWriter
import os, tarfile
from torchvision.utils import save_image
# from guided_filter_pytorch.guided_filter import GuidedFilter
#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch.nn.functional as F

# Guided_filter = GuidedFilter(3, 0.01)
# GBlur_img = GaussianBlur(3,3).to(device)

Sobel = np.array([[-1,-2,-1],
                  [ 0, 0, 0],
                  [ 1, 2, 1]])
Robert = np.array([[0, 0],
                  [-1, 1]])
Sobel = torch.Tensor(Sobel)
Robert = torch.Tensor(Robert)

def KDgradient(maps, direction, device='cuda', kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
        maps = F.pad(maps, (0, 0, 1, 1))
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
        maps = F.pad(maps, (1, 1, 1, 1))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    kernel = kernel.to(device=device)
    # kernel size is (2, 2) so need pad bottom and right side
    gradient_orig = torch.abs(F.conv2d(maps, weight=kernel, padding=0).detach())
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm

    
def illumination_smoothness(I, L):
    L_gray = 0.299*L[:,0,:,:] + 0.587*L[:,1,:,:] + 0.114*L[:,2,:,:]
    L_gray = L_gray.unsqueeze(dim=1)
    I_gradient_x = KDgradient(I, "x")
    L_gradient_x = KDgradient(L_gray, "x")
    epsilon = 0.01*torch.ones_like(L_gradient_x)
    Denominator_x = torch.max(L_gradient_x, epsilon)
    x_loss = torch.abs(torch.div(I_gradient_x, Denominator_x))
    I_gradient_y = KDgradient(I, "y")
    L_gradient_y = KDgradient(L_gray, "y")
    Denominator_y = torch.max(L_gradient_y, epsilon)
    y_loss = torch.abs(torch.div(I_gradient_y, Denominator_y))
    mut_loss = torch.mean(x_loss + y_loss)
    return  mut_loss

def grad_smoothness(L):
    L_gray = 0.299*L[:,0,:,:] + 0.587*L[:,1,:,:] + 0.114*L[:,2,:,:]
    L_gray = L_gray.unsqueeze(dim=1)
    L_gradient_x = KDgradient(L_gray, "x")
    epsilon = 0.01*torch.ones_like(L_gradient_x)
    
    Denominator_x = torch.max(L_gradient_x, epsilon)
    gradient_x = KDgradient(L_gray, "x")
    x_loss = torch.abs(torch.div(gradient_x, Denominator_x))

    
    gradient_y = KDgradient(L_gray, "y")
    Denominator_y = torch.max(gradient_y, epsilon)
    y_loss = torch.abs(torch.div(gradient_y, Denominator_y))

    mut_loss = torch.mean(x_loss + y_loss)
    return  mut_loss#, x_loss, y_loss

def Img_smooth(L, esilon = 0.01):
    L_gray = (0.299*L[:,0,:,:] + 0.587*L[:,1,:,:] + 0.114*L[:,2,:,:]).unsqueeze(1)
    L_gradient_x = KDgradient(L_gray, "x")
    epsilon = esilon*torch.ones_like(L_gradient_x)
    
    Denominator_x = torch.max(L_gradient_x, epsilon)
    gradient_x = KDgradient(L_gray, "x")
    x_loss = torch.abs(torch.div(gradient_x, Denominator_x))

    
    gradient_y = KDgradient(L_gray, "y")
    Denominator_y = torch.max(gradient_y, epsilon)
    y_loss = torch.abs(torch.div(gradient_y, Denominator_y))

    mut_loss = (x_loss + y_loss)/2
    return  mut_loss#, x_loss, y_loss

class Gradient_Net(nn.Module):
    def __init__(self):
        super(Gradient_Net, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

        kernel_y = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        gray_input = (x[:, 0, :, :] + x[:, 1, :, :] + x[:, 2, :, :]).reshape([x.size(0), 1, x.size(2), x.size(2)]) / 3
        grad_x = F.conv2d(gray_input, self.weight_x).to(device)
        grad_y = F.conv2d(gray_input, self.weight_y).to(device)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient

def Img_gradient(x):
    gradient_model = Gradient_Net()
    # gradient_model = Gradient_Net_Single()
    g = gradient_model(x)
    return g

def get_normal_255(normal):
    new_normal = normal * 128 + 128
    new_normal = new_normal.clamp(0, 255) / 255
    return new_normal

def get_normal_P(normal): #[-1, 1]
    new_normal = normal /2 + 0.5     
    return new_normal #[0, 1]

def get_normal_N(normal): #[0, 1]
    new_normal = 2 * (normal - 0.5)    
    return new_normal #[-1, 1]

def make_targz(output_filename, source_dir):
    """
    一次性打包目录为tar.gz
    :param output_filename: 压缩文件名
    :param source_dir: 需要打包的目录
    :return: bool
    """
    try:
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))

        return True
    except Exception as e:
        print(e)
        return False

def get_MAE_RMSE(p, g, mask):
    C_MAE = nn.L1Loss()
    C_MSE = nn.MSELoss()
    with torch.no_grad():
        mae = C_MAE(p * mask, g * mask)*255
        rmse = torch.sqrt(C_MSE(p * mask, g * mask))*255
    return mae, rmse  # , mean


def get_Normal_Std_Mean(normal, gtnormal, mask):
    device = normal.device

    pi = 3.1415926
    b, c, w, h = normal.shape
    mask = torch.where(mask > 0.8, torch.ones(1).to(device), torch.zeros(1).to(device))
    normal = normal * mask
    gtnormal = gtnormal * mask
    x1 = normal[:, 0, :, :]
    y1 = normal[:, 1, :, :]
    z1 = normal[:, 2, :, :]

    gx1 = gtnormal[:, 0, :, :]
    gy1 = gtnormal[:, 1, :, :]
    gz1 = gtnormal[:, 2, :, :]

    up = (x1 * gx1 + y1 * gy1 + z1 * gz1)
    low = torch.sqrt(x1 * x1 + y1 * y1 + z1 * z1) * torch.sqrt(gx1 * gx1 + gy1 * gy1 + gz1 * gz1)
    mask = torch.where(mask > 0.8, torch.ones(1).to(device), torch.zeros(1).to(device))
    pixel_number = torch.sum(mask[:, 0, :, :])

    normal_angle_pi = (up / (low + 0.000001))
    # cos_angle_err = torch.where(normal_angle_pi>0.98, torch.ones(1).to(device), torch.zeros(1).to(device))
    cos_angle_err = normal_angle_pi

    angle_err_hudu = torch.acos(cos_angle_err)
    angle_err_du = (angle_err_hudu / pi * 180) * mask[:, 0, :, :]

    mean = torch.sum((angle_err_du)) / pixel_number
    std = torch.sqrt(torch.sum((angle_err_du - mean) * (angle_err_du - mean) * mask[:, 0, :, :]) / pixel_number)

    ang20 = torch.ones(1).to(device) * 20
    ang25 = torch.ones(1).to(device) * 25
    ang30 = torch.ones(1).to(device) * 30
    ang0 = torch.ones(1).to(device) * 0
    ang1 = torch.ones(1).to(device)
    count_20 = torch.sum(torch.where(angle_err_du < ang20, ang1, ang0) * mask[:, 0, :, :]) / pixel_number
    count_25 = torch.sum(torch.where(angle_err_du < ang25, ang1, ang0) * mask[:, 0, :, :]) / pixel_number
    count_30 = torch.sum(torch.where(angle_err_du < ang30, ang1, ang0) * mask[:, 0, :, :]) / pixel_number
    
    return std, mean, count_20, count_25, count_30


def GetShading20211113(N, L):
    b, c, h, w = N.shape
    for i in range(b):
        N1 = N[i,:,:,:]
        N2 = torch.zeros_like(N1)
        N2[2,:,:] = N1[0,:,:]
        N2[1,:,:] = N1[1,:,:]
        N2[0,:,:] = N1[2,:,:]
        N3 = torch.zeros_like(N2)
        N3[0,:,:] = N2[0,:,:]
        N3[1,:,:] = N2[2,:,:]
        N3[2,:,:] = -1 * N2[1,:,:]
        N3=N3.permute([1,2,0]).reshape([-1,3])
        norm_X = N3[:,0]
        norm_Y = N3[:,1]
        norm_Z = N3[:,2]
        numElem = norm_X.shape[0]
        sh_basis = torch.from_numpy(np.zeros([numElem, 9])).type(torch.FloatTensor)
        att = torch.from_numpy(np.pi * np.array([1, 2.0 / 3.0, 1 / 4.0])).type(torch.FloatTensor)
        sh_basis[:, 0] = torch.from_numpy(np.array(0.5 / np.sqrt(np.pi), dtype=float)).type(torch.FloatTensor) * \
                         att[0]

        sh_basis[:, 1] = torch.from_numpy(np.array(np.sqrt(3) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor) * norm_Y * att[1]
        sh_basis[:, 2] = torch.from_numpy(np.array(np.sqrt(3) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor) * norm_Z * att[1]
        sh_basis[:, 3] = torch.from_numpy(np.array(np.sqrt(3) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor) * norm_X * att[1]

        sh_basis[:, 4] = torch.from_numpy(np.array(np.sqrt(15) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor) * norm_Y * norm_X * att[2]
        sh_basis[:, 5] = torch.from_numpy(np.array(np.sqrt(15) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor) * norm_Y * norm_Z * att[2]
        sh_basis[:, 6] = torch.from_numpy(np.array(np.sqrt(5) / 4 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor) * (3 * norm_Z ** 2 - 1) * att[2]
        sh_basis[:, 7] = torch.from_numpy(np.array(np.sqrt(15) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor) * norm_X * norm_Z * att[2]
        sh_basis[:, 8] = torch.from_numpy(np.array(np.sqrt(15) / 4 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor) * (norm_X ** 2 - norm_Y ** 2) * att[2]

        light = L[i, :]
        shading = torch.matmul(sh_basis, light)
        myshading = (shading - torch.min(shading)) / (torch.max(shading) - torch.min(shading))

        tp = myshading.reshape([-1, h, w])
        if i == 0:
            result = tp
        else:
            result = torch.cat([result, tp], axis=0)

    b, w, h = result.shape
    return result.reshape([b, 1, w, h])

def N_SFS2CM(normal):
    tt = torch.zeros_like(normal)
    tt[:, 0, :, :] = normal[:, 1, :, :]
    tt[:, 1, :, :] = - normal[:, 0, :, :]
    tt[:, 2, :, :] = normal[:, 2, :, :]
    return tt


def normal_DPR2SFS(normal):
    tt = torch.zeros_like(normal)
    tt[:, 0, :, :] = normal[:, 1, :, :]
    tt[:, 1, :, :] = normal[:, 2, :, :]
    tt[:, 2, :, :] = -normal[:, 0, :, :]
    return tt

def normal_SFS2DPR(normal):
    tt = torch.zeros_like(normal)
    tt[:, 0, :, :] = -normal[:, 2, :, :]
    tt[:, 1, :, :] = normal[:, 0, :, :]
    tt[:, 2, :, :] = normal[:, 1, :, :]
    return tt
def normal_DPR2SHTool(normal):
    tt = torch.zeros_like(normal)
    tt[:, 0, :, :] = normal[:, 0, :, :]
    tt[:, 1, :, :] = normal[:, 2, :, :]
    tt[:, 2, :, :] = -normal[:, 1, :, :]
    return tt

def R_X(normal, angle):
    tt = torch.zeros_like(normal)
    x = normal[:, 0, :, :]
    y = normal[:, 1, :, :]
    z = normal[:, 2, :, :]
    tt[:, 0, :, :] = x
    tt[:, 1, :, :] = y*torch.cos(angle)-y*torch.sin(angle)
    tt[:, 2, :, :] = y*torch.sin(angle)+z*torch.cos(angle)
    return tt

def R_Y(normal, angle):
    tt = torch.zeros_like(normal)
    x = normal[:, 0, :, :]
    y = normal[:, 1, :, :]
    z = normal[:, 2, :, :]
    tt[:, 0, :, :] = x*torch.cos(angle) + z*torch.sin(angle)
    tt[:, 1, :, :] = y
    tt[:, 2, :, :] = -x*torch.sin(angle)+z*torch.cos(angle)
    return tt


def R_Z(normal, angle):
    tt = torch.zeros_like(normal)
    x = normal[:, 0, :, :]
    y = normal[:, 1, :, :]
    z = normal[:, 2, :, :]
    tt[:, 0, :, :] = x*torch.cos(angle) - y*torch.sin(angle)
    tt[:, 1, :, :] = x*torch.sin(angle)+y*torch.cos(angle)
    tt[:, 2, :, :] = z
    return tt
def mkdirss(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
from torchvision.transforms import Compose, ToTensor
import glob
from PIL import Image
import numpy as np
import pandas as pd
img_transform = Compose([
  ToTensor()
])
l0 = '/home/xteam1/2022/model_data/example_light/DPR/rotate_light_00.txt'
l1 = '/home/xteam1/2022/model_data/example_light/DPR/rotate_light_01.txt'
l2 = '/home/xteam1/2022/model_data/example_light/DPR/rotate_light_02.txt'
l3 = '/home/xteam1/2022/model_data/example_light/DPR/rotate_light_03.txt'
l4 = '/home/xteam1/2022/model_data/example_light/DPR/rotate_light_04.txt'
l5 = '/home/xteam1/2022/model_data/example_light/DPR/rotate_light_05.txt'
l6 = '/home/xteam1/2022/model_data/example_light/DPR/rotate_light_06.txt'
pd_sh0 = pd.read_csv(l0, sep='\t', header=None, encoding=u'gbk')
sh0 = torch.tensor(pd_sh0.values).type(torch.float).reshape([1, 9])
pd_sh1 = pd.read_csv(l1, sep='\t', header=None, encoding=u'gbk')
sh1 = torch.tensor(pd_sh1.values).type(torch.float).reshape([1, 9])
pd_sh2 = pd.read_csv(l2, sep='\t', header=None, encoding=u'gbk')
sh2 = torch.tensor(pd_sh2.values).type(torch.float).reshape([1, 9])
pd_sh3 = pd.read_csv(l3, sep='\t', header=None, encoding=u'gbk')
sh3 = torch.tensor(pd_sh3.values).type(torch.float).reshape([1, 9])
pd_sh4 = pd.read_csv(l4, sep='\t', header=None, encoding=u'gbk')
sh4 = torch.tensor(pd_sh4.values).type(torch.float).reshape([1, 9])
pd_sh5 = pd.read_csv(l5, sep='\t', header=None, encoding=u'gbk')
sh5 = torch.tensor(pd_sh5.values).type(torch.float).reshape([1, 9])
pd_sh6 = pd.read_csv(l6, sep='\t', header=None, encoding=u'gbk')
sh6 = torch.tensor(pd_sh6.values).type(torch.float).reshape([1, 9])

angle90 = torch.from_numpy(np.array(3.1415926/2)).type(torch.FloatTensor)
angle180 = torch.from_numpy(np.array(3.1415926)).type(torch.FloatTensor)
angle270 = torch.from_numpy(np.array(3.1415926+3.1415926/2)).type(torch.FloatTensor)


att= torch.from_numpy(np.pi*np.array([1, 2.0/3.0, 1/4.0])).type(torch.FloatTensor)
aa0 = torch.from_numpy(np.array(0.5/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)
aa1 = torch.from_numpy(np.array(np.sqrt(3)/2/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)
aa2 = torch.from_numpy(np.array(np.sqrt(15)/2/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)
aa3 = torch.from_numpy(np.array(np.sqrt(5)/4/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)
aa4 = torch.from_numpy(np.array(np.sqrt(15)/4/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)


def get_shading_DPR_B(N, L, VisLight=True):
    # N: DPR normal coordinate
    b, c, h, w = N.shape
    Norm = N.reshape([b, c, -1]).permute([0, 2, 1])  # b*c*65536   ----  b* 65536 * c
    if VisLight == True:
        norm_X = Norm[:, :, 0]
        norm_Y = Norm[:, :, 1]
        norm_Z = Norm[:, :, 2]
    else:
        norm_X = Norm[:, :, 2]
        norm_Y = Norm[:, :, 0]
        norm_Z = -Norm[:, :, 1]
    sh_basis = torch.from_numpy(np.zeros([b, h*w, 9])).type(torch.FloatTensor).to(N.device)
    sh_basis[:, :, 0] = aa0.to(N.device)*att[0]
    sh_basis[:, :, 1] = aa1.to(N.device)*norm_Y*att[1]
    sh_basis[:, :, 2] = aa1.to(N.device)*norm_Z*att[1]
    sh_basis[:, :, 3] = aa1.to(N.device)*norm_X*att[1]

    sh_basis[:, :, 4] = aa2.to(N.device)*norm_Y*norm_X*att[2]
    sh_basis[:, :, 5] = aa2.to(N.device)*norm_Y*norm_Z*att[2]
    sh_basis[:, :, 6] = aa3.to(N.device)*(3*norm_Z**2-1)*att[2]
    sh_basis[:, :, 7] = aa2.to(N.device)*norm_X*norm_Z*att[2]
    sh_basis[:, :, 8] = aa4.to(N.device)*(norm_X**2-norm_Y**2)*att[2]

    shading = torch.matmul(sh_basis, L.unsqueeze(2)).permute([0, 2, 1]).reshape([b, 1, w, h])
    shading = ImageBatchNormalization(shading)
    return shading
   
def Sphere_DPR(normal,lighting):
    # ---------------- create normal for rendering half sphere ------
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x ** 2 + z ** 2)
    valid = mag <= 1
    y = -np.sqrt(1 - (x * valid) ** 2 - (z * valid) ** 2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal_sp = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=2)
    normal_cuda = torch.from_numpy(normal_sp.astype(np.float32)).permute([2,0,1])#.reshape([1,256,256,3])
    normalBatch = torch.zeros_like(normal)
    for i in range(lighting.size()[0]):
        normalBatch[i,:,:,:] = normal_cuda
    # SpVisual = get_shading_DPR_B_0802(normalBatch, lighting)
    return normalBatch

def ImageBatchNormalization(input):
    [b,c,w,h] = input.size()
    tp_max = input.max(dim=1).values.max(dim=1).values.max(dim=1).values.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand([b,c,w,h])
    tp_min = input.min(dim=1).values.min(dim=1).values.min(dim=1).values.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand([b,c,w,h])
    tp_data =  (input - tp_min) / (tp_max - tp_min + 0.00001)
    return tp_data

def get_L_DPR_B(S, N):
    b, c, h, w = N.shape
    Norm = N.reshape([b, c, -1]).permute([0, 2, 1])  # b*c*65536   ----  b* 65536 * c
    norm_X = Norm[:, :, 2]
    norm_Y = Norm[:, :, 0]
    norm_Z = -Norm[:, :, 1]
    sh_basis = torch.from_numpy(np.zeros([b, h*w, 9])).type(torch.FloatTensor).to(N.device)
    sh_basis[:, :, 0] = aa0.to(N.device)*att[0]
    sh_basis[:, :, 1] = aa1.to(N.device)*norm_Y*att[1]
    sh_basis[:, :, 2] = aa1.to(N.device)*norm_Z*att[1]
    sh_basis[:, :, 3] = aa1.to(N.device)*norm_X*att[1]

    sh_basis[:, :, 4] = aa2.to(N.device)*norm_Y*norm_X*att[2]
    sh_basis[:, :, 5] = aa2.to(N.device)*norm_Y*norm_Z*att[2]
    sh_basis[:, :, 6] = aa3.to(N.device)*(3*norm_Z**2-1)*att[2]
    sh_basis[:, :, 7] = aa2.to(N.device)*norm_X*norm_Z*att[2]
    sh_basis[:, :, 8] = aa4.to(N.device)*(norm_X**2-norm_Y**2)*att[2]

    S = S[:, 0, :, :].reshape([b, -1]).unsqueeze(2)
    com_L = torch.bmm(torch.pinverse(sh_basis), S).squeeze(2)

    return com_L

def get_Chrom(face):
    #modify from  A closed-Form solution to Retinex with Nonlocal Texture Constraints
    size = face.shape
    new_size = [size[0], -1, size[2], size[3]]
    tp1 = face*face
    tp2 = torch.sum(tp1,1)
    tp3 = torch.reshape(tp2,new_size)
    tp4 = torch.max(tp3, 1)
    tp5 = tp4.values
    epsilon = (torch.ones(1)*1e-10).to(face.device)
    tp6 = torch.where(tp5 > epsilon, tp5, epsilon)
    tp7 = torch.reshape(torch.sqrt(tp6),new_size)
    intensity = torch.cat((tp7,tp7,tp7),1)
    #print('intensity...' + str(intensity))
    result = face / intensity
    return result


def test3T(vggL, colorEncoder, colorUNet, device):

    img = glob.glob('/media/hdr/oo/Dataset/Face/EN_data/3T/crop/*')
    for i in range(len(img)):
        val_img3 = img_transform(Image.open(img[i])).unsqueeze(0).to(device)
        val_preN = img_transform(Image.open(img[i].replace('crop','preN').replace('.png','_preN.png'))).unsqueeze(0).to(device)

        mask = img_transform(Image.open(img[i].replace('crop','mask').replace('img','mask'))).unsqueeze(0).to(device)
        unmask = torch.where(mask>0, 0, 1).to(device)

        b,c,w,h = val_img3.shape
        val_img3 = val_img3.to(device) 
        val_img3_grey = (val_img3[:,0,:,:]+val_img3[:,1,:,:]+val_img3[:,2,:,:])/3
        val_img3_grey = val_img3_grey.unsqueeze(1)

        val_preN = 2 * (val_preN.to(device) - 0.5)
        val_preN = F.normalize(val_preN)

        val_normal_feat = colorEncoder(val_preN)
        val_fine_normal = colorUNet((val_img3_grey, val_normal_feat))
        val_fine_normal = F.normalize(val_fine_normal)
        val_fine_normal_CN = N_SFS2CM(val_fine_normal)

        tpN = img[i].replace('.png', '_0.png').replace('crop', 'normal')
        svPh = tpN.replace('/media/hdr/oo/Dataset/Face/EN_data/3T/', vggL)
        mkdirss(svPh[: len(svPh) - len(svPh.split('/')[-1])])
        save_image(unmask+mask*get_normal_255(val_fine_normal_CN), svPh, nrow=1, normalize=True)


        tpN = img[i].replace('.png', '_0.png').replace('crop', 'face')
        svPh = tpN.replace('/media/hdr/oo/Dataset/Face/EN_data/3T/', vggL)
        mkdirss(svPh[: len(svPh) - len(svPh.split('/')[-1])])
        save_image(unmask+mask*val_img3, svPh, nrow=1, normalize=True)


        tt_n1 = normal_SFS2DPR(R_Z(R_X(R_Y(val_fine_normal_CN, angle180),angle180), angle270))

        shading0 = GetShading20211113(tt_n1, sh0).to(device)
        shading1 = GetShading20211113(tt_n1, sh1).to(device)
        shading2 = GetShading20211113(tt_n1, sh2).to(device)
        shading3 = GetShading20211113(tt_n1, sh3).to(device)
        shading4 = GetShading20211113(tt_n1, sh4).to(device)
        shading5 = GetShading20211113(tt_n1, sh5).to(device)
        shading6 = GetShading20211113(tt_n1, sh6).to(device)

        tpN = img[i].replace('.png', '_0.png').replace('crop', 'shading')
        svPh = tpN.replace('/media/hdr/oo/Dataset/Face/EN_data/3T/', vggL)
        mkdirss(svPh[: len(svPh) - len(svPh.split('/')[-1])])


        save_image(unmask+mask*shading0, svPh.replace('.png', '_0.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading1, svPh.replace('.png', '_1.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading2, svPh.replace('.png', '_2.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading3, svPh.replace('.png', '_3.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading4, svPh.replace('.png', '_4.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading5, svPh.replace('.png', '_5.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading6, svPh.replace('.png', '_6.png'), nrow=1, normalize=True)

        tpN = img[i].replace('.png', '_0.png').replace('crop', 'relighting')
        svPh = tpN.replace('/media/hdr/oo/Dataset/Face/EN_data/3T/', vggL)
        mkdirss(svPh[: len(svPh) - len(svPh.split('/')[-1])])

        save_image(unmask+mask*shading0*val_img3, svPh.replace('.png', '_0.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading1*val_img3, svPh.replace('.png', '_1.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading2*val_img3, svPh.replace('.png', '_2.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading3*val_img3, svPh.replace('.png', '_3.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading4*val_img3, svPh.replace('.png', '_4.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading5*val_img3, svPh.replace('.png', '_5.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading6*val_img3, svPh.replace('.png', '_6.png'), nrow=1, normalize=True)

        print()



def test3T_0401(datapath, svgpath, colorEncoder, colorUNet, device):

    img = glob.glob(datapath + '/*')
    for i in range(len(img)):
        imgN = img[i]
        t3_face = img_transform(Image.open(imgN)).unsqueeze(0).to(device)
        t3_preN = img_transform(Image.open(imgN.replace('img_','E85_img_').replace('imgs', 'preN'))).unsqueeze(0).to(device)
        mask = img_transform(Image.open(imgN.replace('img_','mask_').replace('imgs', 'mask'))).unsqueeze(0).to(device)
        unmask = torch.where(mask>0, 0*torch.ones(1).to(device), 1*torch.ones(1).to(device))

        t3_preN = 2 * (t3_preN - 0.5)
        t3_preN = F.normalize(t3_preN)

        t3_preN = t3_preN.to(device)
        t3_face = t3_face.to(device)

        t3_face_grey = (t3_face[:,0,:,:]+t3_face[:,1,:,:]+t3_face[:,2,:,:])/3
        t3_face_grey = t3_face_grey.unsqueeze(1)

        t3_preN_feat = colorEncoder(t3_preN)
        fine_normal = colorUNet((t3_face_grey, t3_preN_feat))
        fine_normal = F.normalize(fine_normal)
        vfine_normal_CN = N_SFS2CM(fine_normal)

        svPh = svgpath + '/normal_CN/' 
        mkdirss(svPh)
        save_image(unmask+mask*get_normal_255(vfine_normal_CN), svPh + img[i].split('/')[-1] , nrow=1, normalize=True)

        tt_n1 = normal_SFS2DPR(R_Z(R_X(R_Y(vfine_normal_CN, angle180),angle180), angle270))

        shading0 = GetShading20211113(tt_n1, sh0).to(device)
        shading1 = GetShading20211113(tt_n1, sh1).to(device)
        shading2 = GetShading20211113(tt_n1, sh2).to(device)
        shading3 = GetShading20211113(tt_n1, sh3).to(device)
        shading4 = GetShading20211113(tt_n1, sh4).to(device)
        shading5 = GetShading20211113(tt_n1, sh5).to(device)
        shading6 = GetShading20211113(tt_n1, sh6).to(device)

        svPh = svgpath + '/shading/' 
        mkdirss(svPh)

        save_image(unmask+mask*shading0, (svPh + img[i].split('/')[-1] ).replace('.png', '_0.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading1, (svPh + img[i].split('/')[-1] ).replace('.png', '_1.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading2, (svPh + img[i].split('/')[-1] ).replace('.png', '_2.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading3, (svPh + img[i].split('/')[-1] ).replace('.png', '_3.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading4, (svPh + img[i].split('/')[-1] ).replace('.png', '_4.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading5, (svPh + img[i].split('/')[-1] ).replace('.png', '_5.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading6, (svPh + img[i].split('/')[-1] ).replace('.png', '_6.png'), nrow=1, normalize=True)

        svPh = svgpath + '/relighting/' 
        mkdirss(svPh)

        save_image(unmask+mask*shading0*t3_face, (svPh + img[i].split('/')[-1] ).replace('.png', '_0.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading1*t3_face, (svPh + img[i].split('/')[-1] ).replace('.png', '_1.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading2*t3_face, (svPh + img[i].split('/')[-1] ).replace('.png', '_2.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading3*t3_face, (svPh + img[i].split('/')[-1] ).replace('.png', '_3.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading4*t3_face, (svPh + img[i].split('/')[-1] ).replace('.png', '_4.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading5*t3_face, (svPh + img[i].split('/')[-1] ).replace('.png', '_5.png'), nrow=1, normalize=True)
        save_image(unmask+mask*shading6*t3_face, (svPh + img[i].split('/')[-1] ).replace('.png', '_6.png'), nrow=1, normalize=True)

        print()


def test3T_0401_pre(datapath, svgpath, colorUNet, device):

    img = glob.glob(datapath + '/*')
    for i in range(len(img)):
        imgN = img[i]
        t3_face = img_transform(Image.open(imgN)).unsqueeze(0).to(device)
        t3_face = t3_face.to(device)

        t3_face_grey = (t3_face[:,0,:,:]+t3_face[:,1,:,:]+t3_face[:,2,:,:])/3
        t3_face_grey = t3_face_grey.unsqueeze(1)

        fine_normal = colorUNet(t3_face_grey)
        fine_normal = F.normalize(fine_normal)
        # vfine_normal_CN = N_SFS2CM(fine_normal)

        svPh = svgpath + '/normal_CN/' 
        mkdirss(svPh)
        save_image(get_normal_255(fine_normal), svPh + img[i].split('/')[-1] , nrow=1, normalize=True)



def CopyFiles(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件
        # print ("copy %s -> %s"%(srcfile, dstpath + fname))
