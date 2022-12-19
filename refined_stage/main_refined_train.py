
import argparse
from torch.utils.data import Dataset, DataLoader
import os, glob
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
# from NormalEncoder import NormalEncoder
from models import *
from vgg_model import vgg19
from data_loader import *
from torchvision.utils import save_image
import utils
from tools import get_Normal_Std_Mean, CopyFiles, get_normal_255, get_shading_DPR_B
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.autograd import Variable


def mkdirss(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

from matplotlib import cm
def colormap(error,thres=90):
    error_norm = np.clip(error, 0, thres) / thres
    error_map = cm.jet(error_norm)[:,:,0:3]
    return np.uint8(255*error_map)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def N_SFS2DPR(normal):
    tt = torch.zeros_like(normal)
    tt[:, 0, :, :] = -normal[:, 2, :, :]
    tt[:, 1, :, :] = normal[:, 0, :, :]
    tt[:, 2, :, :] = normal[:, 1, :, :]
    return tt

def uncenter_l(inputs):
    l = inputs[:,:1,:,:] + 50
    ab = inputs[:,1:,:,:]
    return torch.cat((l, ab), 1)
l0 = '/home/xteam1/2022/model_data/example_light/DPR/rotate_light_00.txt'
l1 = '/home/xteam1/2022/model_data/example_light/DPR/rotate_light_01.txt'
l2 = '/home/xteam1/2022/model_data/example_light/DPR/rotate_light_02.txt'
l3 = '/home/xteam1/2022/model_data/example_light/DPR/rotate_light_03.txt'
l4 = '/home/xteam1/2022/model_data/example_light/DPR/rotate_light_04.txt'
l5 = '/home/xteam1/2022/model_data/example_light/DPR/rotate_light_05.txt'
l6 = '/home/xteam1/2022/model_data/example_light/DPR/rotate_light_06.txt'
pd_sh0 = pd.read_csv(l0, sep='\t', header=None, encoding=u'gbk')
pd_sh1 = pd.read_csv(l1, sep='\t', header=None, encoding=u'gbk')
pd_sh2 = pd.read_csv(l2, sep='\t', header=None, encoding=u'gbk')
pd_sh3 = pd.read_csv(l3, sep='\t', header=None, encoding=u'gbk')
pd_sh4 = pd.read_csv(l4, sep='\t', header=None, encoding=u'gbk')
pd_sh5 = pd.read_csv(l5, sep='\t', header=None, encoding=u'gbk')
pd_sh6 = pd.read_csv(l6, sep='\t', header=None, encoding=u'gbk')
def train(
    args,
    train_dl,
    val_dl,
    ph_val_dl,
    NormalEncoder,
    NormalRefNet,
    vggnet,
    g_optim,
    device,
):
    train_loader = sample_data(train_dl)
    pbar = range(args.iter + 1)

    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    g_loss_val = 0
    loss_dict = {}
    CosLoss = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)


    NormalEncoder_module = NormalEncoder
    NormalRefNet_module = NormalRefNet

    lossflag = args.lossflag
    savepath = '/home/xteam1/2022/HFFNE/result/MM_Ext/' + args.exp_name
    weight_path = savepath + '/'
    logPath = weight_path + 'log' + '/'
    mkdirss(logPath)
    logPathcodes = weight_path + 'codes' + '/'
    mkdirss(logPathcodes)
    writer = SummaryWriter(logPath)
    imgsPath = weight_path + 'imgs' + '/'
    mkdirss(imgsPath)
    expPath = weight_path + 'exp' + '/'
    mkdirss(expPath)

    #backup codes
    src_dir = './'
    src_file_list = glob.glob(src_dir + '*')                    # glob获得路径下所有文件，可根据需要修改
    for srcfile in src_file_list:
        CopyFiles(srcfile, logPathcodes)                       # 复制文件
    print('copy codes have done!!!')
    
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        face, coNorm, index = next(train_loader)
        # face, coNorm, randN, index = next(train_loader)

        b,c,w,h = face.shape
        face = face.to(device) 
        face_grey = (face[:,0,:,:]+face[:,1,:,:]+face[:,2,:,:])/3
        face_grey = face_grey.unsqueeze(1)
        coNorm = coNorm.to(device)
        coNorm = F.normalize(coNorm)

        NormalEncoder.train()
        NormalRefNet.train()

        requires_grad(NormalEncoder, True)
        requires_grad(NormalRefNet, True)
        
        normal_feat = NormalEncoder(coNorm)
        fine_normal = NormalRefNet((face_grey, normal_feat)) # [-1, 1]
        fine_normal = F.normalize(fine_normal)
        

        # recon_F = (1 - CosLoss(fine_normal, coNorm).mean()) * args.refNorm
        recon_F = (F.smooth_l1_loss(fine_normal, coNorm)) * args.refNorm

        ## feature loss
        features_A = vggnet(face_grey, layer_name='all')
        features_B = vggnet(fine_normal, layer_name='all')

        fea_loss1 = F.l1_loss(features_A[0], features_B[0]) / 32

        fea_loss = (fea_loss1) * args.refVGG
        # my_ssimloss = (1 - ssim_loss(face, (fine_normal/2 + 0.5))) * args.refST

        writer.add_scalar('recon_F', recon_F.item(), i)
        writer.add_scalar('fea_loss', fea_loss.item(), i)

        # print('******************************************************')
        # print('******************************************************')
        # print('******************************************************')
        LossTotal =  recon_F + fea_loss
        # LossTotal = fea_loss # abs_1 without recon
        # print('******************************************************')
        # print('******************************************************')
        # print('******************************************************')

        g_optim.zero_grad()
        LossTotal.backward()
        g_optim.step()

        # torch.cuda.empty_cache()
        
        pbar.set_description(
            (
                f"i:{i:6d}; reconF:{recon_F.item():.4f}; fea:{fea_loss.item():.4f};  "
            )
        )
        torch.cuda.empty_cache()

        if i % 100 == 0:
            NormalEncoder.eval()
            NormalRefNet.eval()
            for bix, batch in enumerate(val_dl):
                vface, vcoNorm, vindex = batch
                b,c,w,h = vface.shape
                vface = vface.to(device) 
                vface_grey = (vface[:,0,:,:]+vface[:,1,:,:]+vface[:,2,:,:])/3
                vface_grey = vface_grey.unsqueeze(1)
                vcoNorm = vcoNorm.to(device)
                vcoNorm = F.normalize(vcoNorm)
                normal_feat = NormalEncoder(vcoNorm)
                fine_normal = NormalRefNet((vface_grey, normal_feat)) # [-1, 1]
                fine_normal = F.normalize(fine_normal)
                val_fine_normal_CN = N_SFS2DPR(fine_normal)
                
                sh0 = torch.tensor(pd_sh0.values).type(torch.float).reshape([1, 9]).expand([b, 9]).to(device)
                sh1 = torch.tensor(pd_sh1.values).type(torch.float).reshape([1, 9]).expand([b, 9]).to(device)
                sh2 = torch.tensor(pd_sh2.values).type(torch.float).reshape([1, 9]).expand([b, 9]).to(device)
                sh3 = torch.tensor(pd_sh3.values).type(torch.float).reshape([1, 9]).expand([b, 9]).to(device)
                sh4 = torch.tensor(pd_sh4.values).type(torch.float).reshape([1, 9]).expand([b, 9]).to(device)
                sh5 = torch.tensor(pd_sh5.values).type(torch.float).reshape([1, 9]).expand([b, 9]).to(device)
                sh6 = torch.tensor(pd_sh6.values).type(torch.float).reshape([1, 9]).expand([b, 9]).to(device)

                shading0 = get_shading_DPR_B(val_fine_normal_CN, sh0).expand([b,3,w,h])
                shading1 = get_shading_DPR_B(val_fine_normal_CN, sh1).expand([b,3,w,h])
                shading2 = get_shading_DPR_B(val_fine_normal_CN, sh2).expand([b,3,w,h])
                shading3 = get_shading_DPR_B(val_fine_normal_CN, sh3).expand([b,3,w,h])
                shading4 = get_shading_DPR_B(val_fine_normal_CN, sh4).expand([b,3,w,h])
                shading5 = get_shading_DPR_B(val_fine_normal_CN, sh5).expand([b,3,w,h])
                shading6 = get_shading_DPR_B(val_fine_normal_CN, sh6).expand([b,3,w,h])

                sampleImgs = torch.cat([vface, get_normal_255(vcoNorm), get_normal_255(fine_normal), shading0, shading1, shading2,shading3, shading4, shading5, shading6], 0)
                save_image(sampleImgs, imgsPath + 'val.png', nrow=b, normalize=True)
        if i % 25000 == 0 and idx!=0:
            torch.save(
                {
                    "NormalEncoder": NormalEncoder_module.state_dict(),
                    "NormalRefNet": NormalRefNet_module.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "args": args,
                },
                f"%s/{str(i).zfill(6)}.pt"%(expPath),
            )
            with torch.no_grad():
                testflag = 0
                sum_std = 0
                sum_mean = 0
                sum_n20 = 0
                sum_n25 = 0
                sum_n30 = 0
                NormalEncoder.eval()
                NormalRefNet.eval()
                for bix, batch in enumerate(ph_val_dl):
                    ph_face, pre_norm, gt_n, mask, index = batch
                    # ph_face, gt_n, pre_norm, randomNN, mask, index = batch
                    b, c, w, h = ph_face.shape
                    ph_face = ph_face.to(device)
                    ph_face_grey = (ph_face[:,0,:,:]+ph_face[:,1,:,:]+ph_face[:,2,:,:])/3
                    ph_face_grey = ph_face_grey.unsqueeze(1)
                    pre_norm = pre_norm.to(device)
                    mask = mask.to(device)
                    gt_n = F.normalize(gt_n).to(device)
                    pre_norm = F.normalize(pre_norm)

                    ph_norm_feat = NormalEncoder(pre_norm) 
                    ph_fine_normal = NormalRefNet((ph_face_grey, ph_norm_feat)) #[-1, 1]
                    ph_fine_normal = F.normalize(ph_fine_normal)

                    std, mean, n20, n25, n30 =  get_Normal_Std_Mean(ph_fine_normal, gt_n, mask)
                    sum_std = sum_std + std.item()
                    sum_mean = sum_mean + mean.item()
                    sum_n20 += n20.item()
                    sum_n25 += n25.item()
                    sum_n30 += n30.item()
                    testflag += 1  

                    lossflag += 1
                    LOn = F.smooth_l1_loss(ph_fine_normal, gt_n)
                    writer.add_scalar('LOn', LOn.item(), lossflag)
                    sampleImgs = torch.cat([ph_face, get_normal_255(gt_n), get_normal_255(pre_norm), get_normal_255(ph_fine_normal)], 0)
                    if bix == 50:
                        save_image(sampleImgs, imgsPath + 'ph_%d'%(bix)+'_%d'%(i) + '_.png', nrow=b, normalize=True)
                        masknp = mask.cpu().numpy()
                        errorMap = (CosLoss(ph_fine_normal, gt_n)).cpu().numpy()
                        
                        in_err = 0
                        tperr = np.arccos(errorMap[in_err,:,:])/3.1415926 * 180
                        mkerr = np.stack((masknp[in_err, 1,:,:],masknp[in_err, 1,:,:],masknp[in_err, 1,:,:]),axis=2)
                        unmkerr = np.where(mkerr<0.0001, 1, 0) * 255
                        tp = colormap(tperr * masknp[in_err, 1,:,:], thres=45) * mkerr + unmkerr
                        tpimg = Image.fromarray(np.uint8(tp)).convert('RGB')
                        tpimg.save(imgsPath + 'er_0.png')
                    # torch.cuda.empty_cache()
                    # print(bix)
                print("===> Avg. mean: {:.4f}, std:{:.4f}, n20:{:.4f}, n25:{:.4f}, n30:{:.4f}".format(sum_mean / testflag, sum_std / testflag, sum_n20 / testflag, sum_n25 / testflag, sum_n30 / testflag))
                with open(logPath +'mean_std_'+ str(args.refNorm) + '_' + str(args.refVGG) + '.txt', 'a+') as f:
                    details = str(i) + '\t' + str(sum_mean / testflag) + '\t' + str(sum_std /testflag) + '\t' + str(sum_n20 /testflag) + '\t' + str(sum_n25 /testflag) + '\t' + str(sum_n30 /testflag) + '\n'
                    f.write(details)  
                # torch.cuda.empty_cache()
 


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str)
    parser.add_argument("--iter", type=int, default=150000) 
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--ckpt", type=str, default=None)
    # parser.add_argument("--ckpt", type=str, default="/home/xteam1/2022/HFFNE/result/MM_Ext/SAGAN_FPN256_P2_WO_D/exp/050000.pt")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--exp_name", type=str, default="CYSAGan_FPN_P2_WO_D_300w_FFHQ")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--lossflag", type=int, default=0)
    parser.add_argument("--refNorm", type=float, default=1)
    parser.add_argument("--refVGG", type=int, default=0.1)
    parser.add_argument("--refDN", type=float, default=5)
    parser.add_argument("--gpuID", type=int, default=0)
    parser.add_argument("--refST", type=float, default=0.01)


    args = parser.parse_args()
    device = "cuda:" + str(args.gpuID)

    args.start_iter = 0

    vggnet = vgg19(pretrained_path = '/home/xteam1/2022/HFFNE/refine/data/vgg19-dcbb9e9d.pth', require_grad = False)
    vggnet = vggnet.to(device)
    vggnet.eval()

    # #----------------------NormalEncoder_FPN_ATT-------------
    NormalEncoder = NormalEncoder_FPN(color_dim=3, out_dim=256).to(device)
    NormalRefNet = NormalRefNet(bilinear=False).to(device)
    # #-------------------------------------------------------


    g_optim = optim.Adam(
        list(NormalEncoder.parameters()) + list(NormalRefNet.parameters()),
        lr=args.lr,
        betas=(0.9, 0.99),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass
        
        NormalEncoder.load_state_dict(ckpt["NormalEncoder"])
        NormalRefNet.load_state_dict(ckpt["NormalRefNet"])
        g_optim.load_state_dict(ckpt["g_optim"])


    datasets = []

    pathd = '/home/xteam1/2022/HFFNE/data/csv/p2_300w_FFHQ_phdb_train.csv'
    train_dataset, _ = get_300W_phdb_patch(csvPath=pathd, validation_split=0)
    train_dl  = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=8)

    pathd = '/home/xteam1/2022/HFFNE/data/csv/FFHQ_test_6imgs.csv'
    train_dataset, _ = get_300W_phdb_patch(csvPath=pathd, validation_split=0)
    val_dl  = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=8)


    pathd = '/home/xteam1/2022/HFFNE/data/csv/p1_PF_test.csv'
    val_dataset, _ = getPhotoDB_Pre_Att_patch(csvPath=pathd, validation_split=0)
    ph_val_dl  = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=8)

    train(
        args,
        train_dl,
        val_dl,
        ph_val_dl,
        NormalEncoder,
        NormalRefNet,
        vggnet,
        g_optim,
        device,
    )

