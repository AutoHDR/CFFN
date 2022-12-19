import argparse
from torch.utils.data import DataLoader
import os,glob

from PIL import Image

import torch

from torch import nn, optim
from torch.nn import functional as F

from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import UNet, SfSNet, Discriminator, SAGAN_Discriminator,SAGAN_DisSM, CycleGANGen
from data_loader import getPhotoDB_Pre_512, getPhotoDB_PreTrain
from torchvision.utils import save_image

from tools import get_Normal_Std_Mean, get_normal_N, get_normal_255, get_normal_P


def mkdirss(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

PIL2Tensor = transforms.Compose([
    transforms.ToTensor()
])

def train(
    args,
    train_data,
    test_data,
    DDA2PH,
    g_optim,
    device,
):
    loss_dict = {}
    lossflag = args.lossflag
    savepath = './P1_result/' + str(args.exp_name)

    logPath = savepath + '/' + 'log' + '/'
    mkdirss(logPath)
    writer = SummaryWriter(logPath)
    imgsPath = savepath + '/' + 'imgs' + '/'
    mkdirss(imgsPath)
    expPath = savepath + '/' + 'exp' + '/'
    mkdirss(expPath)
    CosLoss = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
    train_loader = sample_data(train_data)
    
    pbar = range(args.iter)
    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    criterion_GAN = nn.BCEWithLogitsLoss().to(device)#
    # D_F = SAGAN_DisSM().to(device)
    D_F = SAGAN_Discriminator().to(device)
    # D_model_path = '/home/xteam1/2022/HFFNE/pretrain/P1_result/UNet_W_Sagan/exp2/050000_DF.pkl'
    # D_F.load_state_dict(torch.load(D_model_path))
    optimizer_D_F = torch.optim.Adam(D_F.parameters(), lr=0.0001, weight_decay=0.0005)

    for idx in pbar:
        torch.cuda.empty_cache()
        i = idx + args.start_iter
        if i > args.iter:
            print("Done!")

            break

        ph_face, gt_n, mask, index = next(train_loader)
        
        b,c,w,h = ph_face.shape
        ph_face = ph_face.to(device)
        gt_n = gt_n.to(device)

        gtN = F.normalize(gt_n)

        DDA2PH.train()
        requires_grad(DDA2PH, True)
        
        # ph_face_grey = (ph_face[:,0,:,:]+ph_face[:,1,:,:]+ph_face[:,2,:,:])/3
        # ph_face_grey = ph_face_grey.unsqueeze(1)

        genN = DDA2PH(ph_face)
        genN = F.normalize(genN)  
        recon_N = torch.ones(1).to(device) - CosLoss(genN, gtN).mean()

        loss_dict["recon_N"] = recon_N

        D_real = D_F(gtN.detach())
        D_fake = D_F(genN.detach())
        valid =  torch.ones_like(D_real, requires_grad=False).to(device)
        fake =  torch.zeros_like(D_fake, requires_grad=False).to(device)

        optimizer_D_F.zero_grad()
        loss_real = criterion_GAN(D_real, valid)
        loss_fake = criterion_GAN(D_fake, fake)
        D_loss = loss_real + loss_fake
        D_loss.backward()
        optimizer_D_F.step()

        D_loss_F = args.refD * (criterion_GAN(D_F(genN), valid))

        LossTotal =  recon_N + D_loss_F
        g_optim.zero_grad()
        LossTotal.backward()
        g_optim.step()

        # print('i/len - {:5d}/{:5d} - reconN_loss:{:.5f}'.format(i, len(train_data), recon_N.item()))
        pbar.set_description(
            (
                f"i:{i:6d}; reconF:{recon_N.item():.4f}; R:{loss_real.item():.4f}; F:{loss_fake.item():.4f}; Dis_N:{D_loss_F.item():.4f}"
            )
        )
        torch.cuda.empty_cache()

        if i % 250 == 0:# and i >1:
            sampleImgs = torch.cat([ph_face, get_normal_255(gt_n), get_normal_255(genN)], 0)
            save_image(sampleImgs, imgsPath + '%d'%(25) + '_.png', nrow=b, normalize=True)
            
        if i % 50000 == 0 and i!=0:
            torch.save(DDA2PH.state_dict(),f"%s/{str(i).zfill(6)}_DDA2PH.pkl"%(expPath))
            torch.save(D_F.state_dict(),f"%s/{str(i).zfill(6)}_DF.pkl"%(expPath))

            with torch.no_grad():
                # img3t = glob.glob('/media/hdr/oo/Dataset/Face/EN_data/en3t-0401/imgs/*')
                # for i3t in range(len(img3t)):
                #     tpimg = img3t[i3t]
                #     tpimg_tensor = PIL2Tensor(Image.open(tpimg)).unsqueeze(0)
                #     tpimg_tensor = tpimg_tensor.to(device)
                #     # t3_grey = (tpimg_tensor[:,0,:,:]+tpimg_tensor[:,1,:,:]+tpimg_tensor[:,2,:,:])/3
                #     # t3_grey = t3_grey.unsqueeze(1)                        
                #     ph_fine_normal = DDA2PH(tpimg_tensor) #[-1, 1]
                #     ph_fine_normal = F.normalize(ph_fine_normal)
                #     save_image(get_normal_255(ph_fine_normal), logPath + str(i) + '_' +  tpimg.split('/')[-1], nrow=1, normalize=True)
                #     break
                testflag = 0
                sum_std = 0
                sum_mean = 0
                sum_n20 = 0
                sum_n25 = 0
                sum_n30 = 0
                DDA2PH.eval()
                for bix, batch in enumerate(test_data):
                    ph_face, gt_n, mask, index = batch
                    b, c, w, h = ph_face.shape
                    ph_face = ph_face.to(device)
                    mask = mask.to(device)

                    # ph_face_grey = (ph_face[:,0,:,:]+ph_face[:,1,:,:]+ph_face[:,2,:,:])/3
                    # ph_face_grey = ph_face_grey.unsqueeze(1)
                    gt_n = gt_n.to(device)
 
                    ph_fine_normal = DDA2PH(ph_face) #[-1, 1]
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
                    sampleImgs = torch.cat([ph_face, get_normal_255(gt_n), get_normal_255(ph_fine_normal)], 0)
                    save_image(sampleImgs, imgsPath + 'ph_%d'%(i) + '_.png', nrow=b, normalize=True)
                    torch.cuda.empty_cache()

                print("===> Avg. mean: {:.4f}, std:{:.4f}, n20:{:.4f}, n25:{:.4f}, n30:{:.4f}".format(sum_mean / testflag, sum_std / testflag, sum_n20 / testflag, sum_n25 / testflag, sum_n30 / testflag))
                with open(logPath +'mean_std.txt', 'a+') as f:
                    details = str(i) + '\t' + str(sum_mean / testflag) + '\t' + str(sum_std /testflag) + '\t' + str(sum_n20 /testflag) + '\t' + str(sum_n25 /testflag) + '\t' + str(sum_n30 /testflag) + '\n'
                    f.write(details)
        torch.cuda.empty_cache()


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str)
    parser.add_argument("--iter", type=int, default=200000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--ckpt", type=str, default=None)
    # parser.add_argument("--ckpt", type=str, default='/home/xteam1/2022/HFFNE/pretrain/P1_result/UNet_W_Sagan/exp2/050000_DDA2PH.pkl')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--exp_name", type=str, default="CycleGAN_SAGAN_8020")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--lossflag", type=int, default=0)
    parser.add_argument("--refD", type=float, default=0.0001)
    parser.add_argument("--gpuID", type=int, default=1)

    args = parser.parse_args()

    device = "cuda:" + str(args.gpuID)

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    args.start_iter = 0
    DDA2PH = CycleGANGen(3, 3).to(device)
    # DDA2PH = SfSNet().to(device)

    g_optim = optim.Adam(
        list(DDA2PH.parameters()),
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
        
        DDA2PH.load_state_dict(ckpt)
        # g_optim.load_state_dict(ckpt["g_optim"])

    pathd = '/home/xteam1/2022/HFFNE/data/csv/p1_PF_train.csv'
    train_dataset, val_dataset = getPhotoDB_PreTrain(csvPath=pathd, IMAGE_SIZE=256, validation_split=0)
    syn_train_dl  = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)

    pathd = '/home/xteam1/2022/HFFNE/data/csv/p1_PF_test.csv'
    val_dataset, _ = getPhotoDB_PreTrain(csvPath=pathd, IMAGE_SIZE=256, validation_split=0)
    syn_val_dl  = DataLoader(val_dataset, batch_size=args.batch, shuffle=True)
    train(
        args,
        syn_train_dl,
        syn_val_dl,
        DDA2PH,
        g_optim,
        device,
    )

