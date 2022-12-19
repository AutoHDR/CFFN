import pandas as pd
import numpy as np
from PIL import Image
import torch, random
from torch.utils.data import Dataset, random_split
from torchvision import transforms


IMAGE_SIZE = 256
IMAGE_SIZE_128 = 128

def get_300W_phdb(csvPath=None, validation_split=0):
    df = pd.read_csv(csvPath)
    face = list(df['face'])
    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    full_dataset = Dataset_300W_phdb(face, transform)
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class Dataset_300W_phdb(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)

    def __getitem__(self, index):
        face_path = self.face[index]
        face = self.transform(Image.open(face_path))
        normal_path = self.face[index].replace('.png', '_preN_Att.png')
        normal = self.transform(Image.open(normal_path))
        normal = 2 * (normal - 0.5)
        ranNorm = self.transform(Image.open(self.face[random.randint(0, self.dataset_len-1)].replace('.png', '_preN_Att.png')))
        ranNorm = 2 * (ranNorm - 0.5)

        mask = self.transform(Image.open(normal_path))
        if face_path.split('/')[-3] == 'Img1024':
            mask = self.transform(Image.open(face_path.replace('.png', '_mask.png')))
            # print(face_path.replace('.png', '_mask.png'))
            # print()
        if face_path.split('/')[6] == '01_Indoor' or face_path.split('/')[6] == '02_Outdoor':
            mask = self.transform(Image.open(face_path.replace('.png', '_mask.png')))
            # print(face_path.replace('.png', '_mask.png'))
            # print()
        if face_path.split('/')[6] == 'PhotofaceCrop':
            mask = self.transform(Image.open(face_path[:77] +  'mask_c.png'))
            # print(face_path[:77] +  'mask_c.png')
            # print()
        return face, normal, ranNorm, mask, normal_path

    def __len__(self):
        return self.dataset_len


def get_300W_phdb_patch(csvPath=None, validation_split=0):
    df = pd.read_csv(csvPath)
    face = list(df['face'])
    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    full_dataset = Dataset_300W_phdb_patch(face, transform)
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class Dataset_300W_phdb_patch(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)

    def __getitem__(self, index):
        face_path = self.face[index]
        face = self.transform(Image.open(face_path))
        savgflag = '_preN_cysagan.png'

        normal_path = self.face[index].replace('.png', savgflag)
        normal = self.transform(Image.open(normal_path))
        normal = 2 * (normal - 0.5)
        return face, normal, normal_path

    def __len__(self):
        return self.dataset_len


#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

def getCelebA_P3_512(csvPath=None, validation_split=0, IMG_SIZE=256):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])
    full_dataset = DatasetCelebA_P3(face, transform)

    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset


class DatasetCelebA_P3(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)

    def __getitem__(self, index):
        face_path = self.face[index]
        face = self.transform(Image.open(face_path))

        randfaceID = self.face[random.randint(0, self.dataset_len - 1)]
        Randface = self.transform(Image.open(randfaceID))

        tp_s = face_path.split('/')
        normal_path = ''

        if tp_s[6] == 'CelebA512':
            normal_path = self.face[index].replace('imgs', 'Test_512_W_resnet_1450000')
        else:
            normal_path = self.face[index].replace('Imgs512', 'PF512_Test_512_W_resnet_1450000')

        normal = self.transform(Image.open(normal_path))
        normal = 2 * (normal - 0.5)


        return face, normal, normal_path, Randface, randfaceID

    def __len__(self):
        return self.dataset_len


def getCelebA_P3_512(csvPath=None, validation_split=0, IMG_SIZE=256):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])
    full_dataset = DatasetCelebA_P3(face, transform)

    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset


class DatasetCelebA_P3(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)

    def __getitem__(self, index):
        face_path = self.face[index]
        face = self.transform(Image.open(face_path))
        
        randfaceID = self.face[random.randint(0, self.dataset_len - 1)]
        Randface = self.transform(Image.open(randfaceID))

        tp_s = face_path.split('/')

        normal_path = self.face[index].replace('imgs', 'Test_512_W_resnet_1450000')
  
        tp = '/media/xteam1/oo/Datasets/PhotofaceDB/Imgs512/1001/2008-02-23_12-21-31/normal_crop_75.png'
        tpn = self.transform(Image.open(tp))
        tpn = 2 * (tpn - 0.5)

        normal = self.transform(Image.open(normal_path))
        normal = 2 * (normal - 0.5)


        return face, normal, normal_path, Randface, randfaceID, tpn

    def __len__(self):
        return self.dataset_len
        
#-------------------------- start photoface dataset ---------------------
def getPhotoDB(csvPath=None, validation_split=0):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    full_dataset = PhotoDB(face, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train.py-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class PhotoDB(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)
        self.DataTrans = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __getitem__(self, index):
        face_path = self.face[index]

        face = self.DataTrans(Image.open(face_path))

        tpimg = face_path.split('/')[-1]
        gt_path = face_path.replace(tpimg, 'sn_c.png')
        gPN = self.DataTrans(Image.open(gt_path))
        gtN = 2 * (gPN - 0.5)

        mask = torch.where(gtN[0,:,:]==1, 0, 1).unsqueeze(0).expand(3,256,256)
        return face, gtN, mask, face_path

    def __len__(self):
        return self.dataset_len
#-------------------------- end hotoface dataset ---------------------

#-------------------------- start p1 ffhq dataset ---------------------
def getFFHQ2Pre(csvPath=None, validation_split=0):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    full_dataset = DatasetFFHQ2Pre(face, transform)

    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset


class DatasetFFHQ2Pre(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)
        self.DataTrans = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        face_path = self.face[index]
        face = self.DataTrans(Image.open(face_path))

        return face, face_path

    def __len__(self):
        return self.dataset_len
#-------------------------- end p1 ffhq dataset ---------------------
        


#-------------------------- start p2 ffhq dataset ---------------------
def get_P2Data(csvPath=None, validation_split=0):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    full_dataset = DatasetP2Train(face, transform)

    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset


class DatasetP2Train(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)
        self.DataTrans = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        face_path = self.face[index]
        face = self.DataTrans(Image.open(face_path))
        
        epochflag = '_preN.png'
        pre_path = face_path.replace('.png', epochflag)
        preN = self.DataTrans(Image.open(pre_path))
        preN = 2 * (preN - 0.5)

        return face, preN, face_path

    def __len__(self):
        return self.dataset_len
#-------------------------- end p2 ffhq dataset ---------------------
        
def getPhotoDB_Pre1(csvPath=None, validation_split=0):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    full_dataset = PhotoDB_Pre1(face, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train.py-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class PhotoDB_Pre1(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)
        self.DataTrans = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        face_path = self.face[index]
        face = self.DataTrans(Image.open(face_path))

        tpimg = face_path.split('/')[-1]
        gt_path = face_path.replace(tpimg, 'sn_c.png')
        gPN = self.DataTrans(Image.open(gt_path))
        gtN = 2 * (gPN - 0.5)

        epochflag = '_preN.png'
        pre_path = face_path.replace('.png', epochflag)
        preN = self.DataTrans(Image.open(pre_path))
        preN = 2 * (preN - 0.5)

        mask_path = face_path.replace(tpimg, 'mask_c.png')
        mask = self.DataTrans(Image.open(mask_path))

        randomN_path = self.face[np.random.randint(0,self.dataset_len-1)]
        tpimgR = randomN_path.split('/')[-1]
        Rn_path = randomN_path.replace('.png', epochflag)
        randomNN = self.DataTrans(Image.open(Rn_path))
        randomNN = 2 * (randomNN - 0.5)

        # print(face_path)
        # print(gt_path)
        # print(pre_path)
        # print(Rn_path)
        # print(mask_path)
        return face, gtN, preN, randomNN, mask, face_path

    def __len__(self):
        return self.dataset_len



################################old#######################################################


 ################################start split train pf data#######################################################
       
def getPhotoDB_SP(csvPath=None, validation_split=0):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    full_dataset = PhotoDB_SP(face, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train.py-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class PhotoDB_SP(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)
        self.DataTrans = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        face_path = self.face[index]
        face = self.DataTrans(Image.open(face_path))

        tpimg = face_path.split('/')[-1]
        gt_path = face_path.replace(tpimg, 'sn_c.png')
        gPN = self.DataTrans(Image.open(gt_path))
        gtN = 2 * (gPN - 0.5)

        epochflag = '_preN.png'
        pre_path = face_path.replace('.png', epochflag)
        preN = self.DataTrans(Image.open(pre_path))
        preN = 2 * (preN - 0.5)

        mask_path = face_path.replace(tpimg, 'mask_c.png')
        mask = self.DataTrans(Image.open(mask_path))

        randomN_path = self.face[np.random.randint(0,self.dataset_len-1)]
        tpimgR = randomN_path.split('/')[-1]
        Rn_path = randomN_path.replace('.png', epochflag)
        randomNN = self.DataTrans(Image.open(Rn_path))
        randomNN = 2 * (randomNN - 0.5)

        # print(face_path)
        # print(gt_path)
        # print(pre_path)
        # print(Rn_path)
        # print(mask_path)
        return face, gtN, preN, randomNN, mask, face_path

    def __len__(self):
        return self.dataset_len



################################end split train pf data#######################################################



###################    128  


#-------------------------- start p2 ffhq dataset ---------------------
def get_P2Data_128(csvPath=None, validation_split=0):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE_128),
        transforms.ToTensor()
    ])
    full_dataset = DatasetP2Train_128(face, transform)

    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset


class DatasetP2Train_128(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)
        self.DataTrans = transforms.Compose([
            transforms.Resize(IMAGE_SIZE_128),
            transforms.ToTensor()
        ])


    def __getitem__(self, index):
        face_path = self.face[index]
        face = self.DataTrans(Image.open(face_path))
        
        epochflag = '_preN.png'
        pre_path = face_path.replace('.png', epochflag)
        preN = self.DataTrans(Image.open(pre_path))
        preN = 2 * (preN - 0.5)

        return face, preN, face_path

    def __len__(self):
        return self.dataset_len


      
def getPhotoDB_Pre_128(csvPath=None, validation_split=0):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE_128),
        transforms.ToTensor()
    ])

    full_dataset = PhotoDB_SP_128(face, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train.py-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class PhotoDB_SP_128(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)
        self.DataTrans = transforms.Compose([
            transforms.Resize(IMAGE_SIZE_128),
            transforms.ToTensor()
        ])


    def __getitem__(self, index):
        face_path = self.face[index]
        face = self.DataTrans(Image.open(face_path))

        tpimg = face_path.split('/')[-1]
        gt_path = face_path.replace(tpimg, 'sn_c.png')
        gPN = self.DataTrans(Image.open(gt_path))
        gtN = 2 * (gPN - 0.5)

        epochflag = '_preN.png'
        pre_path = face_path.replace('.png', epochflag)
        preN = self.DataTrans(Image.open(pre_path))
        preN = 2 * (preN - 0.5)

        mask_path = face_path.replace(tpimg, 'mask_c.png')
        mask = self.DataTrans(Image.open(mask_path))

        randomN_path = self.face[np.random.randint(0,self.dataset_len-1)]
        tpimgR = randomN_path.split('/')[-1]
        Rn_path = randomN_path.replace('.png', epochflag)
        randomNN = self.DataTrans(Image.open(Rn_path))
        randomNN = 2 * (randomNN - 0.5)

        # print(face_path)
        # print(gt_path)
        # print(pre_path)
        # print(Rn_path)
        # print(mask_path)
        return face, gtN, preN, randomNN, mask, face_path

    def __len__(self):
        return self.dataset_len


#-------------------------- end p2 ffhq dataset ---------------------
 


def get_2D3dData_128(csvPath=None, validation_split=0):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    transform = transforms.Compose([
        # transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])
    full_dataset = Dataset2D3D(face, transform)

    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset


class Dataset2D3D(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)
        self.DataTrans = transforms.Compose([
            # transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        face_path = self.face[index]
        face = self.DataTrans(Image.open(face_path))


        normal_path = self.face[index].replace('imgs', 'normal')
        normal = self.DataTrans(Image.open(normal_path))
        normal = 2 * (normal - 0.5)


        return face, normal, face_path

    def __len__(self):
        return self.dataset_len

def getCelebA_P2_512(csvPath=None, validation_split=0, IMG_SIZE=256):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])
    full_dataset = DatasetCelebA_P2(face, transform)

    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset


class DatasetCelebA_P2(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)

    def __getitem__(self, index):
        face_path = self.face[index]
        face = self.transform(Image.open(face_path))

        normal_path = ''

        tp = face_path.split('/')
        if tp[6]=='Imgs512':
            normal_path = self.face[index].replace('.png', '_preN_sagan.png')
        else:
            normal_path = self.face[index].replace('imgs', 'C_Normal')

        normal = self.transform(Image.open(normal_path))
        normal = 2 * (normal - 0.5)


        return face, normal, face_path

    def __len__(self):
        return self.dataset_len




#-------------------------- start photoface dataset ---------------------
def getPhotoDB_Pre_512(csvPath=None, validation_split=0, IMG_SIZE=256):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

    full_dataset = PhotoDB_Pre_512(face, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train.py-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class PhotoDB_Pre_512(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)

    def __getitem__(self, index):
        face_path = self.face[index]

        face = self.transform(Image.open(face_path))

        gt_path = ''
        if(((face_path.split('/')[-1]).split('.')[0]).split('_')[-1] == '50'):
            gt_path = face_path[:-15] + 'normal_crop_50.png'
        if(((face_path.split('/')[-1]).split('.')[0]).split('_')[-1] == '75'):
            gt_path = face_path[:-15] + 'normal_crop_75.png'
        if(((face_path.split('/')[-1]).split('.')[0]).split('_')[-1] == '100'):
            gt_path = face_path[:-16] + 'normal_crop_100.png'
        if(((face_path.split('/')[-1]).split('.')[0]).split('_')[-1] == '125'):
            gt_path = face_path[:-16] + 'normal_crop_125.png'

        gPN = self.transform(Image.open(gt_path))
        gtN = 2 * (gPN - 0.5)

        Pre_N = self.transform(Image.open(face_path.replace('.png', '_preN.png')))
        Pre_N = 2 * (Pre_N - 0.5)

        c, w, h = Pre_N.shape
        mask = torch.where(gtN[0,:,:]==1, 0, 1).unsqueeze(0).expand(3, w, h)
        return face, Pre_N, gtN, mask, face_path

    def __len__(self):
        return self.dataset_len
#-------------------------- end hotoface dataset ---------------------


#-------------------------- start photoface dataset ---------------------
def getPhotoDB_Pre_Att(csvPath=None, validation_split=0):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    full_dataset = PhotoDB_Pre_Att(face, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train.py-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class PhotoDB_Pre_Att(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)

    def __getitem__(self, index):
        face_path = self.face[index]

        face = self.transform(Image.open(face_path))
        tps = face_path.split('/')
        if len(tps[-1]) == 12:
            gPN = self.transform(Image.open(face_path[:-12] + 'sn_c.png'))
            mkpath = Image.open(face_path[:-12] + 'mask_c.png')
        else:
            gPN = self.transform(Image.open(face_path[:-9] + 'sn_c.png'))
            mkpath = Image.open(face_path[:-9] + 'mask_c.png')

        gtN = 2 * (gPN - 0.5)

        gPN = self.transform(Image.open(face_path.replace('.png', '_preN_Att.png')))
        coN = 2 * (gPN - 0.5)

        c, w, h = gtN.shape
        # mask = torch.where(gtN[0,:,:]==1, 0, 1).unsqueeze(0).expand(3, w, h)
        mask = self.transform(mkpath)

        return face, coN, gtN, mask, face_path

    def __len__(self):
        return self.dataset_len



#-------------------------- start photoface dataset ---------------------
def getPhotoDB_Pre_Att_patch(csvPath=None, validation_split=0):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    full_dataset = PhotoDB_Pre_Att_patch(face, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train.py-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class PhotoDB_Pre_Att_patch(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)

    def __getitem__(self, index):
        face_path = self.face[index]

        face = self.transform(Image.open(face_path))
        tps = face_path.split('/')
        if len(tps[-1]) == 12:
            gPN = self.transform(Image.open(face_path[:-12] + 'sn_c.png'))
            mkpath = Image.open(face_path[:-12] + 'mask_c.png')
        else:
            gPN = self.transform(Image.open(face_path[:-9] + 'sn_c.png'))
            mkpath = Image.open(face_path[:-9] + 'mask_c.png')

        gtN = 2 * (gPN - 0.5)
        savgflag = '_preN_cysagan.png'

        gPN = self.transform(Image.open(face_path.replace('.png', savgflag)))
        coN = 2 * (gPN - 0.5)

        c, w, h = gtN.shape
        # mask = torch.where(gtN[0,:,:]==1, 0, 1).unsqueeze(0).expand(3, w, h)
        mask = self.transform(mkpath)

        return face, coN, gtN, mask, face_path

    def __len__(self):
        return self.dataset_len

################################old#######################################################

def get_Pre_trainData_50(csvPath=None, validation_split=0):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    full_dataset = Dataset_Pre_train_50(face, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train.py-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class Dataset_Pre_train_50(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)
        self.DataTrans = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        face_path = self.face[index]
        face = self.DataTrans(Image.open(face_path))

        # epochflag = '_pre_b75.png'
        epochflag = '_preN.png'

        preN_path = face_path.replace('.png', epochflag)
        pre_norm = self.DataTrans(Image.open(preN_path))
        pre_norm = 2 * (pre_norm - 0.5)


        random_path = self.face[np.random.randint(0,self.dataset_len-1)].replace('.png', epochflag)
        randonN = self.DataTrans(Image.open(random_path))
        randonN = 2 * (randonN - 0.5)

        # print(face_path)
        # print(preN_path)
        # print(random_path)

        return face, pre_norm, randonN, face_path

    def __len__(self):
        return self.dataset_len



def get_PhotoDB_50(csvPath=None, validation_split=0):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    full_dataset = Dataset_PhotoDB_50(face, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train.py-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class Dataset_PhotoDB_50(Dataset):
    def __init__(self, face, transform=None):

        self.face = face
        self.transform = transform
        self.dataset_len = len(self.face)
        self.DataTrans = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        face_path = self.face[index]
        imgname = face_path.split('/')[-1]
        if imgname == 'albedo_c.png':
            face_path = face_path.replace('imgname', 'im1_c.png')
        face = self.DataTrans(Image.open(face_path))

        tpimg = face_path.split('/')[-1]
        gt_path = face_path.replace(tpimg, 'sn_c.png')
        gPN = self.DataTrans(Image.open(gt_path))
        gtN = 2 * (gPN - 0.5)

        epochflag = '_preN.png'
        pre_path = face_path.replace('.png', epochflag)
        preN = self.DataTrans(Image.open(pre_path))
        preN = 2 * (preN - 0.5)

        mask_path = face_path.replace(tpimg, 'mask_c.png')
        mask = self.DataTrans(Image.open(mask_path))

        randomN_path = self.face[np.random.randint(0,self.dataset_len-1)]
        tpimgR = randomN_path.split('/')[-1]
        Rn_path = randomN_path.replace('.png', epochflag)
        randomNN = self.DataTrans(Image.open(Rn_path))
        randomNN = 2 * (randomNN - 0.5)

        # print(face_path)
        # print(gt_path)
        # print(pre_path)
        # print(Rn_path)
        # print(mask_path)
        return face, gtN, preN, randomNN, mask, face_path

    def __len__(self):
        return self.dataset_len