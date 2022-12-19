import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms


IMAGE_SIZE = 256
IMAGE_SIZE_128 = 128



def get_PhotoDB(csvPath=None, validation_split=0):

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

    full_dataset = Dataset_PhotoDB(face, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train.py-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class Dataset_PhotoDB(Dataset):
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



def get_Pre_trainData(csvPath=None, validation_split=0):

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

    full_dataset = Dataset_Pre_train(face, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train.py-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class Dataset_Pre_train(Dataset):
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

        preN_path = face_path.replace('.png', '_preN.png')
        pre_norm = self.DataTrans(Image.open(preN_path))
        pre_norm = 2 * (pre_norm - 0.5)


        random_path = self.face[np.random.randint(0,self.dataset_len-1)].replace('.png', '_preN.png')
        randonN = self.DataTrans(Image.open(random_path))
        randonN = 2 * (randonN - 0.5)

        mask = randonN
        if face_path.split('/')[7] == 'PhotofaceCrop':
            mask = self.transform(Image.open(face_path[:77] +  'mask_c.png'))
        else:
            mask_path = self.face[index].replace('.png', '_mask.png')
            mask = self.DataTrans(Image.open(mask_path))
        # print(face_path)
        # print(preN_path)
        # print(random_path)

        return face, mask, pre_norm, randonN, face_path

    def __len__(self):
        return self.dataset_len





#-------------------------- start photoface dataset ---------------------
def getPhotoDB_Pre(csvPath=None, validation_split=0):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    full_dataset = PhotoDB_Pre(face, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train.py-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class PhotoDB_Pre(Dataset):
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

        return face, face_path

    def __len__(self):
        return self.dataset_len
#-------------------------- end p2 ffhq dataset ---------------------
        

#-------------------------- start photoface dataset ---------------------
def getPhotoDB_PreTrain(csvPath=None, IMAGE_SIZE=256, validation_split=0):

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

    full_dataset = PhotoDB_PreTrain(face, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train.py-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class PhotoDB_PreTrain(Dataset):
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

        mask = torch.where(gtN[0,:,:]==1, 0, 1).unsqueeze(0).expand(3,IMAGE_SIZE, IMAGE_SIZE)
        return face, gtN, mask, face_path

    def __len__(self):
        return self.dataset_len
#-------------------------- end hotoface dataset ---------------------






#-------------------------- start photoface dataset ---------------------
def getPhotoDB_Pre_512(csvPath=None, validation_split=0):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
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
        self.DataTrans = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __getitem__(self, index):
        face_path = self.face[index]

        face = self.DataTrans(Image.open(face_path))

        gt_path = ''
        if(((face_path.split('/')[-1]).split('.')[0]).split('_')[-1] == '50'):
            gt_path = face_path[:-15] + 'normal_crop_50.png'
        if(((face_path.split('/')[-1]).split('.')[0]).split('_')[-1] == '75'):
            gt_path = face_path[:-15] + 'normal_crop_75.png'
        if(((face_path.split('/')[-1]).split('.')[0]).split('_')[-1] == '100'):
            gt_path = face_path[:-16] + 'normal_crop_100.png'
        if(((face_path.split('/')[-1]).split('.')[0]).split('_')[-1] == '125'):
            gt_path = face_path[:-16] + 'normal_crop_125.png'

        gPN = self.DataTrans(Image.open(gt_path))
        gtN = 2 * (gPN - 0.5)

        mask = torch.where(gtN[0,:,:]==1, 0, 1).unsqueeze(0).expand(3,512,512)
        return face, gtN, mask, face_path

    def __len__(self):
        return self.dataset_len
#-------------------------- end hotoface dataset ---------------------

def getCelebA_P1_512(csvPath=None, validation_split=0):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    full_dataset = DatasetCelebA_P1(face, transform)

    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset


class DatasetCelebA_P1(Dataset):
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







def get_2D3dData_128(csvPath=None, validation_split=0):

    df = pd.read_csv(csvPath)
    face = list(df['face'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size)
    train_count = dataset_size - validation_count

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
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
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        face_path = self.face[index]
        face = self.DataTrans(Image.open(face_path))

        return face, face_path

    def __len__(self):
        return self.dataset_len