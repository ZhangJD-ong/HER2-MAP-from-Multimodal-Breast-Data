import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.transforms import CenterCrop, Grayscale, RandomHorizontalFlip, RandomRotation
import pandas as pd
import SimpleITK as sitk
from glob import glob
from PIL import Image
import numpy as np
import random
import cv2
import os


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
            img = N + img
            img[img > 255] = 255
            img = Image.fromarray(img.astype('uint8')).convert('L')
            return img
        else:
            return img

class AddBlur(object):
    def __init__(self, kernel=3, p=1):
        self.kernel = kernel
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            img = cv2.blur(img, (self.kernel, self.kernel))
            img = Image.fromarray(img.astype('uint8')).convert('L')
            return img
        else:
            return img


class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=30, axes=None, mode='reflect', order=0, **kwargs):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, m):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        if m.ndim == 3:
            m = rotate(m, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
        else:
            channels = [rotate(m[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1) for c
                        in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


class RandomContrast:
    """
    Adjust contrast by scaling each voxel to `mean + alpha * (v - mean)`.
    """

    def __init__(self, random_state, alpha=(0.5, 1.5), mean=0.0, execution_probability=0.1, **kwargs):
        self.random_state = random_state
        assert len(alpha) == 2
        self.alpha = alpha
        self.mean = mean
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            alpha = self.random_state.uniform(self.alpha[0], self.alpha[1])
            result = self.mean + alpha * (m - self.mean)
            return np.clip(result, -1, 1)

        return m

class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, random_state, axis_prob=0.5, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.axes = (0, 1, 2)
        self.axis_prob = axis_prob

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        for axis in self.axes:
            if self.random_state.uniform() > self.axis_prob:
                if m.ndim == 3:
                    m = np.flip(m, axis)
                else:
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)

        return m


class Custom_Dataset(Dataset):
    def __init__(self, root, transform_3D, transform_2D_US,transform_2D_MM, csv_path, clinical_path,img_size_US,img_size_MM,mode):
        super().__init__()
        self.root = root
        self.transform_3D = transform_3D
        self.transform_2D_US = transform_2D_US
        self.transform_2D_MM = transform_2D_MM
        self.csv = csv_path
        df = pd.read_csv(self.csv)
        self.clinical_data = pd.read_csv(clinical_path)
        self.info = df
        self.img_size_US = img_size_US
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.img_size_MM = img_size_MM
        self.mode = mode

    def __getitem__(self, index):
        patience_info = self.info.iloc[index]
        file_name = str(int(patience_info['ID']))
        label = patience_info['Label']
        MRI_flag = patience_info['MRI']
        US_flag = patience_info['US']
        MM_flag = patience_info['MM']
        LR_flag = patience_info['L1R2']

        ## load MRI data
        if MRI_flag == 1:

            img_DCE1 = load(os.path.join(os.path.join(self.root,'MRI', file_name, 'P2.nii.gz')))
            if self.transform_3D is not None:
                img_MRI = self.transform_3D(img_DCE1[np.newaxis, :])
            else:
                img_MRI = img_DCE1[np.newaxis, :]
        else:
            img_MRI = np.zeros([1,96,96,96])

        ### load US data
        if US_flag == 1:
            file_path = os.path.join(self.root,'US',file_name)
            US_name = self.US_sequence(file_path)
            img_US = Image.open(os.path.join(file_path,US_name))
            if self.transform_2D_US is not None:
                img_US = self.transform_2D_US(img_US)
            else:
                img_US = img_US
        else:
            img_US = self.transform(np.zeros([self.img_size_US,self.img_size_US]))

        ### load MM data
        if MM_flag == 1:
            file_path = os.path.join(self.root,'MM',file_name)
            if LR_flag == 1:
                img_MM_MLO = Image.open(os.path.join(file_path,'original','L_MLO.png'))
                img_MM_CC = Image.open(os.path.join(file_path,'original', 'L_CC.png'))
                img_MM_MLO1 = Image.open(os.path.join(file_path,'crop','L_MLO.png'))
                img_MM_CC1 = Image.open(os.path.join(file_path,'crop', 'L_CC.png'))
            else:
                img_MM_MLO = Image.open(os.path.join(file_path,'original', 'R_MLO.png'))
                img_MM_CC = Image.open(os.path.join(file_path,'original', 'R_CC.png'))
                img_MM_MLO1 = Image.open(os.path.join(file_path,'crop', 'R_MLO.png'))
                img_MM_CC1 = Image.open(os.path.join(file_path,'crop', 'R_CC.png'))

            if self.transform_2D_MM is not None:
                img_MM_mlo = self.transform_2D_MM(img_MM_MLO)
                img_MM_cc = self.transform_2D_MM(img_MM_CC)
                img_MM_mlo1 = self.transform_2D_MM(img_MM_MLO1)
                img_MM_cc1 = self.transform_2D_MM(img_MM_CC1)
                img_MM = np.concatenate((img_MM_cc, img_MM_mlo,img_MM_cc1, img_MM_mlo1), 0)
            else:
                img_MM = np.concatenate((img_MM_CC[np.newaxis, :], img_MM_MLO[np.newaxis, :],img_MM_CC1[np.newaxis, :], img_MM_MLO1[np.newaxis, :]), 0)
        else:
            img_MM_ll = self.transform(np.zeros([self.img_size_MM,self.img_size_MM]))
            img_MM = np.concatenate((img_MM_ll, img_MM_ll,img_MM_ll, img_MM_ll), 0)


        clinical = self.clinical_data.loc[self.clinical_data['ID'] == int(file_name)].values[0, 1:6]
        clinical = np.array(clinical).astype(np.float32)
        Modality_index = self.clinical_data.loc[self.clinical_data['ID'] == int(file_name)].values[0, 10:13]  #US MRI MM
        Modality_index = np.array(Modality_index)


        return {'MRI': img_MRI,'MM': img_MM,'US': img_US, 'labels': label, 'clinial': clinical, 'Modality_index': Modality_index, 'names': file_name}#

    def __len__(self):
        return len(self.info)

    def US_sequence(self,file_path):
        filename = os.listdir(file_path)
        if len(filename)==0:
            random_num = 0
        else:
            random_num = np.random.randint(len(filename))
        if self.mode == 'train':
            file_names = filename[random_num]
        else:
            file_names = filename[0]

        return file_names



def get_dataset(imgpath, csvpath, clinical_path,img_size_US,img_size_MM, mode='train', keyword=None):
    train_transform_3D = transforms.Compose([
        RandomContrast(np.random.RandomState(), alpha=(0.5, 1.5), mean=0.0, execution_probability=0.1),
        RandomFlip(np.random.RandomState(), axis_prob=0.5, axis=0),
        RandomFlip(np.random.RandomState(), axis_prob=0.5, axis=1),
        RandomFlip(np.random.RandomState(), axis_prob=0.5, axis=2),
    ])
    test_transform_3D = None
    train_transform_2D_MM = transforms.Compose([
        transforms.Resize((img_size_MM, img_size_MM)),
        transforms.Grayscale(),
        transforms.CenterCrop((img_size_MM, img_size_MM)),
        AddGaussianNoise(amplitude=random.uniform(0, 1), p=0.5),
        AddBlur(kernel=3, p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 2)),
        transforms.ToTensor(),

    ])
    test_transform_2D_MM = transforms.Compose([
        transforms.Resize((img_size_MM, img_size_MM)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    train_transform_2D_US = transforms.Compose([
        transforms.Resize((img_size_US, img_size_US)),
        transforms.Grayscale(),
        transforms.CenterCrop((img_size_US, img_size_US)),
        AddGaussianNoise(amplitude=random.uniform(0, 1), p=0.5),
        AddBlur(kernel=3, p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 2)),
        transforms.ToTensor(),
    ])
    test_transform_2D_US = transforms.Compose([
        transforms.Resize((img_size_US, img_size_US)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    if mode == 'train':
        transform_3D = train_transform_3D
        transform_2D_MM = train_transform_2D_MM
        transform_2D_US = train_transform_2D_US
    elif mode == 'test':
        transform_3D = test_transform_3D
        transform_2D_MM = test_transform_2D_MM
        transform_2D_US = test_transform_2D_US
    else:
        transform_3D = None
        transform_2D_MM = None
        transform_2D_US = None

    dataset = Custom_Dataset(imgpath, transform_3D, transform_2D_US,transform_2D_MM, csvpath, clinical_path,img_size_US,img_size_MM,mode)

    return dataset


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.flatten(preds)
    labels = torch.flatten(labels)
    for p, t in zip(preds, labels):
        conf_matrix[int(p), int(t)] += torch.tensor(1)
    return conf_matrix


def load(file):
    itkimage = sitk.ReadImage(file)
    image = sitk.GetArrayFromImage(itkimage)
    return image
