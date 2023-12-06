from __future__ import print_function
import torch
import glob
from tqdm import tqdm
import cv2
import numpy as np
import cv2
from torch.utils.data import DataLoader
import nibabel as nib 
import configparser
import os 


class ACDCDatasetBase(object):
    """
    Base of the image denoising dataset.

    :param root_dir: path to the directory with all the images
    :type root_dir: str
    :param color: True for rgb, False for grayscale images
    :type color: bool
    :param prefetch: If to prefetch all data at init
    :type prefetch: bool
    :param transform: Transformations applied to sample
    :type transform: callable
    :param patch_transform: Transformations applied to the prefetched image patches
    :type patch_transform: callable
    :param real_noise: True for real noise, False for artificial noise
    :type real_noise: bool
    :param restrict_data: Determines which percentage to use of the dataset
    :type restrict_data: float
    """

    def __init__(self, root_dir,
                    #    color=False,
                    #    prefetch=False,
                    #    transform=None,
                    #    patch_transform=None,
                    #    real_noise=False, 
                    #    restrict_data=None
                       ) -> None:
        """
        Constructor Method
        """

        extensions = ['png', 'jpg', 'bmp']
        # self.color = color
        # self.prefetch = prefetch
        # self.root_dir = root_dir
        # self.transform = transform
        # self.real_noise = real_noise
        # self.restrict_data = restrict_data
        
        self.targets_fnames = []
        self.input_fnames = []
        for ext in extensions:
            try:
                    self.targets_fnames += sorted(glob.glob(f'{root_dir}/groundtruth/*.{ext}'))
                    self.input_fnames += sorted(glob.glob(f'{root_dir}/input/*.{ext}'))
            except:
                ImportError('No data found in the given path')

        if self.restrict_data:
            length = int(len(self.targets_fnames)*self.restrict_data)
            self.targets_fnames = self.targets_fnames[0:length]
            self.input_fnames = self.input_fnames[0:length]
        print(f'{len(self.targets_fnames)} files found in {root_dir}')
        assert len(self.targets_fnames) != 0, f"Given directory contains 0 images. Please check on the given root: {self.root_dir}"



    @property
    def fnames(self):
        return self.targets_fnames

    @staticmethod
    def save_img(fname:str, img:np.array) -> None:
        """
        Saves the given image (img) und the given path (fname)
        
        :param fname: Determines which percentage to use of the dataset
        :type fname: str
        :param img: The image to save
        :type img: np.array
        """
        img = np.round(np.clip(img, 0, 1) * 255.0).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        cv2.imwrite(fname, img[...,::-1])

    def load_img(self, image_fname:str) -> np.array:
        """
        Loads an image from file in either rgb or grayscale mode
        
        :param image_fname: The filename to load
        :type image_fname: str

        """
        if self.color:
            np_target = cv2.imread(image_fname, cv2.IMREAD_COLOR)[...,::-1] # convert to RGB
        else:
            np_target = cv2.imread(image_fname, cv2.IMREAD_GRAYSCALE)[...,np.newaxis]

        np_target = np_target.astype(np.float32) / 255.0
        return 
    
    def load_nii(self, fname:str):
        nii = nib.load(fname)
        return nii

    def load_meta_patient(self, dirname:str):

        config = configparser.ConfigParser()
        config.read(os.path.join(dirname, "Info.cfg"))

        # Convert the configuration to a dictionary
        config_dict = {section: dict(config[section]) for section in config.sections()}

        return config_dict

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # load the target image
        target = self.load_img(self.targets_fnames[idx])
        input = self.load_img(self.input_fnames[idx])

        return (target, input)
