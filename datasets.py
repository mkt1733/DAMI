from os import listdir
from torch.utils.data import Dataset
from utils import *


class ImageDataset(Dataset):

    def __init__(self, image_dir_a):
        super(ImageDataset, self).__init__()
        self.a_dir = image_dir_a
        self.image_filenames = [x for x in listdir(self.a_dir) if is_image_file(x)]

    def __getitem__(self, index):
        """
        a: not in focus
        b: in focus
        """
        a_name = self.image_filenames[index]
        return a_name

    def __len__(self):
        return len(self.image_filenames)


class ValDataset(Dataset):

    def __init__(self, image_dir_a):
        super(ValDataset, self).__init__()
        self.a_dir = image_dir_a
        self.image_filenames = [x for x in listdir(self.a_dir) if is_image_file(x)]

    def __getitem__(self, index):
        """
        a: not in focus
        b: in focus
        """
        a_name = self.image_filenames[index]
        return a_name

    def __len__(self):
        return len(self.image_filenames)
