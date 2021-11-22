"""This file should contain all application specific datasets"""
import os

from torch.utils.data import Dataset
import torch
import torchaudio
import pathlib
import glob

class AudioDataset(Dataset):
    """Custom dataset for images with labels based on folder structure, example:
    MyFolder/label1/image1.jpg
    MyFolder/label1/image2.jpg
    MyFolder/label2/image4.jpg
    MyFolder/label2/image5.jpg
    """

    def __init__(self, root_path, max_sample_length, transform=None, target_transform=None, shuffel=True):
        super().__init__()
        image_paths, image_classes = self._get_subfolder(root_path)
        self.targets = image_classes
        self.img_paths = image_paths
        self.transform = transform
        self.target_transform = target_transform
        self.shuffel = shuffel
        self.max_len = 0
        self.max_sample_length = max_sample_length

    @staticmethod
    def _get_subfolder(root_path):
        filesAndFolders = os.listdir(root_path)
        training_names = []
        for file in filesAndFolders:
            fullPath = os.path.join(root_path, file)
            if(os.path.isdir(fullPath)):
                training_names.append(file)

        # Get all the path to the images and save them in a list
        # image_paths and the corresponding label in image_paths
        image_paths = []
        image_classes = []
        class_id = 0
        for training_name in training_names:
            training_dir = os.path.join(root_path, training_name)
            class_path = AudioDataset.imlist(training_dir, "*.wav")
            image_paths += class_path
            image_classes += [class_id] * len(class_path)
            class_id += 1

        return image_paths, image_classes

    def __len__(self):
        return len(self.targets)

    @staticmethod
    def pil_loader(path):
        """Open image, required to prevent tensorflow error"""
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            waveform, samplerate = torchaudio.load(f)
            return waveform

    def __getitem__(self, index):
        image = AudioDataset.pil_loader(self.img_paths[index])
        image = torch.tensor(image, dtype = torch.float)
        label = torch.tensor(self.targets[index], dtype=torch.float)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if(image.shape[2] < self.max_sample_length):
            # self.max_len = image.shape[2]
            image = torch.cat((image, torch.zeros(image.shape[0], image.shape[1], self.max_sample_length - image.shape[2])), dim=2)
        image = image.squeeze(0)
        return [image, label]

    @staticmethod
    def imlist(path, fileExtension):
        """
        The function imlist returns all the names of the files in
        the directory path supplied as argument to the function.
        """
        files = []
        for path in pathlib.Path(path).rglob(fileExtension):
            files.append(str(path))
        return files
        return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        # return [os.path.join(path, f) for f in (os.listdir(path) if os.path.isfile(f))]
