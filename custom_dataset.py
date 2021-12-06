"""This file should contain all application specific datasets"""
import os

from torch.utils.data import Dataset
import torch
import torchaudio
import pathlib
import glob

class AudioDataset(Dataset):
    """Custom dataset for audio files with labels based on folder structure, example:
    MyFolder/label1/audio1.wav
    MyFolder/label1/audio2.wav
    MyFolder/label2/audio4.wav
    MyFolder/label2/audio5.wav
    """

    def __init__(self, root_path, max_sample_length, transform=None, target_transform=None, shuffel=True):
        super().__init__()
        audio_paths, audio_classes = self._get_subfolder(root_path)
        self.targets = audio_classes
        self.audio_paths = audio_paths
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

        # Get all the path to the audios and save them in a list
        # audio_paths and the corresponding label in audio_paths
        audio_paths = []
        audio_classes = []
        class_id = 0
        for training_name in training_names:
            training_dir = os.path.join(root_path, training_name)
            class_path = AudioDataset.imlist(training_dir, "*.wav")
            audio_paths += class_path
            audio_classes += [class_id] * len(class_path)
            class_id += 1

        return audio_paths, audio_classes

    def __len__(self):
        return len(self.targets)

    @staticmethod
    def pil_loader(path):
        """Open audio, required to prevent tensorflow error"""
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            waveform, samplerate = torchaudio.load(f)
            return waveform

    def __getitem__(self, index):
        audio = AudioDataset.pil_loader(self.audio_paths[index])
        audio = audio.float()
        label = torch.tensor(self.targets[index], dtype=torch.float)
        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            label = self.target_transform(label)
        if(audio.shape[2] < self.max_sample_length):
            # self.max_len = audio.shape[2]
            audio = torch.cat((audio, torch.zeros(audio.shape[0], audio.shape[1], self.max_sample_length - audio.shape[2])), dim=2)
        audio = audio.squeeze(0)
        return [audio, label]

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
