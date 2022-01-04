import sys

from hparams import hparams

sys.path.append('/home/server01/jyeongho_workspace/3d_face_gcns/')

import os
import torch
import numpy as np
import librosa
from utils import landmarkdict_to_normalized_mesh_tensor, landmarkdict_to_mesh_tensor
from audiodvp_utils import util
from torch.utils.data import Dataset
from natsort import natsorted
import torchvision.transforms as transforms
import cv2
from PIL import Image
import audio as audioLibrary
import random

class Lipsync3DMeshDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.src_dir = opt.src_dir
        self.tgt_dir = opt.tgt_dir

        self.stabilized_mesh = [os.path.join(self.tgt_dir, 'stabilized_norm_mesh', x) for x in natsorted(os.listdir(os.path.join(self.tgt_dir, 'stabilized_norm_mesh')))]


        stft_path = os.path.join(self.src_dir, 'audio/audio_stft.pt')
        if not os.path.exists(stft_path):
            audio = librosa.load(os.path.join(self.src_dir, 'audio/audio.wav'),16000)[0]
            audio_stft = librosa.stft(audio, n_fft=510, hop_length=160, win_length=480)
            self.audio_stft = torch.from_numpy(np.stack((audio_stft.real, audio_stft.imag)))
            torch.save(self.audio_stft, os.path.join(self.src_dir, 'audio/audio_stft.pt'))
        else:
            self.audio_stft = torch.load(os.path.join(self.src_dir, 'audio/audio_stft.pt'))
        
        self.mesh_dict_list = util.load_coef(os.path.join(self.tgt_dir, 'mesh_dict'))
        self.filenames = util.get_file_list(os.path.join(self.tgt_dir, 'mesh_dict'))
        reference_mesh_dict = torch.load(os.path.join(self.tgt_dir, 'reference_mesh.pt'))

        self.reference_mesh = landmarkdict_to_normalized_mesh_tensor(reference_mesh_dict)
        
        if opt.isTrain:
            minlen = min(len(self.mesh_dict_list), self.audio_stft.shape[2] // 4)
            train_idx = int(minlen * self.opt.train_rate)
            self.mesh_dict_list = self.mesh_dict_list[:train_idx]
            self.filenames = self.filenames[:train_idx]

        print('Training set size: ', len(self.filenames))

    def __len__(self):
        return min(self.audio_stft.shape[2] // 4, len(self.filenames))

    def __getitem__(self, index):

        audio_idx = index * 4

        audio_feature_list = []
        for i in range(audio_idx - 12, audio_idx + 12):
            if i < 0:
                audio_feature_list.append(self.audio_stft[:, :, 0])
            elif i >= self.audio_stft.shape[2]:
                audio_feature_list.append(self.audio_stft[:, :, -1])
            else:
                audio_feature_list.append(self.audio_stft[:, :, i])

        audio_feature = torch.stack(audio_feature_list, 2)

        filename = os.path.basename(self.filenames[index])

        if not self.opt.isTrain:
            landmark_dict = self.mesh_dict_list[index]
            normalized_mesh = landmarkdict_to_normalized_mesh_tensor(landmark_dict)
            # stabilized_mesh = torch.tensor(torch.load(self.stabilized_mesh[index]))

            R = torch.from_numpy(landmark_dict['R']).float()
            t = torch.from_numpy(landmark_dict['t']).float()
            c = float(landmark_dict['c'])

            return {'audio_feature': audio_feature, 'filename': filename, 
                    'reference_mesh': self.reference_mesh, 'normalized_mesh': normalized_mesh,
                    'R': R, 't': t, 'c': c}

        else:
            landmark_dict = self.mesh_dict_list[index]
            normalized_mesh = landmarkdict_to_normalized_mesh_tensor(landmark_dict)
            # stabilized_mesh = torch.tensor(torch.load(self.stabilized_mesh[index]))
            return {
                'audio_feature': audio_feature, 'filename': filename,
                'reference_mesh' : self.reference_mesh, 'normalized_mesh': normalized_mesh
            }