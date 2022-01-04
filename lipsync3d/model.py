from cv2 import getOptimalNewCameraMatrix
import torch
import torch.nn as nn
from torch.nn import functional as F

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(*self.shape)

class Lipsync3DMesh(nn.Module):
    def __init__(self):
        super().__init__()

        #TODO
        self.AudioEncoder = nn.Sequential(
            # Define Network Architecture (Hint: Architecture mentioned in the paper, Change in latent space dimensions are as follows)
            # 2 x 256 x 24 -> 72 x 128 x 24
            # 72 x 128 x 24 -> 108 x 64 x 24
            # 108 x 64 x 24 -> 162 x 32 x 24
            # 162 x 32 x 24 -> 243 x 16 x 24
            # 243 x 16 x 24 -> 256 x 8 x 24
            # 256 x 8 x 24 -> 256 x 4 x 24
            # 256 x 4 x 24 -> 128 x 4 x 13
            # 128 x 4 x 13 -> 64 x 4 x 8
            # 64 x 4 x 8 -> 32 x 4 x 5
            # 32 x 4 x 5 -> 16 x 4 x 4
            # 16 x 4 x 4 -> 8 x 4 x 3
            # 8 x 4 x 3 -> 4 x 4 x 2
            View([-1, 32]),
        )

        self.GeometryDecoder = nn.Sequential(
            nn.Linear(32, 150),
            nn.Dropout(0.5),
            nn.Linear(150, 1434)
        )

    def forward(self, spec, latentMode=False):
        # spec : B x 2 x 256 x 24
        # texture : B x 3 x 128 x 128

        latent = self.AudioEncoder(spec)
        if latentMode:
            return latent
        geometry_diff = self.GeometryDecoder(latent)

        return geometry_diff