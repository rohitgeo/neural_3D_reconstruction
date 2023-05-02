from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))


        # define decoder
        if args.type == "vox":
            # architecture 1: conv3D tranpose
            # Input: b x 512
            # Output: b x 1 x 32 x 32 x 32
            self.decoder = nn.Sequential(
                    nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(256,affine=False),
                    nn.ReLU(), 
                    nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(128,affine=False),
                    nn.ReLU(), 
                    nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(64,affine=False),
                    nn.ReLU(),
                    nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(32,affine=False),
                    nn.ReLU(),
                    nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(16,affine=False),
                    nn.ReLU(),
                    nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0),
                    )

            # architecture 2: Linear
            # self.decoder = nn.Sequential(
            #         nn.Linear(in_features=512, out_features=32),
            #         nn.ReLU(),
            #         nn.Linear(in_features=32, out_features=8),
            #         nn.ReLU(), 
            #         nn.Linear(in_features=8, out_features=32768),
            #         nn.Unflatten(1, (1, 32, 32, 32)))                 

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # Architecture 1: Conv3D Tranpose
            voxels_pred = self.decoder(encoded_feat.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            
            # Architecture 2: Linear
            # voxels_pred = self.decoder(encoded_feat)

            return voxels_pred     

