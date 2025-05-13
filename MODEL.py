from SwinTransformer import SwinTransformerV2_3D
from hovertrans import create_model
import torch
import torch.nn as nn
import numpy as np


class Model3(nn.Module):
    def __init__(self, device = 'cuda:0',norm_layer=nn.LayerNorm,**kwargs):
        super().__init__()

        self.backbone_mm = create_model(img_size=256, in_chans=4, num_classes=2, drop_rate=0.1, attn_drop_rate=0.1,
            patch_size=[2, 2, 2, 2], dim=[4, 8, 16, 32], depth=[2, 4, 4, 2], num_heads=[2, 4, 8, 16],
            num_inner_head=[2, 4, 8, 16])
        self.backbone_us = create_model(img_size=256, in_chans=1, num_classes=2, drop_rate=0.1, attn_drop_rate=0.1,
            patch_size=[2, 2, 2, 2], dim=[4, 8, 16, 32], depth=[2, 4, 4, 2], num_heads=[2, 4, 8, 16],
            num_inner_head=[2, 4, 8, 16])
        self.backbone_mri = SwinTransformerV2_3D(Devise=device)

        self.norm = norm_layer(64)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(768+5, 2)

        self.act1 = nn.GELU()
        self.act2 = nn.Sigmoid()
        self.head1 = nn.Linear(768, 768)
        self.head_clinical = nn.Linear(5, 2)

    def forward(self, MRI, US, MM,clinical_data,modality_index):
        mri = self.backbone_mri(MRI)
        us = self.backbone_us(US)
        mm = self.backbone_mm(MM)

        x0 = (us*modality_index[:,0].unsqueeze(1)+mm*modality_index[:,2].unsqueeze(1)+mri*modality_index[:,1].unsqueeze(1))/modality_index.sum(1).unsqueeze(1)

        multi_out = self.head(torch.cat([x0,clinical_data],1))
        mri_out = self.head(torch.cat([mri,clinical_data],1))
        us_out = self.head(torch.cat([us,clinical_data],1))
        mm_out = self.head(torch.cat([mm,clinical_data],1))

        return multi_out,mri_out,us_out,mm_out,mri,us,mm

