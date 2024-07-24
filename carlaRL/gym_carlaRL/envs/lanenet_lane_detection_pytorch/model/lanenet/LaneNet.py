# coding: utf-8
"""
LaneNet model
https://arxiv.org/pdf/1807.01726.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# from gym_carlaRL.envs.lanenet_lane_detection_pytorch.model.lanenet.loss import DiscriminativeLoss
# from gym_carlaRL.envs.lanenet_lane_detection_pytorch.model.lanenet.backbone.UNet import UNet_Encoder, UNet_Decoder
from gym_carlaRL.envs.lanenet_lane_detection_pytorch.model.lanenet.backbone.ENet import ENet_Encoder, ENet_Decoder
# from gym_carlaRL.envs.lanenet_lane_detection_pytorch.model.lanenet.backbone.deeplabv3_plus.deeplabv3plus import Deeplabv3plus_Encoder, Deeplabv3plus_Decoder

# from model.lanenet.backbone.ENet import ENet_Encoder, ENet_Decoder


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LaneNet(nn.Module):
    def __init__(self, in_ch = 3, arch="ENet"):
        super(LaneNet, self).__init__()
        # no of instances for segmentation
        self.no_of_instances = 3  # if you want to output RGB instance map, it should be 3.
        print("Use {} as backbone".format(arch))
        self._arch = arch

        if self._arch == 'ENet':
            self._encoder = ENet_Encoder(in_ch)
            self._encoder.to(DEVICE)

            self._decoder_binary = ENet_Decoder(2)
            self._decoder_instance = ENet_Decoder(self.no_of_instances)
            self._decoder_binary.to(DEVICE)
            self._decoder_instance.to(DEVICE)

        else:
            raise("Please select right model.")

        self.relu = nn.ReLU().to(DEVICE)
        self.sigmoid = nn.Sigmoid().to(DEVICE)

    def forward(self, input_tensor):

        if self._arch == 'ENet':
            c = self._encoder(input_tensor)
            binary = self._decoder_binary(c)
            instance = self._decoder_instance(c)
        else:
            raise("Please select right model.")

        binary_seg_ret = torch.argmax(F.softmax(binary, dim=1), dim=1, keepdim=True)

        pix_embedding = self.sigmoid(instance)

        return {
            'instance_seg_logits': pix_embedding,
            'binary_seg_pred': binary_seg_ret,
            'binary_seg_logits': binary
        }
