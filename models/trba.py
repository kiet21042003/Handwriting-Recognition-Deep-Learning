import torch
import torch.nn as nn
import torch.optim
from torchmetrics.text import CharErrorRate
from torch.nn import init

from blocks.feature_extractor import ResNet_FeatureExtractor
from blocks.sequence_modelling import BidirectionalLSTM
from blocks.predictor import Attention
from blocks.transformation import TPS_SpatialTransformerNetwork

class TrBA(nn.Module):
    def __init__(self, img_channel, img_height, img_width, num_class,
                 max_len, dropout=0.0, stn_on=True):
        super(TrBA, self).__init__()
        self.feature_extractor = ResNet_FeatureExtractor(img_channel, 512)
        self.sequence_modelling = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, 256)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.predictor = Attention(256, 256, num_class)
        
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        self.img_height = img_height
        self.img_width = img_width
        
        if stn_on:
            self.tps = TPS_SpatialTransformerNetwork(
                20, (img_height, img_width), (img_height, img_width), img_channel
            )
        else:
            self.tps = nn.Identity()
        
        # weight initialization
        initialize_weights(self.feature_extractor)
        initialize_weights(self.sequence_modelling)
        initialize_weights(self.adaptive_pool)
        initialize_weights(self.predictor)
        
    def forward(self, images, text=None, is_train=True, seqlen=None):
        # shape of images: (B, C, H, W)
        if seqlen is None:
            seqlen = self.max_len
        
        # Transformation
        images = self.tps(images)

        # Feature extraction
        feature_map = self.feature_extractor(images)
        visual_feature = self.dropout(feature_map)  # Dropout
        visual_feature = self.adaptive_pool(visual_feature.permute(0, 3, 1, 2)) # (B, C, H, W) -> (B, W, C, H) -> (B, W, C, 1)
        visual_feature = visual_feature.squeeze(3) # (B, W, C, 1) -> (B, W, C)

        # Sequence modeling
        contextual_feature = self.sequence_modelling(visual_feature)

        # Prediction
        prediction = self.predictor(contextual_feature.contiguous(), text, is_train, seqlen)
        
        return prediction, feature_map
        
        
def initialize_weights(model):
    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue