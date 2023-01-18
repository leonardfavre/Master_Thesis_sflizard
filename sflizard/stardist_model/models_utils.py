from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stardist import non_maximum_suppression, polygons_to_label

class MyL1BCELoss(torch.nn.Module):
    def __init__(self, scale=[1, 1]):
        super(MyL1BCELoss, self).__init__()
        assert len(scale) == 2
        self.scale = scale

    def forward(self, prediction, obj_probabilities, target_dists):
        # Predicted distances errors are weighted by object prob
        l1loss = F.l1_loss(prediction[0], target_dists, size_average=True, reduce=False)
        # weights = self.getWeights(target_dists)
        l1loss = torch.mean(obj_probabilities * l1loss)
        bceloss = F.binary_cross_entropy(
            prediction[1],
            obj_probabilities,
            weight=None,
            size_average=True,
            reduce=True,
        )
        return self.scale[0] * l1loss + self.scale[1] * bceloss


class ClassL1BCELoss(torch.nn.Module):
    def __init__(self, class_weights, scale=[1, 1, 1]):
        super(ClassL1BCELoss, self).__init__()
        assert len(scale) == 3
        self.scale = scale
        self.class_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.mean = [0, 0, 0]
        self.total = 0

    def forward(self, prediction, obj_probabilities, target_dists, classes):
        # Predicted distances errors are weighted by object prob
        l1loss = F.l1_loss(prediction[0], target_dists, size_average=True, reduce=False)
        # weights = self.getWeights(target_dists)
        l1loss = torch.mean(obj_probabilities * l1loss)
        bceloss = F.binary_cross_entropy(
            prediction[1],
            obj_probabilities,
            weight=None,
            size_average=True,
            reduce=True,
        )
        classloss = self.class_loss(prediction[2], classes)
        return (
            self.scale[0] * l1loss + self.scale[1] * bceloss + self.scale[2] * classloss
        )


class UNetStar(nn.Module):
    def __init__(self, n_channels, n_rays, n_classes=None, last_layer_out=False):
        """Init the class."""
        super(UNetStar, self).__init__()
        self.output_classes = n_classes is not None
        self.output_last_layer = last_layer_out
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 128)
        self.up1 = up(256, 64)
        self.up2 = up(128, 32)
        self.up3 = up(64, 32)
        self.features = nn.Conv2d(32, 128, 3, padding=1)
        self.out_ray = outconv(128, n_rays)
        self.final_activation_ray = nn.ReLU()
        self.out_prob = outconv(128, 1)
        self.final_activation_prob = nn.Sigmoid()
        if self.output_classes:
            self.out_class = outconv(128, n_classes)
            self.final_activation_class = nn.Softmax(dim=1)

    def forward(self, x):
        """Forward the input in the network."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.features(x)
        if self.output_last_layer:
            return [x]
        out_ray = self.out_ray(x)
        out_ray = self.final_activation_ray(out_ray)
        out_prob = self.out_prob(x)
        out_prob = self.final_activation_prob(out_prob)
        if self.output_classes:
            out_class = self.out_class(x)
            out_class = self.final_activation_class(out_class)
            return [out_ray, out_prob, out_class]
        else:
            return [out_ray, out_prob]

    def compute_star_label(
        self,
        image: torch.Tensor,
        dist: torch.Tensor,
        prob: torch.Tensor,
    ):
        """Compute the stare label of images according dist and prob."""
        star_labels = []
        for i in range(image.shape[0]):
            dist_numpy = dist[i].detach().cpu().numpy().squeeze()
            prob_numpy = prob[i].detach().cpu().numpy().squeeze()
            dist_numpy = np.transpose(dist_numpy, (1, 2, 0))
            points, probs, dists = non_maximum_suppression(
                dist_numpy, prob_numpy, nms_thresh=0.5, prob_thresh=0.5
            )
            star_label = polygons_to_label(
                dists, points, (image.shape[3], image.shape[2]), probs
            )

            star_labels.append(star_label)
        star_labels = np.array(star_labels)
        return star_labels


# Utilities for UNetStar model
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Init the class."""
        super(double_conv, self).__init__()
        num_groups = out_ch // 8
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
            nn.ELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        """Forward the input in the network."""
        return self.conv(x)


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Init the class."""
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        """Forward the input in the network."""
        return self.conv(x)


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Init the class."""
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        """Forward the input in the network."""
        return self.mpconv(x)


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Init the class."""
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        """Forward the input in the network."""
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Init the class."""
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        """Forward the input in the network."""
        return self.conv(x)
