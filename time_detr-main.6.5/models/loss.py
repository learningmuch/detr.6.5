import torch
import math
def box_cxcywh_to_xyxy(boxes):
    x_min = boxes[:, 0] - boxes[:, 2] / 2.0
    y_min = boxes[:, 1] - boxes[:, 3] / 2.0
    x_max = boxes[:, 0] + boxes[:, 2] / 2.0
    y_max = boxes[:, 1] + boxes[:, 3] / 2.0
    return torch.stack((x_min, y_min, x_max, y_max), dim=1)


# # 在这里原来的情况bbox[2]改为bbox[:,2]
def getCenterPoint(bbox):
    return (bbox[:,2]+bbox[:,0]) / 2. , (bbox[:,3]+bbox[:,1]) / 2.



def getDistance(point1, point2):
    return torch.sqrt(torch.sum((point1 - point2) ** 2)).unsqueeze(0)


# 计算最小外接矩形的对角线距离
def getConvexShape(target_boxes, src_boxes):
    xmin = torch.min(torch.cat([target_boxes[:, 0], src_boxes[:, 0]]))
    ymin = torch.min(torch.cat([target_boxes[:, 1], src_boxes[:, 1]]))
    xmax = torch.max(torch.cat([target_boxes[:, 2], src_boxes[:, 2]]))
    ymax = torch.max(torch.cat([target_boxes[:, 3], src_boxes[:, 3]]))

    d = torch.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
    return d

import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        return focal_loss.mean()
