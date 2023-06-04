# 这只是一个用来测试的文件
import torch
from torchvision.ops import box_iou, box_area
import math
from models.loss import box_cxcywh_to_xyxy
from models.loss import getCenterPoint
# from .loss import getDistance_2
from models.loss import getDistance
from models.loss import getConvexShape


target_boxes =(torch.tensor([[0.4039, 0.5281, 0.1612, 0.3886],
        [0.4199, 0.5254, 0.1172, 0.2988]], device='cuda:0'))
src_boxes= (torch.tensor([[0.3596, 0.3402, 0.1426, 0.2110],
        [0.5264, 0.4752, 0.1871, 0.4018]], device='cuda:0'))
losses = {}
iou = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
center_point_gt = getCenterPoint(target_boxes)
center_point_pd = getCenterPoint(src_boxes)
first_gt = center_point_gt[0]
second_gt = center_point_gt[1]
second_pd = center_point_pd[1]
first_pd = center_point_pd[0]
rho_0 = getDistance(first_gt, first_pd) ** 2
rho_1 = getDistance(second_gt, second_pd) ** 2
rho_2 = (rho_0+rho_1)/2
bbox_cvx = getConvexShape(target_boxes, src_boxes)
c_2 = bbox_cvx
cious = iou - (rho_2 / (c_2+1e-7))
v = (4 / (math.pi ** 2)) * torch.pow(
            (torch.atan(src_boxes[:, 2] / src_boxes[:, 3]) - torch.atan(target_boxes[:, 2] / target_boxes[:, 3])),
            2).unsqueeze(1)
alpha = v / (1 - iou + v)
loss_ciou = 1 - cious + alpha * v
losses['loss_ciou'] = loss_ciou.sum() / 2
print(f"losses is {losses} ,num_boxes is 2")
#