import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess_utils import *


class EpipolarLoss_full(nn.Module):
    def __init__(self, configs, device=None):
        super(EpipolarLoss_full, self).__init__()
        self.__lossname__ = 'EpipolarLoss_fullinfo'
        self.config = configs
        self.w_g = self.config['weight_grid']
        self.w_w = self.config['weight_window']

    def epipolar_cost(self, coord1, coord2, fmatrix, im_size):
        coord1_h = homogenize(coord1).transpose(1, 2)
        coord2_h = homogenize(coord2).transpose(1, 2)
        epipolar_line = fmatrix.bmm(coord1_h)  # Bx3xn
        epipolar_line_ = epipolar_line / torch.clamp(torch.norm(epipolar_line[:, :2, :], dim=1, keepdim=True), min=1e-8)
        essential_cost = torch.abs(torch.sum(coord2_h * epipolar_line_, dim=1))  # Bxn
        return essential_cost


    def set_weight(self, inverse_std, mask=None, regularizer=0.0):
        if self.config['use_std_as_weight']:
            # inverse_std = 1. / torch.clamp(std+regularizer, min=1e-10)
            weight = inverse_std / torch.mean(inverse_std)
            weight = weight.detach()  # Bxn
        else:
            weight = torch.ones_like(std)

        if mask is not None:
            weight *= mask.float()
            weight /= (torch.mean(weight) + 1e-8)
        return weight 

    def forward(self, inputs, outputs, processed):
        coord1 = processed['coord1']
        coord2 = processed['coord2']
        temperature = processed['temperature']

        feat1g_corloc = processed['feat1g_corloc']
        feat2g_corloc = processed['feat2g_corloc']
        feat1w_corloc = processed['feat1w_corloc']
        feat2w_corloc = processed['feat2w_corloc']

        feat1g_std = processed['feat1g_std']
        feat2g_std = processed['feat2g_std']
        feat1w_std = processed['feat1w_std']
        feat2w_std = processed['feat2w_std']

        Fmat1 = inputs['F1']
        Fmat2 = inputs['F2']
        im_size1 = inputs['im1'].size()[2:]
        im_size2 = inputs['im2'].size()[2:]
        shorter_edge, longer_edge = min(im_size1), max(im_size1)

        cost_g1 = self.epipolar_cost(coord1, feat1g_corloc, Fmat1, im_size1)
        cost_w1 = self.epipolar_cost(coord1, feat1w_corloc, Fmat1, im_size1)

        cost_g2 = self.epipolar_cost(coord2, feat2g_corloc, Fmat2, im_size2)
        cost_w2 = self.epipolar_cost(coord2, feat2w_corloc, Fmat2, im_size2)

        # filter out the large values, similar to CAPS
        # 去除异常loss，参考CAPS
        mask_g1 = cost_g1 < (shorter_edge*self.config['grid_cost_thr'])
        mask_w1 = cost_w1 < (shorter_edge*self.config['win_cost_thr'])
        mask_g2 = cost_g2 < (shorter_edge*self.config['grid_cost_thr'])
        mask_w2 = cost_w2 < (shorter_edge*self.config['win_cost_thr'])

        if 'valid_epi1' in list(processed.keys()):
            mask_g1 = mask_g1 & processed['valid_epi1']
            mask_w1 = mask_w1 & processed['valid_epi1']
            mask_g2 = mask_g2 & processed['valid_epi2']
            mask_w2 = mask_w2 & processed['valid_epi2']
        weight_w1 = 1
        weight_w2 = 1 

        weight_g1 = self.set_weight(1/feat1g_std.clamp(min=1e-10), mask_g1)
        weight_w1 = self.set_weight(weight_w1/feat1w_std.clamp(min=1e-10), mask_w1)
        weight_g2 = self.set_weight(1/feat2g_std.clamp(min=1e-10), mask_g2)
        weight_w2 = self.set_weight(weight_w2/feat2w_std.clamp(min=1e-10), mask_w2)

        loss_g1 = (weight_g1*cost_g1).mean()
        loss_w1 = (weight_w1*cost_w1).mean()
        loss_g2 = (weight_g2*cost_g2).mean()
        loss_w2 = (weight_w2*cost_w2).mean()

        loss = self.w_g*(loss_g1+loss_g2)+self.w_w*(loss_w1+loss_w2)

        percent_g = (mask_g1.sum()/(mask_g1.shape[0]*mask_g1.shape[1]) + mask_g2.sum()/(mask_g2.shape[0]*mask_g2.shape[1]))/2
        percent_w =  (mask_w1.sum()/(mask_w1.shape[0]*mask_w1.shape[1]) + mask_w2.sum()/(mask_w2.shape[0]*mask_w2.shape[1]))/2

        components = {
            'loss_g1': loss_g1, 'loss_w1':loss_w1, 
            'loss_g2':loss_g2, 'loss_w2':loss_w2, 
            'percent_g':percent_g, 'percent_w':percent_w
            }

        return loss, components