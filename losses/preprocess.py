import torch
import torch.nn as nn
import torch.nn.functional as F
from . import preprocess_utils as putils
from .preprocess_utils import *

class Preprocess_Line2Window(nn.Module):
    '''
    the preprocess class for grid-with-line pipeline
    '''
    def __init__(self, configs, device=None, vis=False):
        super(Preprocess_Line2Window, self).__init__()
        self.__lossname__ = 'Preprocess_Line2Window'
        self.config = configs
        self.kps_generator = getattr(putils, self.config['kps_generator'])
        self.t_base = self.config['temperature_base']
        self.t_max = self.config['temperature_max']
        if device is not None:
            self.device = device

    def name(self):
        return self.__lossname__

    def forward(self, inputs, outputs):
        preds1 = outputs['preds1']
        preds2 = outputs['preds2']

        xc1, xf1 = preds1['global_map'], preds1['local_map']
        xc2, xf2 = preds2['global_map'], preds2['local_map']
        h1i, w1i = inputs['im1'].size()[2:]
        h2i, w2i = inputs['im2'].size()[2:]
        b, _, hf, wf = xf1.shape
        temperature = min(self.t_base + outputs['epoch'], self.t_max)

        """
        firstly, we search locate the correspondence with grid points
        with keep_spatial==True, coord (score) is with bxhxwx2 (bxhxwx1)
        with keep_spatial==False, coord (score) is with bx(h*w)x2 (bx(h*w)x1)
        the keep_spatial is defined in self.config['kps_generator_config']
        首先，我们随所有的抽样点进行匹配搜索
        在配置文件中有一个选项 keep_spatial 可以控制输出的抽样点的shape

        This is a coarse search with grid points matching,which is similar to the coarse search in caps
        in fact this coarse matching is just for ablation, and the results are not used in the final loss
        you can comment out this search
        这里包含了一部分粗略匹配的代码，类似于CAPS中的粗略匹配
        粗匹配的结果是最开始实验时进行的探索，实际上并没有用于最后的损失函数计算
        可以注释掉粗略匹配的代码
        """

        coord1_n, coord2_n, score1, score2 = self.kps_generator(inputs, outputs, **self.config['kps_generator_config'])
        _, hkps, wkps, _ = coord1_n.shape
        coord1 = denormalize_coords(coord1_n.reshape(b,-1,2), h1i, w1i)
        coord2 = denormalize_coords(coord2_n.reshape(b,-1,2), h2i, w2i)

        feat1_fine = sample_feat_by_coord(xf1, coord1_n.reshape(b,-1,2), self.config['loss_distance']=='cos')
        feat2_fine = sample_feat_by_coord(xf2, coord2_n.reshape(b,-1,2), self.config['loss_distance']=='cos')

        cos_sim = feat1_fine @ feat2_fine.transpose(1,2) # bxmxn
        feat1g_corloc = (F.softmax(temperature*cos_sim, dim=2)).unsqueeze(-1)*coord2.reshape(b,-1,2).unsqueeze(1) #bxmxnx2
        feat1g_corloc = feat1g_corloc.sum(2) #bxmx2
        feat2g_corloc = (F.softmax(temperature*cos_sim, dim=1)).unsqueeze(-1)*coord1.reshape(b,-1,2).unsqueeze(2) #bxmxnx2
        feat2g_corloc = feat2g_corloc.sum(1) #bxnx2


        with torch.no_grad():
            if self.config['use_nn_grid']:
                _, max_idx1 = cor_mat.max(2)
                feat1g_corloc_n = coord2_n.reshape(b,-1,2).gather(dim=1, index=max_idx1[:,:,None].repeat(1,1,2))
                _, max_idx2 = cor_mat.max(1)
                feat2g_corloc_n = coord1_n.reshape(b,-1,2).gather(dim=1, index=max_idx2[:,:,None].repeat(1,1,2))
            else:
                feat1g_corloc_n = normalize_coords(feat1g_corloc, h2i, w2i)
                feat2g_corloc_n = normalize_coords(feat2g_corloc, h1i, w1i)

        feat1g_std = (F.softmax(temperature*cos_sim, dim=2)).unsqueeze(-1)*(coord2_n.reshape(b,1,-1,2)**2)
        feat1g_std = feat1g_std.sum(2) - (feat1g_corloc_n**2)
        feat1g_std = feat1g_std.clamp(min=1e-6).sqrt().sum(-1) #bxn
        feat2g_std = (F.softmax(temperature*cos_sim, dim=1)).unsqueeze(-1)*(coord1_n.reshape(b,-1,1,2)**2)
        feat2g_std = feat2g_std.sum(1) - (feat2g_corloc_n**2)
        feat2g_std = feat2g_std.clamp(min=1e-6).sqrt().sum(-1) #bxn

        if self.config['use_line_search']:
            feat1_c_corloc_n_, feat1_c_corloc_n_org, valid1, epi_std1 = epipolar_line_search(coord1, inputs['F1'], feat1_fine, 
                temperature*F.normalize(xf2,p=2.0,dim=1), h2i, w2i, window_size=self.config['window_size'], **self.config['line_search_config'])
            feat2_c_corloc_n_, feat2_c_corloc_n_org, valid2, epi_std2 = epipolar_line_search(coord2, inputs['F2'], feat2_fine, 
                temperature*F.normalize(xf1,p=2.0,dim=1), h1i, w1i, window_size=self.config['window_size'], **self.config['line_search_config'])
            feat1c_corloc_org = denormalize_coords(feat1_c_corloc_n_org, h2i, w2i)
            feat2c_corloc_org = denormalize_coords(feat2_c_corloc_n_org, h1i, w1i)
        else:
            feat1_c_corloc_n_ = feat1g_corloc_n.detach()
            feat2_c_corloc_n_ = feat2g_corloc_n.detach()
            feat1c_corloc_org = feat1_c_corloc_n_
            feat2c_corloc_org = feat2_c_corloc_n_
            valid1 = torch.ones_like(feat1g_std).bool()
            valid2 = torch.ones_like(feat2g_std).bool()

        feat1w_corloc_n, window_coords_n_1in2, feat1w_std, _ = get_expected_correspondence_within_window(
            feat1_fine, temperature*F.normalize(xf2,p=2.0,dim=1), feat1_c_corloc_n_, self.config['window_size'], with_std=True)
        feat2w_corloc_n, window_coords_n_2in1, feat2w_std, _ = get_expected_correspondence_within_window(
            feat2_fine, temperature*F.normalize(xf1,p=2.0,dim=1), feat2_c_corloc_n_, self.config['window_size'], with_std=True)

        feat1w_corloc = denormalize_coords(feat1w_corloc_n, h2i, w2i)
        feat2w_corloc = denormalize_coords(feat2w_corloc_n, h1i, w1i)

        return {
                'coord1':coord1, 'coord2':coord2,
                'feat1g_corloc':feat1g_corloc,
                'feat2g_corloc':feat2g_corloc,
                'feat1w_corloc':feat1w_corloc,
                'feat2w_corloc':feat2w_corloc,
                'feat1c_corloc_org':feat1c_corloc_org,
                'feat2c_corloc_org':feat2_c_corloc_n_org,
                'feat1g_std':feat1g_std, 'feat2g_std':feat2g_std,
                'feat1w_std':feat1w_std, 'feat2w_std':feat2w_std,
                'temperature':temperature,
                'valid_epi1':valid1, 'valid_epi2':valid2
                }

class Preprocess_Skip(nn.Module):
    '''
    the preprocess class for keypoint detection net training
    '''
    def __init__(self, **kargs):
        super(Preprocess_Skip, self).__init__()
        self.__lossname__ = 'Preprocess_Skip'

    def forward(self, inputs, outputs):
        return None
