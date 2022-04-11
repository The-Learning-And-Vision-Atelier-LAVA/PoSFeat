import torch
import torch.nn as nn
import torch.nn.functional as F

class KeypointDet(nn.Module):
    """
    spatical attention header 
    """
    def __init__(self, in_channels, out_channels=1, prior='SSIM', act='Sigmoid'):
        super(KeypointDet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels+64, 128, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(128)
        self.conv3 = nn.Conv2d(128, out_channels, 1, 1, 0)
        self.norm3 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.PReLU()
        self.prior = getattr(self, prior)
        self.act = getattr(nn, act)()

        self.convimg = nn.Conv2d(3, 64, 3, 1, 1)
        self.normimg = nn.InstanceNorm2d(64)

    def SSIM(self, x):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        x_pad = F.pad(x.abs(), (0,1,0,1), 'reflect')
        x_lu = x_pad[:,:,:-1,:-1]
        x_rb = x_pad[:,:,1:,1:]

        x_lu = F.pad(x_lu, (1,1,1,1), 'reflect')
        x_rb = F.pad(x_rb, (1,1,1,1), 'reflect')

        m_x_lu = F.avg_pool2d(x_lu, 3, 1)
        m_x_rb = F.avg_pool2d(x_rb, 3, 1)

        sigma_x_lu = F.avg_pool2d(x_lu**2, 3, 1) - m_x_lu**2
        sigma_x_rb = F.avg_pool2d(x_rb**2, 3, 1) - m_x_rb**2
        sigma_x_lu_rb = F.avg_pool2d(x_lu*x_rb, 3, 1) - m_x_lu*m_x_rb

        SSIM_n = (2 * m_x_lu * m_x_rb + C1) * (2 * sigma_x_lu_rb + C2)
        SSIM_d = (m_x_lu ** 2 + m_x_rb ** 2 + C1) * (sigma_x_lu + sigma_x_rb + C2)

        return torch.clamp((1 - SSIM_n / SSIM_d)/2, 0, 1)

    def D2(self, x):
        b,c,h,w = x.shape
        window_size = 3
        padding_size = window_size//2

        x = F.relu(x)
        max_per_sample = torch.max(x.view(b,-1), dim=1)[0]
        exp = torch.exp(x/max_per_sample.view(b,1,1,1))
        sum_exp = (
            window_size**2*
                F.avg_pool2d(
                    F.pad(exp, [padding_size]*4, mode='constant', value=1.),
                    window_size, stride=1
                    )
                )

        local_max_score = exp / sum_exp

        depth_wise_max = torch.max(x, dim=1)[0]
        depth_wise_max_score = x / depth_wise_max.unsqueeze(1)

        all_scores = local_max_score * depth_wise_max_score
        score = torch.max(all_scores, dim=1)[0]

        # score = score / torch.sum(score.view(b, -1), dim=1).view(b, 1, 1)

        return score.unsqueeze(1)

    def ASL_Peak(self, x):
        b,c,h,w = x.shape
        window_size = 3
        padding_size = window_size//2

        # x = F.relu(x)
        max_per_sample = torch.max(x.view(b,-1), dim=1)[0]
        x = x/max_per_sample.view(b,1,1,1)

        alpha_input = x - F.avg_pool2d(
                    F.pad(x, [padding_size]*4, mode='reflect'),
                    window_size, stride=1
                    )
        alpha = F.softplus(alpha_input)

        beta_input = x - x.mean(1, True)
        beta = F.softplus(beta_input)

        all_scores = (alpha*beta).max(1,True)[0]

        return all_scores

    def identity(self, x):
        scores = torch.ones_like(x)
        return scores.mean(1,True)


    def forward(self, fine_maps):
        fine_map = fine_maps[0]
        img_tensor = fine_maps[1]
        x_pf = self.prior(fine_map)
        x_pi = self.prior(img_tensor)

        x = self.relu(self.norm1(self.conv1(x_pf*fine_map)))
        x = F.interpolate(x, img_tensor.shape[2:], align_corners=False, mode='bilinear')
        img_tensor = self.normimg(self.convimg(x_pi*img_tensor))
        x = torch.cat([x, img_tensor], dim=1)
        x = self.relu(self.norm2(self.conv2(x)))
        score = self.act(self.norm3(self.conv3(x)))

        # thr = self.act(self.conv_thr(x))
        # score = self.relu(score-thr)

        score =F.interpolate(x_pf, img_tensor.shape[2:], align_corners=False, mode='bilinear').mean(1,True) * \
             x_pi.mean(1,True) * score

        return score