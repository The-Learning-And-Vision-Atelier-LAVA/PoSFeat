import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical, Bernoulli
import math
import numpy as np

def homogenize(coord):
    # coord = torch.cat((coord, torch.ones_like(coord[:, :, [0]])), -1)
    coord = torch.cat((coord, torch.ones_like(coord[..., [0]])), -1)
    return coord

def normalize_coords(coord, h, w):
    '''
    turn the coordinates from pixel indices to the range of [-1, 1]
    :param coord: [..., 2]
    :param h: the image height
    :param w: the image width
    :return: the normalized coordinates [..., 2]
    '''
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).to(coord.device).float()
    # print(coord[:,:,0].max(), coord[:,:,1].max(), w, h)
    coord_norm = (coord - c) / c
    # print(coord_norm[:,:,0].max(), coord_norm[:,:,1].max(), coord_norm[:,:,0].min(), coord_norm[:,:,1].min())
    return coord_norm

def denormalize_coords(coord_norm, h, w):
    '''
    turn the coordinates from normalized value ([-1, 1]) to actual pixel indices
    :param coord_norm: [..., 2]
    :param h: the image height
    :param w: the image width
    :return: actual pixel coordinates
    '''
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).to(coord_norm.device)
    coord = coord_norm * c + c
    return coord

def sample_feat_by_coord(x, coord_n, norm=False):
    '''
    sample from normalized coordinates
    :param x: feature map [batch_size, n_dim, h, w]
    :param coord_n: normalized coordinates, [batch_size, n_pts, 2]
    :param norm: if l2 normalize features
    :return: the extracted features, [batch_size, n_pts, n_dim]
    '''
    feat = F.grid_sample(x, coord_n.unsqueeze(2), padding_mode='zeros', align_corners=False).squeeze(-1)
    # print(feat.shape)
    if norm:
        feat = F.normalize(feat, p=2, dim=1)
    feat = feat.transpose(1, 2)
    return feat

def get_expected_correspondence_locs(feat1, featmap2, with_std=False):
    '''
    compute the expected correspondence locations
    :param feat1: the feature vectors of query points [batch_size, n_pts, n_dim]
    :param featmap2: the feature maps of the reference image [batch_size, n_dim, h, w]
    :param with_std: if return the standard deviation
    :return: the normalized expected correspondence locations [batch_size, n_pts, 2]
    '''
    B, d, h2, w2 = featmap2.size()
    grid_n = gen_grid(-1, 1, -1, 1, h2, w2).to(featmap2.device)
    featmap2_flatten = featmap2.reshape(B, d, h2*w2).transpose(1, 2)  # BX(hw)xd
    prob = compute_prob(feat1, featmap2_flatten)  # Bxnx(hw)

    grid_n = grid_n.unsqueeze(0).unsqueeze(0)  # 1x1x(hw)x2
    expected_coord_n = torch.sum(grid_n * prob.unsqueeze(-1), dim=2)  # Bxnx2

    if with_std:
        # convert to normalized scale [-1, 1]
        var = torch.sum(grid_n**2 * prob.unsqueeze(-1), dim=2) - expected_coord_n**2  # Bxnx2
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # Bxn
        # var_prob = (prob-prob.mean(-1,True)).square().sum(-1,True)/prob.shape[-1] # Bxnx1
        # kurtosis = torch.pow(prob-prob.mean(-1,True),4).sum(-1,True)/(prob.shape[2]*var_prob**2)
        kurtosis = torch.pow(grid_n-expected_coord_n.unsqueeze(-2), 4).mean(-2)/torch.pow(var, 2)
        kurtosis = (kurtosis/10.).clamp(0,1)
        # kurtosis = var
        return expected_coord_n, std, kurtosis.mean(-1), prob#, var_prob
    else:
        return expected_coord_n

def gen_grid(h_min, h_max, w_min, w_max, len_h, len_w):
    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w), torch.linspace(h_min, h_max, len_h)])
    grid = torch.stack((x, y), -1).transpose(0, 1).reshape(-1, 2).float()
    return grid

def compute_prob(feat1, feat2, loss_distance='cos', with_scale=False, return_sim=False):
    '''
    compute probability
    :param feat1: query features, [batch_size, m, n_dim]
    :param feat2: reference features, [batch_size, n, n_dim]
    :return: probability, [batch_size, m, n]
    '''
    assert loss_distance in ['cos', 'euc']
    if return_sim:
        assert loss_distance=='cos'
    if loss_distance == 'cos':
        sim = feat1.bmm(feat2.transpose(1, 2))
        if with_scale:
            scale = sim.new_tensor(feat2.shape[1])
            scale = scale.sqrt()
        else:
            scale = 1
        prob = F.softmax(scale*sim, dim=-1)  # Bxmxn
    else:
        dist = torch.sum(feat1**2, dim=-1, keepdim=True) + \
               torch.sum(feat2**2, dim=-1, keepdim=True).transpose(1, 2) - \
               2 * feat1.bmm(feat2.transpose(1, 2))
        prob = F.softmax(-dist, dim=-1)  # Bxmxn
    if return_sim:
        return prob, sim
    else:
        return prob

def OT_sinkhorn_log(costs, iters=20, temperature=None):
    '''
    find the correspondece with sinkhorn algorithm
    :param costs: [b, m, n]
    :param iters: the number of iterations
    :return: the optimized scores [b,m,n]
    '''
    b, m, n = costs.shape
    one = costs.new_tensor(1)
    ms, ns = (m*one).to(costs), (n*one).to(costs)
    norm = - (ms + ns).log()

    P = -temperature*costs
    log_m = norm*torch.ones(b,m,1).to(costs)
    log_n = norm*torch.ones(b,1,n).to(costs)
    u,v = torch.zeros_like(log_m), torch.zeros_like(log_n)
    for _ in range(iters):
        u = log_m - torch.logsumexp(P + v, dim=2, keepdim=True)
        v = log_n - torch.logsumexp(P + u, dim=1, keepdim=True)
    P = P + u + v
    P = P - norm
    optimal = P.exp()

    return optimal, None

def OT_sinkhorn_log_unmatch(costs, iters=20, temperature=None):
    '''
    find the correspondece with sinkhorn algorithm
    :param costs: [b, m, n]
    :param iters: the number of iterations
    :return: the optimized scores [b,m,n]
    '''
    b, m, n = costs.shape
    one = costs.new_tensor(1)
    ms, ns = (m*one).to(costs), (n*one).to(costs)

    bins1 = 1-costs.min(2, True)[0] #bxmx1
    bins2 = 1-costs.min(1, True)[0] #bx1xn
    corner = (bins1.mean(1,True)+bins2.mean(2,True))/2

    costs = torch.cat([torch.cat([costs, bins1], -1),
                       torch.cat([bins2, corner], -1)], 1) #bx(m+1)x(n+1)
    norm = - (ms + ns).log()

    P = -temperature*costs
    log_m = norm*torch.ones(b,m+1,1).to(costs)
    log_n = norm*torch.ones(b,1,n+1).to(costs)
    log_m[:,-1,:] = ns.log()+norm
    log_n[:,:,-1] = ms.log()+norm
    u,v = torch.zeros_like(log_m), torch.zeros_like(log_n)
    for _ in range(iters):
        u = log_m - torch.logsumexp(P + v, dim=2, keepdim=True)
        v = log_n - torch.logsumexp(P + u, dim=1, keepdim=True)
    P = P + u + v
    P = P - norm
    optimal = P.exp()

    return optimal[:, :-1, :-1], optimal

def Dual_Softmax(costs, iters=None, temperature=None):
    '''
    find the correspondece with sinkhorn algorithm
    :param costs: [b, m, n]
    :param iters: the number of iterations
    :return: the optimized scores [b,m,n]
    '''
    b, m, n = costs.shape
    # scale = max(m,n)
    scale = 1
    if temperature is None:
        costs_input = - 15 * scale * costs
    else:
        costs_input = - temperature * scale * costs
    prob_col = F.softmax(costs_input, dim=2)
    prob_row = F.softmax(costs_input, dim=1)
    prob = prob_col*prob_col

    return prob, None

def generate_kpts(inputs, outputs, nms_radius, num_pts=False, stable_prob=0.9, use_nms=True, stride=1):
    """
    generate keypoints on the entire image
    """
    preds1 = outputs['preds1']
    preds2 = outputs['preds2']
    kp_map1, kp_map2 = preds1['local_point'], preds2['local_point']

    if torch.rand(1)<stable_prob: # stable select
        kps1, kp_score1 = generate_kpts_single(kp_map1, nms_radius, num_pts, scale=4, stride=stride, use_nms=use_nms)
        kps2, kp_score2 = generate_kpts_single(kp_map2, nms_radius, num_pts, scale=4, stride=stride, use_nms=use_nms)
    else: # random select
        temperature = 0.01/(outputs['epoch']+1)
        kps1, kp_score1 = generate_kpts_single(kp_map1, nms_radius, num_pts, scale=4, 
            stable=False, temperature=temperature, stride=stride, use_nms=use_nms)
        kps2, kp_score2 = generate_kpts_single(kp_map2, nms_radius, num_pts, scale=4, 
            stable=False, temperature=temperature, stride=stride, use_nms=use_nms)
    return kps1, kps2, kp_score1, kp_score2

def generate_kpts_single(kp_map, nms_radius, num_pts=False, scale=4, stable=True, temperature=1, stride=1, use_nms=True, thr=False, thr_mod='mean'):
    b, _, h, w = kp_map.shape
    grids_org = gen_grid(h_min=-1, h_max=1, w_min=-1, w_max=1, len_h=h, len_w=w)
    # h, w = scale*h, scale*w
    # grids_org = gen_grid(h_min=0, h_max=h-1, w_min=0, w_max=w-1, len_h=h, len_w=w)
    grids_org = grids_org.reshape(h, w, 2)[None, :, :, :].repeat(b, 1, 1, 1).to(kp_map)
    grids_org = grids_org.permute(0,3,1,2) # bx2xhxw

    # nms omits the boarder pixels of the original score map 
    # so that the mask size will be the same as processed score map
    if use_nms == 'softnms': # softnms for softnms
        nms_mask = soft_nms(kp_map[:,:,1:-1,1:-1], nms_radius)
    elif use_nms: # True for hard nms
        nms_mask = nms(kp_map[:,:,1:-1,1:-1], nms_radius) 
    elif not use_nms: # False for no nms
        nms_mask = torch.ones((b,1,h-2,w-2)).to(kp_map)

    if thr :
        if thr_mod == 'max':
            kp_thr = (kp_map[:,:,1:-1,1:-1]).reshape(b,1,-1).max(2)[0]
        elif thr_mod == 'mean':
            kp_thr = (kp_map[:,:,1:-1,1:-1]).reshape(b,1,-1).mean(2)
        elif thr_mod == 'abs':
            kp_thr = torch.tensor(1.).to(kp_map).repeat(b)
        thr_mask = kp_map[:,:,1:-1,1:-1]>thr*kp_thr.view(b,1,1,1)
        nms_mask = thr_mask*nms_mask

    # process the score map and grids
    grids = kp_map*grids_org
    grids = F.avg_pool2d(grids, 3, stride=stride, padding=0)
    kp_weight = F.avg_pool2d(kp_map, 3, stride=stride, padding=0)
    grids = grids/kp_weight
    kp_score_map = F.max_pool2d(kp_map, 3, stride=stride, padding=0)

    if not num_pts:
        if use_nms != 'softnms':
            num_pts = (nms_mask.view(b,-1).sum(1).min()).int()
        else:
            # num_pts = (((nms_mask*kp_map[:,:,1:-1,1:-1]).view(b,-1)>thr*((nms_mask*kp_map[:,:,1:-1,1:-1]).view(b,-1).mean(1, True))).sum(1).min()).int()
            num_pts = (thr_mask.view(b,-1).sum(1).min()).int()
    else:
        if use_nms != 'softnms' and num_pts>nms_mask.view(b,-1).sum(1).min():
            num_pts = (nms_mask.view(b,-1).sum(1).min()).int()
        if use_nms == 'softnms' and num_pts>thr_mask.view(b,-1).sum(1).min():
            num_pts = (thr_mask.view(b,-1).sum(1).min()).int()
    if num_pts < 128:
        num_pts = 128

    if stable:
        _, idx = (nms_mask*kp_map[:,:,1:-1,1:-1]).permute(0,2,3,1).contiguous().view(b,-1).topk(num_pts)

        kps = grids.permute(0,2,3,1).view(b,-1,2).gather(dim=1,index=idx.unsqueeze(-1).repeat(1,1,2))
        kp_score = kp_score_map.permute(0,2,3,1).view(b,-1,1).gather(dim=1,index=idx.unsqueeze(-1))
    else:
        # select = gumbel_softmax(kp_map, num_pts, temperature) # bxnxhw

        # kps = select@grids_org.permute(0,2,3,1).view(b,h*w,2)
        # kp_score = select@kp_map.permute(0,2,3,1).view(b,h*w,1)
        select = gumbel_softmax(nms_mask*kp_map[:,:,1:-1,1:-1], num_pts, temperature) # bxnxhw

        kps = select@grids.permute(0,2,3,1).reshape(b,(h-2)*(w-2),2)
        kp_score = select@kp_map[:,:,1:-1,1:-1].permute(0,2,3,1).reshape(b,(h-2)*(w-2),1)

    return kps, kp_score

def generate_kpts_single_noavg(kp_map, nms_radius, num_pts=False, scale=4, stable=True, temperature=1, stride=1, use_nms=True, thr=False, thr_mod='mean'):
    b, _, h, w = kp_map.shape
    grids_org = gen_grid(h_min=-1, h_max=1, w_min=-1, w_max=1, len_h=h, len_w=w)
    # h, w = scale*h, scale*w
    # grids_org = gen_grid(h_min=0, h_max=h-1, w_min=0, w_max=w-1, len_h=h, len_w=w)
    grids_org = grids_org.reshape(h, w, 2)[None, :, :, :].repeat(b, 1, 1, 1).to(kp_map)
    grids_org = grids_org.permute(0,3,1,2) # bx2xhxw

    # nms omits the boarder pixels of the original score map 
    # so that the mask size will be the same as processed score map
    if use_nms == 'softnms': # softnms for softnms
        nms_mask = soft_nms(kp_map, nms_radius)
    elif use_nms: # True for hard nms
        nms_mask = nms(kp_map, nms_radius) 
    elif not use_nms: # False for no nms
        nms_mask = torch.ones((b,1,h,w)).to(kp_map)

    if thr :
        if thr_mod == 'max':
            kp_thr = (kp_map).reshape(b,1,-1).max(2)[0]
        elif thr_mod == 'mean':
            kp_thr = (kp_map).reshape(b,1,-1).mean(2)
        thr_mask = kp_map>thr*kp_thr.view(b,1,1,1)
        nms_mask = thr_mask*nms_mask

    grids = grids_org

    if not num_pts:
        if use_nms != 'softnms':
            num_pts = (nms_mask.view(b,-1).sum(1).min()).int()
        else:
            # num_pts = (((nms_mask*kp_map[:,:,1:-1,1:-1]).view(b,-1)>thr*((nms_mask*kp_map[:,:,1:-1,1:-1]).view(b,-1).mean(1, True))).sum(1).min()).int()
            num_pts = (thr_mask.view(b,-1).sum(1).min()).int()
    else:
        if use_nms != 'softnms' and num_pts>nms_mask.view(b,-1).sum(1).min():
            num_pts = (nms_mask.view(b,-1).sum(1).min()).int()
        if use_nms == 'softnms' and num_pts>thr_mask.view(b,-1).sum(1).min():
            num_pts = (thr_mask.view(b,-1).sum(1).min()).int()
    if num_pts < 128:
        num_pts = 128

    if stable:
        _, idx = (nms_mask*kp_map).permute(0,2,3,1).contiguous().view(b,-1).topk(num_pts)

        kps = grids.permute(0,2,3,1).view(b,-1,2).gather(dim=1,index=idx.unsqueeze(-1).repeat(1,1,2))
        kp_score = kp_map.permute(0,2,3,1).view(b,-1,1).gather(dim=1,index=idx.unsqueeze(-1))
    else:
        # select = gumbel_softmax(kp_map, num_pts, temperature) # bxnxhw

        # kps = select@grids_org.permute(0,2,3,1).view(b,h*w,2)
        # kp_score = select@kp_map.permute(0,2,3,1).view(b,h*w,1)
        select = gumbel_softmax(nms_mask*kp_map, num_pts, temperature) # bxnxhw

        kps = select@grids.permute(0,2,3,1).reshape(b,(h-2)*(w-2),2)
        kp_score = select@kp_map.permute(0,2,3,1).reshape(b,(h-2)*(w-2),1)

    return kps, kp_score

# def unfold(tensor, grid_size):
#     b,c,h,w = tensor.shape 
#     unfold_tensor = tensor.unfold(2, grid_size, grid_size).unfold(3, grid_size, grid_size) \
#         .reshape(b, c, h//grid_size, w//grid_size, grid_size*grid_size)
#     return unfold_tensor

def unfold(tensor, grid_size, stride=None):
    if stride is None:
        stride = grid_size
    unfold_tensor = tensor.unfold(2, grid_size, stride).unfold(3, grid_size, stride)
    b,c,h,w,g1,g2 = unfold_tensor.shape
    unfold_tensor = unfold_tensor.reshape(b,c,h,w,g1*g2)
    return unfold_tensor

def regular_sample(tensor):
    b,c,h,w,g = tensor.shape 
    idx = torch.multinomial(tensor.reshape(-1,g), 1)
    idx = idx.reshape(b,c,h,w,1)
    return idx

def generate_kpts_regular_grid(inputs, outputs, grid_size, num_pts=False, stable_prob=0.9, use_nms=True, nms_radius=None):
    preds1 = outputs['preds1']
    preds2 = outputs['preds2']
    kp_map1, kp_map2 = preds1['local_point'], preds2['local_point']

    if torch.rand(1)<stable_prob: # stable select
        kps1, kp_score1 = generate_kpts_regular_grid_single(kp_map1, grid_size, num_pts, scale=4, stable=True, use_nms=use_nms, 
            nms_radius=nms_radius)
        kps2, kp_score2 = generate_kpts_regular_grid_single(kp_map2, grid_size, num_pts, scale=4, stable=True, use_nms=use_nms, 
            nms_radius=nms_radius)
    else: # random select
        kps1, kp_score1 = generate_kpts_regular_grid_single(kp_map1, grid_size, num_pts, scale=4, stable=False, use_nms=use_nms, 
            nms_radius=nms_radius)
        kps2, kp_score2 = generate_kpts_regular_grid_single(kp_map2, grid_size, num_pts, scale=4, stable=False, use_nms=use_nms, 
            nms_radius=nms_radius)
    return kps1, kps2, kp_score1, kp_score2

def generate_kpts_regular_grid_single(kp_map, grid_size, num_pts=False, scale=4, stable=True, use_nms=True, nms_radius=None, thr=None, thr_mod='mean'):
    b, _, h, w = kp_map.shape
    grids_org = gen_grid(h_min=-1, h_max=1, w_min=-1, w_max=1, len_h=h, len_w=w)
    # h, w = scale*h, scale*w
    # grids_org = gen_grid(h_min=0, h_max=h-1, w_min=0, w_max=w-1, len_h=h, len_w=w)
    grids_org = grids_org.reshape(h, w, 2)[None, :, :, :].repeat(b, 1, 1, 1).to(kp_map)
    grids_org = grids_org.permute(0,3,1,2) # bx2xhxw
    if use_nms == 'softnms':
        soft_mask = soft_nms(kp_map, nms_radius)
        kp_map = soft_mask*kp_map
        nms_mask = torch.ones_like(soft_mask).bool()
    elif use_nms:
        nms_mask = nms(kp_map, nms_radius)
    else:
        nms_mask = torch.ones_like(kp_map).bool()

    if thr is not None:
        if thr_mod == 'max':
            kp_thr = kp_map.view(b,1,-1).max(2)[0]
        elif thr_mod == 'mean':
            kp_thr = kp_map.view(b,1,-1).mean(2)
        thr_mask = kp_map>thr*kp_thr.view(b,1,1,1)
        nms_mask = thr_mask&nms_mask

    grids_unfold = unfold(grids_org, grid_size)
    kpmap_unfold = unfold(kp_map, grid_size) 
    nms_unfold = unfold(nms_mask, grid_size)

    kpmap_unfold_n = F.softmax(kpmap_unfold, dim=4)
    if stable:
        idx = kpmap_unfold_n.argmax(-1,True)
    else:
        idx = regular_sample(kpmap_unfold_n)

    kps = grids_unfold.gather(dim=4, index=idx.repeat(1,2,1,1,1)) # bx2x(h//g)x(w//g)x1
    kp_score = kpmap_unfold.gather(dim=4, index=idx) # bx1x(h//g)x(w//g)x1
    mask = nms_unfold.gather(dim=4, index=idx) # bx1x(h//g)x(w//g)x1

    kps = kps.reshape(b,2,-1).transpose(1,2) # bxnx2
    kp_score = kp_score.reshape(b,1,-1).transpose(1,2) # bxnx1
    mask = mask.reshape(b,1,-1).transpose(1,2) # bxnx1

    if num_pts:
        if num_pts > mask.sum(1).min():
            num_pts=mask.sum(1).min()
        kp_score, top_idx = (mask*kp_score).topk(num_pts, dim=1)
        kps = kps.gather(dim=1, index=top_idx)
    else:
        if use_nms :
            num_pts=mask.sum(1).min()
            if num_pts < 128:
                num_pts = 128
            kp_score, top_idx = (mask*kp_score).topk(num_pts, dim=1)
            kps = kps.gather(dim=1, index=top_idx.repeat(1,1,2))
    return kps, kp_score

def soft_nms(score, patch_radius):
    b,c,h,w = score.shape
    window_size = 2*patch_radius + 1
    padding_size = patch_radius

    score = score.detach().contiguous()
    # max_per_sample = torch.max(score.view(b,-1), dim=1)[0]
    # score = score/max_per_sample.view(b,1,1,1)
    # score = score.detach()

    alpha_input = score - F.avg_pool2d(
                F.pad(score, [padding_size]*4, mode='reflect'),
                window_size, stride=1
                )
    alpha = F.softplus(alpha_input)

    return alpha

def nms(score, patch_radius):
    patch_size = 2*patch_radius+1
    score_pad = F.pad(score.detach(), (patch_radius, patch_radius, patch_radius, patch_radius), mode='reflect')
    # max_score = F.max_pool2d(score_pad, patch_size, stride=1, padding=0)
    # mask = score==max_score

    _, idx = F.max_pool2d(score_pad, patch_size, stride=1, padding=0, return_indices=True)
    # if len(idx.shape) == 4:
    #     assert idx.shape[0] == 1
    #     idx = idx.squeeze(0)
    b,_, h, w = score.shape
    coords = torch.arange((h+2*patch_radius) * (w+2*patch_radius), device=score.device)\
        .reshape(1, 1, h+2*patch_radius, w+2*patch_radius).repeat(b,1,1,1)
    coords = coords[:,:,patch_radius:-patch_radius,patch_radius:-patch_radius]
    mask = idx == coords
    return mask


def gumbel_noise(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(prob, num_points, temperature=1):
    b, one, h, w = prob.shape
    y = prob.view(b,1,h*w).repeat(1, num_points, 1) + gumbel_noise((b, num_points, h*w))
    one_hot_soft = F.softmax(y/temperature, dim=2)
    return one_hot_soft

def gumbel_softmax(prob, num_points, temperature=1, hard=False):
    one_hot_soft = gumbel_softmax_sample(prob, num_points, temperature) # bx1xhw
    if not hard:
        return one_hot_soft
    b, num, hw = one_hot_soft.shape
    _, idx = one_hot_soft.max(dim=2)
    one_hot = torch.zeros_like(one_hot_soft).view(-1, hw)
    one_hot.scatter(dim=2, index=idx.view(-1, 1), src=1)
    one_hot = one_hot.view(b, num, hw)
    one_hot = (one_hot - one_hot_soft).detach() + one_hot
    return one_hot

@torch.no_grad()
def valid_points(epipolar_line, im_size, linelen_thr):
    '''
    this function is actually the same as get_endpoints
    return endpoints1 endpoints2 bxnx2
    return valid bxn
    '''
    batch_size, _, n_pts = epipolar_line.shape
    h, w = im_size
    a = epipolar_line[:,0,:] #Bxn
    b = epipolar_line[:,1,:]
    c = epipolar_line[:,2,:]
    point_l = torch.stack([torch.zeros_like(a), -c/b], -1) #Bxnx2
    point_r = torch.stack([(w-1)*torch.ones_like(a), -(a*(w-1)+c)/b], -1)
    point_u = torch.stack([-(b*(h-1)+c)/a, (h-1)*torch.ones_like(a)], -1)
    point_b = torch.stack([-c/a, torch.zeros_like(a)], -1)
    points = torch.stack([point_l, point_r, point_u, point_b], -1).transpose(2,3) #Bxnx4x2
    mask = (points[:,:,:,0]>=0) & (points[:,:,:,0]<=w-1) & (points[:,:,:,1]>=0) & (points[:,:,:,1]<=h-1) #Bxnx4
    valid = mask.sum(-1) == 2 #Bxn

    mask[~valid] = torch.tensor([True, True, False, False]).to(mask.device)
    points = points[mask].reshape(batch_size, n_pts, 2, 2)
    points1 = points[:,:,0,:]
    points2 = points[:,:,1,:]
    endpoints_1_n = normalize_coords(points1, h, w)
    endpoints_2_n = normalize_coords(points2, h, w)
    line_len = endpoints_2_n - endpoints_1_n
    len_mask = (line_len**2).sum(-1).sqrt()>linelen_thr
    valid = valid&len_mask

    return valid

@torch.no_grad()
def SSIM(x):
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

    return torch.clamp((1 - SSIM_n / SSIM_d)/2, 0, 1).mean(1,True)

@torch.no_grad()
def D2(x):
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

@torch.no_grad()
def ASL_Peak(x):
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

@torch.no_grad()
def generate_kpts_regular_grid_random(inputs, outputs, grid_size, map_init='identity', keep_spatial=False, random_select='random'):
    """
    this is the function used to generate key points within regualr grid in descriptor initialization stage
    """
    preds1 = outputs['preds1']
    preds2 = outputs['preds2']
    if map_init == 'identity':
        kp_map1, kp_map2 = torch.ones_like(preds1['local_point']), torch.ones_like(preds2['local_point'])
    elif map_init in ['SSIM', 'D2', 'ASL_Peak']:
        func = eval(map_init)
        kp_map1 = func(F.interpolate(preds1['local_map'], inputs['im1'].shape[2:], mode='bilinear'))
        kp_map2 = func(F.interpolate(preds2['local_map'], inputs['im2'].shape[2:], mode='bilinear'))

    kps1, kp_score1 = generate_kpts_regular_grid_random_single(kp_map1, grid_size, random_select)
    kps2, kp_score2 = generate_kpts_regular_grid_random_single(kp_map2, grid_size, random_select)

    if not keep_spatial:
        b = kps1.shape[0]
        kps1, kps2 = kps1.reshape(b,2,-1).transpose(1,2), kps2.reshape(b,2,-1).transpose(1,2)
        kp_score1, kp_score2 = kp_score1.reshape(b,1,-1).transpose(1,2), kp_score2.reshape(b,1,-1).transpose(1,2)
    else:
        kps1, kps2 = kps1.squeeze(-1).permute(0,2,3,1), kps2.squeeze(-1).permute(0,2,3,1)
        kp_score1, kp_score2 = kp_score1.permute(0,2,3,1), kp_score2.permute(0,2,3,1)
    return kps1, kps2, kp_score1, kp_score2

def generate_kpts_regular_grid_random_single(kp_map, grid_size, random_select):
    """
    note that the score returned by this function is the logp within the grid_size window
    """
    b, _, h, w = kp_map.shape
    if random_select == 'random':
        grids_org = gen_grid(h_min=-1, h_max=1, w_min=-1, w_max=1, len_h=h, len_w=w)
        grids_org = grids_org.reshape(h, w, 2)[None, :, :, :].repeat(b, 1, 1, 1).to(kp_map)
        grids_org = grids_org.permute(0,3,1,2) # bx2xhxw
        
        kpmap_unfold = unfold(kp_map, grid_size) # bx1x(h//g)x(w//g)x(g*g)
        proposal_dist = Categorical(logits=kpmap_unfold)
        proposals     = proposal_dist.sample() # bx1x(h//g)x(w//g)
        proposal_logp = proposal_dist.log_prob(proposals) # bx1x(h//g)x(w//g)
        kp_score = torch.gather(kpmap_unfold, dim=-1, index=proposals[..., None]).squeeze(-1) # bx1x(h//g)x(w//g)
        
        grids_unfold = unfold(grids_org, grid_size) # bx2x(h//g)x(w//g)x(g*g)
        kps = grids_unfold.gather(dim=4, index=proposals.unsqueeze(-1).repeat(1,2,1,1,1))
    elif random_select == 'regular_random':
        start = 0.5*grid_size/h
        num_w = w//grid_size
        num_h = h//grid_size
        kps = gen_grid(h_min=-1+start, h_max=1-start, w_min=-1+start, w_max=1-start, len_h=num_h, len_w=num_w)
        regular_rand = start*(2*torch.rand(b,1,1,2)-1).to(kp_map)
        kps = kps.reshape(num_h, num_w, 2)[None, :, :, :].repeat(b, 1, 1, 1).to(kp_map) + regular_rand
        kp_score = F.grid_sample(kp_map, kps, padding_mode='zeros', align_corners=False) # bx1x(h//g)x(w//g)
        kps = kps.permute(0,3,1,2)
    else:
        start = 0.5*grid_size/h
        num_w = w//grid_size
        num_h = h//grid_size
        kps = gen_grid(h_min=-1+start, h_max=1-start, w_min=-1+start, w_max=1-start, len_h=num_h, len_w=num_w)
        kps = kps.reshape(h, w, 2)[None, :, :, :].repeat(b, 1, 1, 1).to(kp_map)
        kp_score = F.grid_sample(kp_map, kps, padding_mode='zeros', align_corners=False) # bx1x(h//g)x(w//g)
        kps = kps.permute(0,3,1,2) # bx2x(h//g)x(w//g)
    return kps, kp_score

@torch.no_grad()
def epipolar_line_search(coord, Fmat, feat1, featmap2, h, w, line_step=100, use_nn=True, loc_rand=True, window_size=0.125, visualize=False):
    batch_size, n_dim, h2, w2 = featmap2.shape
    n_pts = coord.shape[1]
    endpoints_1_n, endpoints_2_n, valid=get_endpoints(coord, Fmat, h, w)
    sample_grids = torch.stack([torch.linspace(0., 1., line_step), torch.linspace(0., 1., line_step)], -1).to(coord.device) # stepx2
    line_len = endpoints_2_n - endpoints_1_n #bxnx2

    # weight_len = (line_len[:,:,0]**2+line_len[:,:,1]**2).sqrt() #bxn decide the weight according to the epipolar line length, which belongs to [0, 2*sqrt(2)]
    sample_grids = line_len[:,:,None,:]*sample_grids[None,None,:,:] #bxnxstepx2
    sample_grids = sample_grids+endpoints_1_n[:,:,None,:]

    sample_points = F.grid_sample(featmap2, sample_grids, padding_mode='border', align_corners=False).permute(0, 2, 3, 1)  # Bxnxstepxd
    prob = compute_prob(feat1.reshape(batch_size*n_pts, 1, n_dim), 
                sample_points.reshape(batch_size*n_pts, line_step, n_dim)).reshape(batch_size, n_pts, line_step)

    # expected_coord = torch.sum(sample_grids * prob.unsqueeze(-1), dim=2)  # Bxnx2
    if use_nn:
        mask = prob==prob.max(-1,True)[0]
        expected_coord = (mask.unsqueeze(-1)*sample_grids).sum(2) # bxnx2
    else:
        expected_coord = (prob.unsqueeze(-1)*sample_grids).sum(2)  # Bxnx2
    if loc_rand:
        expected_coord_org = expected_coord
        expected_coord = expected_coord + 0.707*window_size*(2*torch.rand(expected_coord.shape).type_as(expected_coord)-1)
    boarder_mask = (expected_coord[:,:,0]>=-1) & (expected_coord[:,:,0]<=1) & (expected_coord[:,:,1]>=-1) & (expected_coord[:,:,1]<=1)
    valid = valid & boarder_mask

    var = torch.sum(sample_grids**2 * prob.unsqueeze(-1), dim=2) - expected_coord**2  # Bxnx2
    std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1) 
    if visualize:
        return expected_coord, expected_coord_org, valid, std, prob
    else:
        return expected_coord, expected_coord_org, valid, std

@torch.no_grad()
def get_endpoints(coords, Fmat, h, w):
    '''
    return endpoints1 endpoints2 bxnx2
    return valid bxn
    '''
    batch_size, n_pts, _ = coords.shape
    coord_h = homogenize(coords).transpose(1, 2)
    epipolar_line = Fmat.bmm(coord_h)
    a = epipolar_line[:,0,:] #Bxn
    b = epipolar_line[:,1,:]
    c = epipolar_line[:,2,:]
    point_l = torch.stack([torch.zeros_like(a), -c/b], -1) #Bxnx2
    point_r = torch.stack([(w-1)*torch.ones_like(a), -(a*(w-1)+c)/b], -1)
    point_u = torch.stack([-(b*(h-1)+c)/a, (h-1)*torch.ones_like(a)], -1)
    point_b = torch.stack([-c/a, torch.zeros_like(a)], -1)
    points = torch.stack([point_l, point_r, point_u, point_b], -1).transpose(2,3) #Bxnx4x2
    mask = (points[:,:,:,0]>=0) & (points[:,:,:,0]<=w-1) & (points[:,:,:,1]>=0) & (points[:,:,:,1]<=h-1) #Bxnx4
    valid = mask.sum(-1) == 2 #Bxn
    mask[~valid] = torch.tensor([True, True, False, False]).to(mask.device)
    points = points[mask].reshape(batch_size, n_pts, 2, 2)
    points1 = points[:,:,0,:]
    points2 = points[:,:,1,:]
    return normalize_coords(points1,h,w), normalize_coords(points2,h,w), valid

def get_expected_correspondence_within_window(feat1, featmap2, coord2_n, window_size, with_std=False, with_sim=False):
    '''
    :param feat1: the feature vectors of query points [batch_size, n_pts, n_dim]
    :param featmap2: the feature maps of the reference image [batch_size, n_dim, h, w]
    :param coord2_n: normalized center locations [batch_size, n_pts, 2]
    :param with_std: if True, return the standard deviation
    :return: the normalized expected correspondence locations, [batch_size, n_pts, 2], optionally with std
    '''
    batch_size, n_dim, h2, w2 = featmap2.shape
    n_pts = coord2_n.shape[1]
    grid_n = gen_grid(h_min=-window_size, h_max=window_size,
                      w_min=-window_size, w_max=window_size,
                      len_h=int(window_size*h2), len_w=int(window_size*w2))

    grid_n_ = grid_n.repeat(batch_size, 1, 1, 1).to(coord2_n)  # Bx1xhwx2
    coord2_n_grid = coord2_n.unsqueeze(-2) + grid_n_  # Bxnxhwx2
    feat2_win = F.grid_sample(featmap2, coord2_n_grid, padding_mode='zeros', align_corners=False).permute(0, 2, 3, 1)  # Bxnxhwxd

    feat1 = feat1.unsqueeze(-2)

    prob, sim = compute_prob(feat1.reshape(batch_size*n_pts, -1, n_dim),
                        feat2_win.reshape(batch_size*n_pts, -1, n_dim), return_sim=True)#.reshape(batch_size, n_pts, -1)
    prob = prob.reshape(batch_size, n_pts, -1)

    expected_coord2_n = torch.sum(coord2_n_grid * prob.unsqueeze(-1), dim=2)  # Bxnx2

    re_list = [expected_coord2_n, coord2_n_grid]
    if with_std:
        var = torch.sum(coord2_n_grid**2 * prob.unsqueeze(-1), dim=2) - expected_coord2_n**2  # Bxnx2
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # Bxn
        # return expected_coord2_n, coord2_n_grid, std, prob
        re_list.append(std)
        re_list.append(prob)
    # else:
    #     return expected_coord2_n, coord2_n_grid
    if with_sim:
        re_list.append(sim.reshape(batch_size, n_pts, int(window_size*h2), int(window_size*w2)))
    return tuple(re_list)


@torch.no_grad()
def generate_kpts_disk(inputs, outputs, grid_size, keep_spatial=False):
    preds1 = outputs['preds1']
    preds2 = outputs['preds2']

    kp_map1, kp_map2 = preds1['local_point'], preds2['local_point']
    kps1, logp1, accept_mask1 = generate_kpts_disk_single(kp_map1, grid_size)
    kps2, logp2, accept_mask2 = generate_kpts_disk_single(kp_map2, grid_size)
    return kps1, kps2, logp1, logp2

def generate_kpts_disk_single(kp_map, grid_size):
    b,_,h,w = kp_map.shape 
    grids_org = gen_grid(h_min=-1, h_max=1, w_min=-1, w_max=1, len_h=h, len_w=w)
    grids_org = grids_org.reshape(h, w, 2)[None, :, :, :].repeat(b, 1, 1, 1).to(kp_map)
    grids_org = grids_org.permute(0,3,1,2)

    grids_unfold = unfold(grids_org, grid_size) # bx2x(h//g)x(w//g)x(g*g)
    kpmap_unfold = unfold(kp_map, grid_size)

    proposal_dist = Categorical(logits=kpmap_unfold)
    proposals     = proposal_dist.sample() # bx1x(h//g)x(w//g)
    proposal_logp = proposal_dist.log_prob(proposals)

    accept_logits = torch.gather(logits, dim=-1, index=proposals[..., None]).squeeze(-1) # bx1x(h//g)x(w//g)

    accept_dist    = Bernoulli(logits=accept_logits)
    accept_samples = accept_dist.sample() # bx1x(h//g)x(w//g)
    accept_logp    = accept_dist.log_prob(accept_samples) # for accepted points, equals to sigmoid() then log(); for denied, (1-sigmoid).log
    accept_mask    = accept_samples == 1.

    logp = proposal_logp + accept_logp
    kps = grids_unfold.gather(dim=4, index=proposals.unsqueeze(-1).repeat(1,2,1,1,1))
    return kps, logp, accept_mask

def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()

def cycle(iterable):
    while True:
        for x in iterable:
            yield x