import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from scipy.io import loadmat
from tqdm import tqdm
from path import Path

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

methods = ['hesaff', 'hesaffnet', 'contextdesc', 'd2-net', 'r2d2', 'aslfeat', 'disk-d-8k-official', 'delf-new', 'superpoint', 'caps', 'disk-epipolar', 'PoSFeat_CVPR']
names = ['Hes. Aff. + Root-SIFT', 'HAN + HN++', 'SIFT + ContextDesc', 'D2-Net', 'R2D2', 'ASLFeat', 'DISK', 'DELF', 'SuperPoint', 'SIFT + CAPS', 'DISK-W', 'PoSFeat']
colors = ['tan', 'orange', 'peru', 'skyblue', 'purple', 'tomato', 'yellowgreen', 'gray', 'darkcyan', 'slateblue', 'yellowgreen', 'red']
linestyles = ['--','--','--','--','--','--','--','-', '-', '-', '-', '-', '-']

top_k = None
n_i = 52
n_v = 56
dataset_path = Path('/data/kunb/hpatches/hpatches-sequences-release')
features_path = Path('../../ckpts/hpatches/PoSFeat_mytrain/desc')

lim = [1, 15]
rng = np.arange(lim[0], lim[1] + 1)

@torch.no_grad()
def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    # print(sim)
    # print(sim.max(), sim.min())
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()

def benchmark_features(read_feats):
    seq_names = sorted(dataset_path.listdir())

    n_feats = []
    n_matches = []
    seq_type = []
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}

    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        seq_name = seq_name.name
        keypoints_a, descriptors_a = read_feats(seq_name, 1)
        if keypoints_a.shape[0] > 60000:
            keypoints_a = keypoints_a[:60000,:]
            descriptors_a = descriptors_a[:60000, :]
        n_feats.append(keypoints_a.shape[0])

        for im_idx in range(2, 7):
            keypoints_b, descriptors_b = read_feats(seq_name, im_idx)
            if keypoints_b.shape[0] > 60000:
                keypoints_b = keypoints_b[:60000,:]
                descriptors_b = descriptors_b[:60000, :]
            n_feats.append(keypoints_b.shape[0])

            matches = mnn_matcher(
                torch.from_numpy(descriptors_a).to(device=device), 
                torch.from_numpy(descriptors_b).to(device=device)
            )
            
            homography = np.loadtxt(dataset_path/"{}/H_1_{}".format(seq_name, im_idx))
            
            pos_a = keypoints_a[matches[:, 0], : 2] 
            pos_a_h = np.concatenate([pos_a, np.ones([matches.shape[0], 1])], axis=1)
            pos_b_proj_h = np.transpose(np.dot(homography, np.transpose(pos_a_h)))
            pos_b_proj = pos_b_proj_h[:, : 2] / pos_b_proj_h[:, 2 :]

            pos_b = keypoints_b[matches[:, 1], : 2]

            dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))

            n_matches.append(matches.shape[0])
            seq_type.append(seq_name[0])
            
            if dist.shape[0] == 0:
                dist = np.array([float("inf")])
            
            for thr in rng:
                if seq_name[0] == 'i':
                    i_err[thr] += np.mean(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)
    
    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)
    
    return i_err, v_err, [seq_type, n_feats, n_matches]

def summary(stats):
    seq_type, n_feats, n_matches = stats
    print('# Features: {:f} - [{:d}, {:d}]'.format(np.mean(n_feats), np.min(n_feats), np.max(n_feats)))
    print('# Matches: Overall {:f}, Illumination {:f}, Viewpoint {:f}'.format(
        np.sum(n_matches) / ((n_i + n_v) * 5), 
        np.sum(n_matches[seq_type == 'i']) / (n_i * 5), 
        np.sum(n_matches[seq_type == 'v']) / (n_v * 5))
    )
def generate_read_function(method, extension='ppm'):
    def read_function(seq_name, im_idx):
        aux = np.load(features_path/"{}/{}.{}.{}".format(seq_name, im_idx, extension, method))
        if top_k is None:
            return aux['keypoints'], aux['descriptors']
        else:
            assert('scores' in aux)
            ids = np.argsort(aux['scores'])[-top_k :]
            return aux['keypoints'][ids, :], aux['descriptors'][ids, :]
    return read_function

def sift_to_rootsift(descriptors):
    return np.sqrt(descriptors / np.expand_dims(np.sum(np.abs(descriptors), axis=1), axis=1) + 1e-16)
def parse_mat(mat):
    keypoints = mat['keypoints'][:, : 2]
    raw_descriptors = mat['descriptors']
    l2_norm_descriptors = raw_descriptors / np.expand_dims(np.sum(raw_descriptors ** 2, axis=1), axis=1)
    descriptors = sift_to_rootsift(l2_norm_descriptors)
    if top_k is None:
        return keypoints, descriptors
    else:
        assert('scores' in mat)
        ids = np.argsort(mat['scores'][0])[-top_k :]
        return keypoints[ids, :], descriptors[ids, :]

if top_k is None:
    cache_dir = 'cache'
else:
    cache_dir = 'cache-top'
if not os.path.isdir(cache_dir):
    os.mkdir(cache_dir)

errors = {}

for method in methods:
    output_file = os.path.join(cache_dir, method + '.npy')
    print(method)
    if method == 'hesaff':
        read_function = lambda seq_name, im_idx: parse_mat(loadmat(os.path.join(dataset_path, seq_name, '%d.ppm.hesaff' % im_idx), appendmat=False))
    else:
        if method == 'delf' or method == 'delf-new':
            read_function = generate_read_function(method, extension='png')
        else:
            read_function = generate_read_function(method)
    if os.path.exists(output_file):
        print('Loading precomputed errors...')
        errors[method] = np.load(output_file, allow_pickle=True)
    else:
        errors[method] = benchmark_features(read_function)
        saved = np.array(errors[method], dtype=object)
        np.save(output_file, saved)
    summary(errors[method][-1])

# evalute MMA score
MMAscore = {}
for method in methods:
    i_err, v_err, _ = errors[method]
    tmp_a = []
    tmp_i = []
    tmp_v = []
    for thr in range(1,11):
        tmp_a.append((i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5))
        tmp_i.append(i_err[thr] / (n_i * 5))
        tmp_v.append(v_err[thr] / (n_v * 5))
    cur_a = 0 
    cur_i = 0 
    cur_v = 0
    upper_bound = 0
    for idx, (mma_a, mma_i, mma_v) in enumerate(zip(tmp_a, tmp_i, tmp_v)):
        cur_a += (2-(idx+1)/10.)*mma_a
        cur_i += (2-(idx+1)/10.)*mma_i
        cur_v += (2-(idx+1)/10.)*mma_v
        upper_bound += (2-(idx+1)/10.)*1
    MMAscore[method] = (cur_a/upper_bound, cur_i/upper_bound, cur_v/upper_bound)

# plot
plt_lim = [1, 10]
plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)
plt_ylim = [0, 1]

plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=25)

labelsize = 20
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for method, name, color, ls in zip(methods, names, colors, linestyles):
    i_err, v_err, _ = errors[method]
    plt.plot(plt_rng, [(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
plt.title('Overall')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylabel('MMA')
plt.ylim(plt_ylim)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=labelsize)
# plt.legend()

plt.subplot(1, 3, 2)
for method, name, color, ls in zip(methods, names, colors, linestyles):
    i_err, v_err, _ = errors[method]
    plt.plot(plt_rng, [i_err[thr] / (n_i * 5) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
plt.title('Illumination')
plt.xlabel('threshold [px]')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylim(plt_ylim)
plt.gca().axes.set_yticklabels([])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=labelsize)

plt.subplot(1, 3, 3)
for method, name, color, ls in zip(methods, names, colors, linestyles):
    i_err, v_err, _ = errors[method]
    plt.plot(plt_rng, [v_err[thr] / (n_v * 5) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
plt.title('Viewpoint')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylim(plt_ylim)
plt.gca().axes.set_yticklabels([])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=labelsize)

import datetime
timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")

if top_k is None:
    plt.savefig('hseq{}.pdf'.format(timestamp), bbox_inches='tight', dpi=300)
    plt.savefig('hseq{}.eps'.format(timestamp), bbox_inches='tight', dpi=300)
else:
    plt.savefig('hseq-top.pdf', bbox_inches='tight', dpi=300)

plt.legend()
if top_k is None:
    plt.savefig('hseq{}_label.pdf'.format(timestamp), bbox_inches='tight', dpi=300)
else:
    plt.savefig('hseq-top_label.pdf', bbox_inches='tight', dpi=300)

with open('hseq{}.txt'.format(timestamp), 'w') as f:
    lines = ''
    for name, method in zip(names, methods):
        name = name.ljust(25, ' ')
        tmp_stat = errors[method][-1]
        seq_type, n_feats, n_matches = tmp_stat
        num_feat = np.mean(n_feats)
        num_match = np.sum(n_matches) / ((n_i + n_v) * 5)
        mmascore = MMAscore[method]
        lines += '{} & {:.1f} & {:.1f} & {:.3f} & {:.3f} & {:.3f}\n'.format(
            name, num_feat, num_match, mmascore[0], mmascore[1], mmascore[2])

    f.write(lines)