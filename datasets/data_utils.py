import numpy as np
import cv2
import torch

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def rotateImage(image, angle):
    h, w = image.shape[:2]
    angle_radius = np.abs(angle / 180. * np.pi)
    cos = np.cos(angle_radius)
    sin = np.sin(angle_radius)
    tan = np.tan(angle_radius)
    scale_h = (h / cos + (w - h * tan) * sin) / h
    scale_w = (h / sin + (w - h / tan) * cos) / w
    scale = max(scale_h, scale_w)
    image_center = tuple(np.array(image.shape[1::-1]) / 2.)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    rotation = np.eye(4)
    rotation[:2, :2] = rot_mat[:2, :2]
    return result, rotation


def perspective_transform(img, param=0.001):
    h, w = img.shape[:2]
    random_state = np.random.RandomState(None)
    M = np.array([[1 - param + 2 * param * random_state.rand(),
                   -param + 2 * param * random_state.rand(),
                   -param + 2 * param * random_state.rand()],
                  [-param + 2 * param * random_state.rand(),
                   1 - param + 2 * param * random_state.rand(),
                   -param + 2 * param * random_state.rand()],
                  [-param + 2 * param * random_state.rand(),
                   -param + 2 * param * random_state.rand(),
                   1 - param + 2 * param * random_state.rand()]])

    dst = cv2.warpPerspective(img, M, (w, h))
    return dst, M


def generate_query_kpts(img, mode, num_pts, h, w):
    # generate candidate query points
    if mode == 'random':
        kp1_x = np.random.rand(num_pts) * (w - 1)
        kp1_y = np.random.rand(num_pts) * (h - 1)
        coord = np.stack((kp1_x, kp1_y)).T

    elif mode == 'sift':
        gray1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_pts)
        kp1 = sift.detect(gray1)
        coord = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1])

    elif mode == 'mixed':
        kp1_x = np.random.rand(1 * int(0.1 * num_pts)) * (w - 1)
        kp1_y = np.random.rand(1 * int(0.1 * num_pts)) * (h - 1)
        kp1_rand = np.stack((kp1_x, kp1_y)).T

        sift = cv2.xfeatures2d.SIFT_create(nfeatures=int(0.9 * num_pts))
        gray1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp1_sift = sift.detect(gray1)
        kp1_sift = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1_sift])
        if len(kp1_sift) == 0:
            coord = kp1_rand
        else:
            coord = np.concatenate((kp1_rand, kp1_sift), 0)

    else:
        raise Exception('unknown type of keypoints')

    return coord


def prune_kpts(coord1, F_gt, im2_size, intrinsic1, intrinsic2, pose, d_min, d_max):
    # compute the epipolar lines corresponding to coord1
    coord1_h = np.concatenate([coord1, np.ones_like(coord1[:, [0]])], axis=1).T  # 3xn
    epipolar_line = F_gt.dot(coord1_h)  # 3xn
    epipolar_line /= np.clip(np.linalg.norm(epipolar_line[:2], axis=0), a_min=1e-10, a_max=None)  # 3xn

    # determine whether the epipolar lines intersect with the second image
    h2, w2 = im2_size
    corners = np.array([[0, 0, 1], [0, h2 - 1, 1], [w2 - 1, 0, 1], [w2 - 1, h2 - 1, 1]])  # 4x3
    dists = np.abs(corners.dot(epipolar_line))
    # if the epipolar line is far away from any image corners than sqrt(h^2+w^2)
    # it doesn't intersect with the image
    non_intersect = (dists > np.sqrt(w2 ** 2 + h2 ** 2)).any(axis=0)

    # determine if points in coord1 is likely to have correspondence in the other image by the rough depth range
    intrinsic1_4x4 = np.eye(4)
    intrinsic1_4x4[:3, :3] = intrinsic1
    intrinsic2_4x4 = np.eye(4)
    intrinsic2_4x4[:3, :3] = intrinsic2
    coord1_h_min = np.concatenate([d_min * coord1,
                                   d_min * np.ones_like(coord1[:, [0]]),
                                   np.ones_like(coord1[:, [0]])], axis=1).T
    coord1_h_max = np.concatenate([d_max * coord1,
                                   d_max * np.ones_like(coord1[:, [0]]),
                                   np.ones_like(coord1[:, [0]])], axis=1).T
    coord2_h_min = intrinsic2_4x4.dot(pose).dot(np.linalg.inv(intrinsic1_4x4)).dot(coord1_h_min)
    coord2_h_max = intrinsic2_4x4.dot(pose).dot(np.linalg.inv(intrinsic1_4x4)).dot(coord1_h_max)
    coord2_min = coord2_h_min[:2] / (coord1_h_min[2] + 1e-10)
    coord2_max = coord2_h_max[:2] / (coord1_h_max[2] + 1e-10)
    out_range = ((coord2_min[0] < 0) & (coord2_max[0] < 0)) | \
                ((coord2_min[1] < 0) & (coord2_max[1] < 0)) | \
                ((coord2_min[0] > w2 - 1) & (coord2_max[0] > w2 - 1)) | \
                ((coord2_min[1] > h2 - 1) & (coord2_max[1] > h2 - 1))

    ind_intersect = ~(non_intersect | out_range)
    return ind_intersect

def random_choice(array, size):
    rand = np.random.RandomState(1234)
    num_data = len(array)
    if num_data > size:
        idx = rand.choice(num_data, size, replace=False)
    else:
        idx = rand.choice(num_data, size, replace=True)
    return array[idx]

def tensor2array(tensor, max_value=None, colormap='coolwarm'):
    def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
        # Construct the list colormap, with interpolated values for higer resolution
        # For a linear segmented colormap, you can just specify the number of point in
        # cm.get_cmap(name, lutsize) with the parameter lutsize
        x = np.linspace(0, 1, low_res_cmap.N)
        low_res = low_res_cmap(x)
        new_x = np.linspace(0, max_value, resolution)
        high_res = np.stack([np.interp(new_x, x, low_res[:, i])
                             for i in range(low_res.shape[1])], axis=1)
        return ListedColormap(high_res)


    def opencv_rainbow(resolution=1000):
        # Construct the opencv equivalent of Rainbow
        opencv_rainbow_data = (
            (0.000, (1.00, 0.00, 0.00)),
            (0.400, (1.00, 1.00, 0.00)),
            (0.600, (0.00, 1.00, 0.00)),
            (0.800, (0.00, 0.00, 1.00)),
            (1.000, (0.60, 0.00, 1.00))
        )

        return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)
    COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000),
             'seismic':high_res_colormap(cm.get_cmap('seismic')),
             'coolwarm':high_res_colormap(cm.get_cmap('coolwarm'))}
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        if colormap in list(COLORMAPS.keys()):
            map_func = COLORMAPS[colormap]
        else:
            map_func = high_res_colormap(cm.get_cmap(colormap))
        array = map_func(norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        # assert(tensor.size(0) == 3)
        array = tensor.numpy()
    return array