import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import skimage.io as io
import torchvision.transforms as transforms
# import utils
import collections
from tqdm import tqdm
from path import Path
import datasets.data_utils as data_utils

rand = np.random.RandomState(234)

class MegaDepth_superpoint(Dataset):
    def __init__(self, configs, is_train=True):
        super(MegaDepth_superpoint, self).__init__()
        if is_train:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.ColorJitter
                                                 (brightness=1, contrast=1, saturation=1, hue=0.4),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                      std=(0.229, 0.224, 0.225)),
                                                 ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                      std=(0.229, 0.224, 0.225)),
                                                 ])

        self.is_train = is_train
        self.configs = configs
        self.root = Path(self.configs['data_path'])
        self.images = self.read_img_cam()
        self.imf1s, self.imf2s = self.read_pairs()
        print('total number of image pairs loaded: {}'.format(len(self.imf1s)))
        # shuffle data
        index = np.arange(len(self.imf1s))
        rand.shuffle(index)
        self.imf1s = list(np.array(self.imf1s)[index])
        self.imf2s = list(np.array(self.imf2s)[index])

    def read_img_cam(self):
        images = {}
        Image = collections.namedtuple(
            "Image", ["name", "w", "h", "fx", "fy", "cx", "cy", "rvec", "tvec"])
        for scene_id in os.listdir(self.root):
            densefs = [f for f in os.listdir(os.path.join(self.root, scene_id))
                       if 'dense' in f and os.path.isdir(os.path.join(self.root, scene_id, f))]
            for densef in densefs:
                folder = self.root/'{}/{}/aligned'.format(scene_id, densef) #os.path.join(self.root, scene_id, densef, 'aligned')
                img_cam_txt_path = folder/'img_cam.txt' #os.path.join(folder, 'img_cam.txt')
                with open(img_cam_txt_path, "r") as fid:
                    while True:
                        line = fid.readline()
                        if not line:
                            break
                        line = line.strip()
                        if len(line) > 0 and line[0] != "#":
                            elems = line.split()
                            image_name = elems[0]
                            img_path = folder/'images/'+image_name #os.path.join(folder, 'images', image_name)
                            w, h = int(elems[1]), int(elems[2])
                            fx, fy = float(elems[3]), float(elems[4])
                            cx, cy = float(elems[5]), float(elems[6])
                            R = np.array(elems[7:16])
                            T = np.array(elems[16:19])
                            if self.is_train:
                                label = self.data_dict[scene_id]
                                images[img_path] = Image(
                                    name=image_name, w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy, rvec=R, tvec=T
                                )
                            else:
                                images[img_path] = Image(
                                    name=image_name, w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy, rvec=R, tvec=T
                                )
        return images 

    def read_pairs(self):
        imf1s, imf2s = [], []
        print('reading image pairs from {}...'.format(self.root))
        for scene_id in tqdm(os.listdir(self.root), desc='# loading data from scene folders'):
            densefs = [f for f in os.listdir(os.path.join(self.root, scene_id))
                       if 'dense' in f and os.path.isdir(os.path.join(self.root, scene_id, f))]
            for densef in densefs:
                imf1s_ = []
                imf2s_ = []
                folder = self.root/'{}/{}/aligned'.format(scene_id, densef)
                pairf = folder/'pairs.txt' #os.path.join(folder, 'pairs.txt')

                if os.path.exists(pairf):
                    f = open(pairf, 'r')
                    for line in f:
                        imf1, imf2 = line.strip().split(' ')
                        imf1s_.append(folder/'images/'+imf1)
                        imf2s_.append(folder/'images/'+imf2)
                        # imf1s_.append(os.path.join(folder, 'images', imf1))
                        # imf2s_.append(os.path.join(folder, 'images', imf2))

                # make # image pairs per scene more balanced
                if len(imf1s_) > 5000:
                    index = np.arange(len(imf1s_))
                    rand.shuffle(index)
                    imf1s_ = list(np.array(imf1s_)[index[:5000]])
                    imf2s_ = list(np.array(imf2s_)[index[:5000]])

                imf1s.extend(imf1s_)
                imf2s.extend(imf2s_)

        return imf1s, imf2s

    @staticmethod
    def get_intrinsics(im_meta):
        return np.array([[im_meta.fx, 0, im_meta.cx],
                         [0, im_meta.fy, im_meta.cy],
                         [0, 0, 1]])

    @staticmethod
    def get_point_labels(file_path):
        label_root = file_path.dirname().dirname()
        name = file_path.name().replace('jpg','npz')
        label_file = np.load(label_root/name)['pts'] 
        label_file = label_file[:,:2]
        return label_file

    @staticmethod
    def get_extrinsics(im_meta):
        R = im_meta.rvec.reshape(3, 3)
        t = im_meta.tvec
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        return extrinsic

    def __getitem__(self, item):
        imf1 = self.imf1s[item]
        imf2 = self.imf2s[item]
        im1_meta = self.images[imf1]
        im2_meta = self.images[imf2]
        im1 = io.imread(imf1)
        im2 = io.imread(imf2)
        h, w = im1.shape[:2]

        intrinsic1 = self.get_intrinsics(im1_meta)
        intrinsic2 = self.get_intrinsics(im2_meta)

        extrinsic1 = self.get_extrinsics(im1_meta)
        extrinsic2 = self.get_extrinsics(im2_meta)

        relative = extrinsic2.dot(np.linalg.inv(extrinsic1))
        R = relative[:3, :3]
        # remove pairs that have a relative rotation angle larger than 80 degrees
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) * 180 / np.pi
        if theta > self.configs['rot_thr'] and self.is_train:
            item += 1
            if item >= self.__len__():
                item =0
            return self.__getitem__(item)

        T = relative[:3, 3]
        tx = data_utils.skew(T)
        E_gt = np.dot(tx, R)
        F_gt = np.linalg.inv(intrinsic2).T.dot(E_gt).dot(np.linalg.inv(intrinsic1))

        relative2 = extrinsic1.dot(np.linalg.inv(extrinsic2))
        R2 = relative2[:3, :3]
        # remove pairs that have a relative rotation angle larger than 80 degrees
        theta2 = np.arccos(np.clip((np.trace(R2) - 1) / 2, -1, 1)) * 180 / np.pi
        if theta2 > self.configs['rot_thr'] and self.is_train:
            item += 1
            if item >= self.__len__():
                item =0
            return self.__getitem__(item)

        T2 = relative2[:3, 3]
        tx2 = data_utils.skew(T2)
        E_gt2 = np.dot(tx2, R2)
        F_gt2 = np.linalg.inv(intrinsic1).T.dot(E_gt2).dot(np.linalg.inv(intrinsic2))

        # generate candidate query points
        # coord1 = data_utils.generate_query_kpts(im1, self.args.train_kp, 10*self.args.num_pts, h, w)
        coord1 = self.get_point_labels(imf1)
        coord2 = self.get_point_labels(imf2)

        # if no keypoints are detected
        if len(coord1) == 0 or len(coord2) == 0:
            item += 1
            if item >= self.__len__():
                item =0
            return self.__getitem__(item)

        # prune query keypoints that are not likely to have correspondence in the other image
        if self.configs['prune_kp']:
            ind_intersect = data_utils.prune_kpts(coord1[:,:2], F_gt, im2.shape[:2], intrinsic1, intrinsic2,
                                                  relative, d_min=4, d_max=400)
            if np.sum(ind_intersect) == 0:
                item += 1
                if item >= self.__len__():
                    item =0
                return self.__getitem__(item)
            coord1 = coord1[ind_intersect]

            ind_intersect2 = data_utils.prune_kpts(coord2[:,:2], F_gt2, im1.shape[:2], intrinsic2, intrinsic1,
                                                  relative2, d_min=4, d_max=400)
            if np.sum(ind_intersect2) == 0:
                item += 1
                if item >= self.__len__():
                    item =0
                return self.__getitem__(item)
            coord2 = coord2[ind_intersect2]

        if len(coord1) < self.configs['num_pts'] or len(coord2) < self.configs['num_pts']:
            item += 1
            if item >= self.__len__():
                item =0
            return self.__getitem__(item)

        coord1 = data_utils.random_choice(coord1, self.configs['num_pts'])
        coord1 = torch.from_numpy(coord1).float()
        coord2 = data_utils.random_choice(coord2, self.configs['num_pts'])
        coord2 = torch.from_numpy(coord2).float()

        im1_ori, im2_ori = torch.from_numpy(im1), torch.from_numpy(im2)

        F_gt = torch.from_numpy(F_gt).float() / (F_gt[-1, -1] + 1e-10)
        F_gt2 = torch.from_numpy(F_gt2).float() / (F_gt2[-1, -1] + 1e-10)
        intrinsic1 = torch.from_numpy(intrinsic1).float()
        intrinsic2 = torch.from_numpy(intrinsic2).float()
        pose = torch.from_numpy(relative[:3, :]).float()
        pose2 = torch.from_numpy(relative2[:3, :]).float()
        im1_tensor = self.transform(im1)
        im2_tensor = self.transform(im2)

        out = {'im1': im1_tensor,
               'im2': im2_tensor,
               'im1_ori': im1_ori,
               'im2_ori': im2_ori,
               'pose1': pose,
               'pose2': pose2,
               'F1': F_gt,
               'F2': F_gt2,
               'intrinsic1': intrinsic1,
               'intrinsic2': intrinsic2,
               'coord1': coord1,
               'coord2': coord2}

        return out

    def __len__(self):
        return len(self.imf1s)


class MegaDepth_SIFT(Dataset):
    def __init__(self, configs, is_train=True):
        super(MegaDepth_SIFT, self).__init__()
        if is_train:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.ColorJitter
                                                 (brightness=1, contrast=1, saturation=1, hue=0.4),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                      std=(0.229, 0.224, 0.225)),
                                                 ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                      std=(0.229, 0.224, 0.225)),
                                                 ])

        self.is_train = is_train
        self.configs = configs
        self.root = Path(self.configs['data_path'])
        self.images = self.read_img_cam()
        self.imf1s, self.imf2s = self.read_pairs()
        print('total number of image pairs loaded: {}'.format(len(self.imf1s)))
        # shuffle data
        index = np.arange(len(self.imf1s))
        rand.shuffle(index)
        self.imf1s = list(np.array(self.imf1s)[index])
        self.imf2s = list(np.array(self.imf2s)[index])

    def read_img_cam(self):
        images = {}
        Image = collections.namedtuple(
            "Image", ["name", "w", "h", "fx", "fy", "cx", "cy", "rvec", "tvec"])
        for scene_id in self.root.listdir():
            if not scene_id.isdir():
                continue
            densefs = [f.name for f in scene_id.listdir()
                       if 'dense' in f.name and f.isdir()]
            for densef in densefs:
                folder = scene_id/'{}/aligned'.format(densef) #os.path.join(self.root, scene_id, densef, 'aligned')
                img_cam_txt_path = folder/'img_cam.txt' #os.path.join(folder, 'img_cam.txt')
                with open(img_cam_txt_path, "r") as fid:
                    while True:
                        line = fid.readline()
                        if not line:
                            break
                        line = line.strip()
                        if len(line) > 0 and line[0] != "#":
                            elems = line.split()
                            image_name = elems[0]
                            img_path = folder/'images/'+image_name #os.path.join(folder, 'images', image_name)
                            w, h = int(elems[1]), int(elems[2])
                            fx, fy = float(elems[3]), float(elems[4])
                            cx, cy = float(elems[5]), float(elems[6])
                            R = np.array(elems[7:16])
                            T = np.array(elems[16:19])
                            images[img_path] = Image(
                                name=image_name, w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy, rvec=R, tvec=T
                            )
        return images 

    def read_pairs(self):
        imf1s, imf2s = [], []
        print('reading image pairs from {}...'.format(self.root))
        for scene_id in tqdm(self.root.listdir(), desc='# loading data from scene folders'):
            if not scene_id.isdir():
                continue
            densefs = [f.name for f in scene_id.listdir()
                       if 'dense' in f.name and f.isdir()]
            for densef in densefs:
                imf1s_ = []
                imf2s_ = []
                folder = scene_id/'{}/aligned'.format(densef)
                pairf = folder/'pairs.txt' #os.path.join(folder, 'pairs.txt')

                if os.path.exists(pairf):
                    f = open(pairf, 'r')
                    for line in f:
                        imf1, imf2 = line.strip().split(' ')
                        imf1s_.append(folder/'images/'+imf1)
                        imf2s_.append(folder/'images/'+imf2)
                        # imf1s_.append(os.path.join(folder, 'images', imf1))
                        # imf2s_.append(os.path.join(folder, 'images', imf2))

                # make # image pairs per scene more balanced
                if len(imf1s_) > 5000:
                    index = np.arange(len(imf1s_))
                    rand.shuffle(index)
                    imf1s_ = list(np.array(imf1s_)[index[:5000]])
                    imf2s_ = list(np.array(imf2s_)[index[:5000]])

                imf1s.extend(imf1s_)
                imf2s.extend(imf2s_)

        return imf1s, imf2s

    @staticmethod
    def get_intrinsics(im_meta):
        return np.array([[im_meta.fx, 0, im_meta.cx],
                         [0, im_meta.fy, im_meta.cy],
                         [0, 0, 1]])

    # @staticmethod
    def generate_query_kpts(self, img, num_pts, h, w, mode='mixed'):
    # generate candidate query points
        if mode == 'random':
            kp1_x = np.random.rand(num_pts) * (w - 1)
            kp1_y = np.random.rand(num_pts) * (h - 1)
            coord = np.stack((kp1_x, kp1_y, np.zeros(kp1_x.shape))).T

        elif mode == 'sift':
            gray1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_pts)
            sift = cv2.SIFT_create(nfeatures=num_pts)
            kp1 = sift.detect(gray1)
            coord = np.array([[kp.pt[0], kp.pt[1], 1] for kp in kp1])

        elif mode == 'mixed':
            kp1_x = np.random.rand(1 * int(self.configs['random_percent'] * num_pts)) * (w - 1)
            kp1_y = np.random.rand(1 * int(self.configs['random_percent'] * num_pts)) * (h - 1)
            kp1_rand = np.stack((kp1_x, kp1_y, np.zeros(kp1_x.shape))).T

            # sift = cv2.xfeatures2d.SIFT_create(nfeatures=int(0.5 * num_pts))
            sift = cv2.SIFT_create(nfeatures=int((1-self.configs['random_percent']) * num_pts))
            # sift = cv2.SIFT_create()
            gray1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            kp1_sift = sift.detect(gray1)
            kp1_sift = np.array([[kp.pt[0], kp.pt[1], 1] for kp in kp1_sift])
            if len(kp1_sift) == 0:
                coord = kp1_rand
            else:
                coord = np.concatenate((kp1_rand, kp1_sift), 0)

        else:
            raise Exception('unknown type of keypoints')

        return coord

    @staticmethod
    def get_extrinsics(im_meta):
        R = im_meta.rvec.reshape(3, 3)
        t = im_meta.tvec
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        return extrinsic

    def get_data_aug(self, item):
        if torch.rand(1) < 0.5:
            imf1 = self.imf1s[item]
        else:
            imf1 = self.imf2s[item]
        im1_meta = self.images[imf1]
        im1 = io.imread(imf1)

    def __getitem__(self, item):
        imf1 = self.imf1s[item]
        imf2 = self.imf2s[item]
        im1_meta = self.images[imf1]
        im2_meta = self.images[imf2]
        im1 = io.imread(imf1)
        im2 = io.imread(imf2)
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]

        intrinsic1 = self.get_intrinsics(im1_meta)
        intrinsic2 = self.get_intrinsics(im2_meta)

        extrinsic1 = self.get_extrinsics(im1_meta)
        extrinsic2 = self.get_extrinsics(im2_meta)

        relative = extrinsic2.dot(np.linalg.inv(extrinsic1))
        R = relative[:3, :3]
        # remove pairs that have a relative rotation angle larger than 80 degrees
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) * 180 / np.pi
        if theta > self.configs['rot_thr'] and self.is_train:
            return None

        T = relative[:3, 3]
        tx = data_utils.skew(T)
        E_gt = np.dot(tx, R)
        F_gt = np.linalg.inv(intrinsic2).T.dot(E_gt).dot(np.linalg.inv(intrinsic1))

        relative2 = extrinsic1.dot(np.linalg.inv(extrinsic2))
        R2 = relative2[:3, :3]
        # remove pairs that have a relative rotation angle larger than 80 degrees
        theta2 = np.arccos(np.clip((np.trace(R2) - 1) / 2, -1, 1)) * 180 / np.pi
        if theta2 > self.configs['rot_thr'] and self.is_train:
            return None

        T2 = relative2[:3, 3]
        tx2 = data_utils.skew(T2)
        E_gt2 = np.dot(tx2, R2)
        F_gt2 = np.linalg.inv(intrinsic1).T.dot(E_gt2).dot(np.linalg.inv(intrinsic2))

        # generate candidate query points
        # coord1 = data_utils.generate_query_kpts(im1, self.args.train_kp, 10*self.args.num_pts, h, w)
        coord1 = self.generate_query_kpts(im1, 10*self.configs['num_pts'], h1, w1)
        coord2 = self.generate_query_kpts(im2, 10*self.configs['num_pts'], h2, w2)

        # if no keypoints are detected
        if len(coord1) == 0 or len(coord2) == 0:
            return None

        # prune query keypoints that are not likely to have correspondence in the other image
        if self.configs['prune_kp']:
            ind_intersect = data_utils.prune_kpts(coord1[:,:2], F_gt, im2.shape[:2], intrinsic1, intrinsic2,
                                                  relative, d_min=4, d_max=400)
            if np.sum(ind_intersect) == 0:
                return None
            coord1 = coord1[ind_intersect]

            ind_intersect2 = data_utils.prune_kpts(coord2[:,:2], F_gt2, im1.shape[:2], intrinsic2, intrinsic1,
                                                  relative2, d_min=4, d_max=400)
            if np.sum(ind_intersect2) == 0:
                return None
            coord2 = coord2[ind_intersect2]

        if len(coord1) < self.configs['num_pts'] or len(coord2) < self.configs['num_pts']:
            return None

        coord1 = data_utils.random_choice(coord1, self.configs['num_pts'])
        coord1 = torch.from_numpy(coord1).float()
        coord2 = data_utils.random_choice(coord2, self.configs['num_pts'])
        coord2 = torch.from_numpy(coord2).float()

        im1_ori, im2_ori = torch.from_numpy(im1), torch.from_numpy(im2)

        F_gt = torch.from_numpy(F_gt).float() / (F_gt[-1, -1] + 1e-10)
        F_gt2 = torch.from_numpy(F_gt2).float() / (F_gt2[-1, -1] + 1e-10)
        intrinsic1 = torch.from_numpy(intrinsic1).float()
        intrinsic2 = torch.from_numpy(intrinsic2).float()
        pose = torch.from_numpy(relative[:3, :]).float()
        pose2 = torch.from_numpy(relative2[:3, :]).float()
        im1_tensor = self.transform(im1)
        im2_tensor = self.transform(im2)

        no_cuda = ('name1', 'name2')

        out = {'im1': im1_tensor,
               'im2': im2_tensor,
               'im1_ori': im1_ori,
               'im2_ori': im2_ori,
               'pose1': pose,
               'pose2': pose2,
               'F1': F_gt,
               'F2': F_gt2,
               'intrinsic1': intrinsic1,
               'intrinsic2': intrinsic2,
               'coord1': coord1,
               'coord2': coord2,
               'name1':im1_meta.name,
               'name2':im2_meta.name}

        return out

    def __len__(self):
        return len(self.imf1s)


class MegaDepth_Depth(Dataset):
    def __init__(self, configs, is_train=True):
        super(MegaDepth_Depth, self).__init__()
        if is_train:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.ColorJitter
                                                 (brightness=1, contrast=1, saturation=1, hue=0.4),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                      std=(0.229, 0.224, 0.225)),
                                                 ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                      std=(0.229, 0.224, 0.225)),
                                                 ])

        self.is_train = is_train
        self.configs = configs
        self.root = Path(self.configs['data_path'])
        self.images = self.read_img_cam()
        self.imf1s, self.imf2s = self.read_pairs()
        print('total number of image pairs loaded: {}'.format(len(self.imf1s)))
        # shuffle data
        index = np.arange(len(self.imf1s))
        rand.shuffle(index)
        self.imf1s = list(np.array(self.imf1s)[index])
        self.imf2s = list(np.array(self.imf2s)[index])

    def read_img_cam(self):
        images = {}
        Image = collections.namedtuple(
            "Image", ["name", "w", "h", "fx", "fy", "cx", "cy", "rvec", "tvec", "depth"])
        for scene_id in self.root.listdir():
            if not scene_id.isdir():
                continue
            densefs = [f.name for f in scene_id.listdir()
                       if 'dense' in f.name and f.isdir()]
            for densef in densefs:
                folder = scene_id/'{}/aligned'.format(densef) #os.path.join(self.root, scene_id, densef, 'aligned')
                img_cam_txt_path = folder/'img_cam.txt' #os.path.join(folder, 'img_cam.txt')
                with open(img_cam_txt_path, "r") as fid:
                    while True:
                        line = fid.readline()
                        if not line:
                            break
                        line = line.strip()
                        if len(line) > 0 and line[0] != "#":
                            elems = line.split()
                            image_name = elems[0]
                            img_path = folder/'images/'+image_name #os.path.join(folder, 'images', image_name)
                            depth = folder/'depths/'+image_name.replace('.jpg','.h5')
                            w, h = int(elems[1]), int(elems[2])
                            fx, fy = float(elems[3]), float(elems[4])
                            cx, cy = float(elems[5]), float(elems[6])
                            R = np.array(elems[7:16])
                            T = np.array(elems[16:19])
                            images[img_path] = Image(
                                name=image_name, w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy, rvec=R, tvec=T, depth=depth_path
                            )
        return images 

    def read_pairs(self):
        imf1s, imf2s = [], []
        print('reading image pairs from {}...'.format(self.root))
        for scene_id in tqdm(self.root.listdir(), desc='# loading data from scene folders'):
            if not scene_id.isdir():
                continue
            densefs = [f.name for f in scene_id.listdir()
                       if 'dense' in f.name and f.isdir()]
            for densef in densefs:
                imf1s_ = []
                imf2s_ = []
                folder = scene_id/'{}/aligned'.format(densef)
                pairf = folder/'pairs.txt' #os.path.join(folder, 'pairs.txt')

                if os.path.exists(pairf):
                    f = open(pairf, 'r')
                    for line in f:
                        imf1, imf2 = line.strip().split(' ')
                        imf1s_.append(folder/'images/'+imf1)
                        imf2s_.append(folder/'images/'+imf2)
                        # imf1s_.append(os.path.join(folder, 'images', imf1))
                        # imf2s_.append(os.path.join(folder, 'images', imf2))

                # make # image pairs per scene more balanced
                if len(imf1s_) > 5000:
                    index = np.arange(len(imf1s_))
                    rand.shuffle(index)
                    imf1s_ = list(np.array(imf1s_)[index[:5000]])
                    imf2s_ = list(np.array(imf2s_)[index[:5000]])

                imf1s.extend(imf1s_)
                imf2s.extend(imf2s_)

        return imf1s, imf2s

    @staticmethod
    def get_intrinsics(im_meta):
        return np.array([[im_meta.fx, 0, im_meta.cx],
                         [0, im_meta.fy, im_meta.cy],
                         [0, 0, 1]])

    # @staticmethod
    def generate_query_kpts(self, img, num_pts, h, w, mode='mixed'):
        """
        Although we define this function, the key points here are not used. Actually, the keypoints used during training
        are generated in the preprocess step.
        我们参照caps的代码定义了这个函数，以方便进行ablation，但我们的方法并不会用到这里的关键点，训练中的关键点是在
        preprocess步骤中生成的
        """
        # generate candidate query points
        if mode == 'random':
            kp1_x = np.random.rand(num_pts) * (w - 1)
            kp1_y = np.random.rand(num_pts) * (h - 1)
            coord = np.stack((kp1_x, kp1_y, np.zeros(kp1_x.shape))).T

        elif mode == 'sift':
            gray1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_pts)
            sift = cv2.SIFT_create(nfeatures=num_pts)
            kp1 = sift.detect(gray1)
            coord = np.array([[kp.pt[0], kp.pt[1], 1] for kp in kp1])

        elif mode == 'mixed':
            kp1_x = np.random.rand(1 * int(self.configs['random_percent'] * num_pts)) * (w - 1)
            kp1_y = np.random.rand(1 * int(self.configs['random_percent'] * num_pts)) * (h - 1)
            kp1_rand = np.stack((kp1_x, kp1_y, np.zeros(kp1_x.shape))).T

            # sift = cv2.xfeatures2d.SIFT_create(nfeatures=int(0.5 * num_pts))
            sift = cv2.SIFT_create(nfeatures=int((1-self.configs['random_percent']) * num_pts))
            # sift = cv2.SIFT_create()
            gray1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            kp1_sift = sift.detect(gray1)
            kp1_sift = np.array([[kp.pt[0], kp.pt[1], 1] for kp in kp1_sift])
            if len(kp1_sift) == 0:
                coord = kp1_rand
            else:
                coord = np.concatenate((kp1_rand, kp1_sift), 0)

        else:
            raise Exception('unknown type of keypoints')

        return coord

    @staticmethod
    def get_extrinsics(im_meta):
        R = im_meta.rvec.reshape(3, 3)
        t = im_meta.tvec
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        return extrinsic

    def get_data_aug(self, item):
        if torch.rand(1) < 0.5:
            imf1 = self.imf1s[item]
        else:
            imf1 = self.imf2s[item]
        im1_meta = self.images[imf1]
        im1 = io.imread(imf1)

    def __getitem__(self, item):
        imf1 = self.imf1s[item]
        imf2 = self.imf2s[item]
        im1_meta = self.images[imf1]
        im2_meta = self.images[imf2]
        im1 = io.imread(imf1)
        im2 = io.imread(imf2)
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]

        intrinsic1 = self.get_intrinsics(im1_meta)
        intrinsic2 = self.get_intrinsics(im2_meta)

        extrinsic1 = self.get_extrinsics(im1_meta)
        extrinsic2 = self.get_extrinsics(im2_meta)

        relative = extrinsic2.dot(np.linalg.inv(extrinsic1))
        R = relative[:3, :3]
        # remove pairs that have a relative rotation angle larger than 80 degrees
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) * 180 / np.pi
        if theta > self.configs['rot_thr'] and self.is_train:
            item += 1
            if item >= self.__len__():
                item =0
            return self.__getitem__(item)

        T = relative[:3, 3]
        tx = data_utils.skew(T)
        E_gt = np.dot(tx, R)
        F_gt = np.linalg.inv(intrinsic2).T.dot(E_gt).dot(np.linalg.inv(intrinsic1))

        relative2 = extrinsic1.dot(np.linalg.inv(extrinsic2))
        R2 = relative2[:3, :3]
        # remove pairs that have a relative rotation angle larger than 80 degrees
        theta2 = np.arccos(np.clip((np.trace(R2) - 1) / 2, -1, 1)) * 180 / np.pi
        if theta2 > self.configs['rot_thr'] and self.is_train:
            return None

        T2 = relative2[:3, 3]
        tx2 = data_utils.skew(T2)
        E_gt2 = np.dot(tx2, R2)
        F_gt2 = np.linalg.inv(intrinsic1).T.dot(E_gt2).dot(np.linalg.inv(intrinsic2))

        # generate candidate query points
        # coord1 = data_utils.generate_query_kpts(im1, self.args.train_kp, 10*self.args.num_pts, h, w)
        coord1 = self.generate_query_kpts(im1, 10*self.configs['num_pts'], h1, w1)
        coord2 = self.generate_query_kpts(im2, 10*self.configs['num_pts'], h2, w2)

        # if no keypoints are detected
        if len(coord1) == 0 or len(coord2) == 0:
            item += 1
            if item >= self.__len__():
                item =0
            return self.__getitem__(item)

        # prune query keypoints that are not likely to have correspondence in the other image
        if self.configs['prune_kp']:
            ind_intersect = data_utils.prune_kpts(coord1[:,:2], F_gt, im2.shape[:2], intrinsic1, intrinsic2,
                                                  relative, d_min=4, d_max=400)
            if np.sum(ind_intersect) == 0:
                item += 1
                if item >= self.__len__():
                    item =0
                return self.__getitem__(item)
            coord1 = coord1[ind_intersect]

            ind_intersect2 = data_utils.prune_kpts(coord2[:,:2], F_gt2, im1.shape[:2], intrinsic2, intrinsic1,
                                                  relative2, d_min=4, d_max=400)
            if np.sum(ind_intersect2) == 0:
                item += 1
                if item >= self.__len__():
                    item =0
                return self.__getitem__(item)
            coord2 = coord2[ind_intersect2]

        if len(coord1) < self.configs['num_pts'] or len(coord2) < self.configs['num_pts']:
            item += 1
            if item >= self.__len__():
                item =0
            return self.__getitem__(item)

        coord1 = data_utils.random_choice(coord1, self.configs['num_pts'])
        coord1 = torch.from_numpy(coord1).float()
        coord2 = data_utils.random_choice(coord2, self.configs['num_pts'])
        coord2 = torch.from_numpy(coord2).float()

        im1_ori, im2_ori = torch.from_numpy(im1), torch.from_numpy(im2)

        F_gt = torch.from_numpy(F_gt).float() / (F_gt[-1, -1] + 1e-10)
        F_gt2 = torch.from_numpy(F_gt2).float() / (F_gt2[-1, -1] + 1e-10)
        intrinsic1 = torch.from_numpy(intrinsic1).float()
        intrinsic2 = torch.from_numpy(intrinsic2).float()
        pose = torch.from_numpy(relative[:3, :]).float()
        pose2 = torch.from_numpy(relative2[:3, :]).float()
        im1_tensor = self.transform(im1)
        im2_tensor = self.transform(im2)

        no_cuda = ('name1', 'name2')

        depth = h5py.File(im1_meta.depth, 'r')['depth'][:]
        depth = cv2.resize(depth, (640,480))
        depth = torch.from_numpy(depth).float()

        out = {'im1': im1_tensor,
               'im2': im2_tensor,
               'im1_ori': im1_ori,
               'im2_ori': im2_ori,
               'pose1': pose,
               'pose2': pose2,
               'F1': F_gt,
               'F2': F_gt2,
               'intrinsic1': intrinsic1,
               'intrinsic2': intrinsic2,
               'coord1': coord1,
               'coord2': coord2,
               'name1':im1_meta.name,
               'name2':im2_meta.name,
               'depth': depth}

        return out

    def __len__(self):
        return len(self.imf1s)