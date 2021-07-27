# import os
# import torch
# from PIL import Image
# from torch.utils.data import Dataset
# from torchvision.transforms import transforms
# import config as cfg
# import numpy as np
#
# class CUB_Dataset(Dataset):
#     """Face Landmarks dataset."""
#     def __init__(self, input_size, train=True, crop=False, cam_img=False):
#         self.dir = "../CI-CAM-final/dataset/CUB_200_2011"
#         self.train = train
#         self.image_list = []
#         self.label_list = []
#         self.bbox_list = []
#         self.img_size = []
#         mean_vals = [0.485, 0.456, 0.406]
#         std_vals = [0.229, 0.224, 0.225]
#
#         _imagenet_pca = {}
#         _imagenet_pca["eigval"] = torch.from_numpy(np.asarray([0.2175, 0.0188, 0.0045]))
#         _imagenet_pca["eigvec"] = torch.from_numpy(np.asarray([
#             [-0.5675, 0.7192, 0.4009],
#             [-0.5808, -0.0045, -0.8140],
#             [-0.5836, -0.6948, 0.4203]
#         ]))
#
#         if cam_img:
#             self.list_path = os.path.join(self.dir, "datalist", "train_list.txt")
#             self.func_transforms = transforms.Compose([transforms.Resize((cfg.crop_size, cfg.crop_size)),
#                                                        transforms.ToTensor(),
#                                                        transforms.Normalize(mean_vals, std_vals)
#                                                        ])
#         else:
#             if self.train:
#                 self.list_path = os.path.join(self.dir, "datalist", "train_list.txt")
#                 self.func_transforms = transforms.Compose(
#                         [transforms.Resize(input_size),
#                          transforms.RandomCrop(cfg.crop_size),
#                          transforms.RandomHorizontalFlip(),
#                          transforms.ToTensor(),
#                          transforms.Normalize(mean_vals, std_vals)
#                          ])
#             else:
#                 self.list_path = os.path.join(self.dir, "datalist", "test_list.txt")
#                 if crop:
#                     self.func_transforms = transforms.Compose([transforms.Resize(input_size),
#                                                                transforms.CenterCrop(cfg.crop_size),
#                                                                transforms.ToTensor(),
#                                                                transforms.Normalize(mean_vals, std_vals)
#                                                                ])
#                 else:
#                     self.func_transforms = transforms.Compose([transforms.Resize((cfg.crop_size, cfg.crop_size)),
#                                                                transforms.ToTensor(),
#                                                                transforms.Normalize(mean_vals, std_vals)
#                                                                ])
#                 self.read_bbox_list()
#
#         self.read_labeled_image_list()
#
#     def __len__(self):
#         return len(self.image_list)
#
#     def __getitem__(self, idx):
#         img_name = self.image_list[idx]
#         image = Image.open(img_name).convert('RGB')
#         image = self.func_transforms(image).float()
#         if self.train:
#             return image, torch.tensor(self.label_list[idx])
#         else:
#             return image, torch.tensor(self.label_list[idx]), torch.tensor(self.img_size[idx])
#
#     def read_bbox_list(self):
#         with open(os.path.join(self.dir, "datalist", "test_bounding_box.txt"), 'r') as f:
#             lines = f.readlines()
#             for i in range(len(lines)):
#                 line = lines[i].split(' ')
#                 self.bbox_list.append([[float(line[0]), float(line[1]), float(line[2]), float(line[3])]])  # [[xmin, ymin, xmax, ymax]]
#                 self.img_size.append([float(line[4]), float(line[5])])  # [height, width]
#
#     def read_labeled_image_list(self):
#         with open(self.list_path, 'r') as f:
#             for line in f:
#                 image, label = line.strip("\n").split(';')
#                 self.image_list.append(os.path.join(self.dir, "images", image.strip()))
#                 self.label_list.append(int(label.strip()))



import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import config as cfg
import numpy as np

class CUB_Dataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, input_size, train=True, crop=False, danet=False, cam_img=False):
        self.dir = "../CI-CAM-final/dataset/CUB_200_2011"
        self.train = train
        self.image_list = []
        self.label_list = []
        self.bbox_list = []
        self.img_size = []
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

        _imagenet_pca = {}
        _imagenet_pca["eigval"] = torch.from_numpy(np.asarray([0.2175, 0.0188, 0.0045]))
        _imagenet_pca["eigvec"] = torch.from_numpy(np.asarray([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203]
        ]))

        if cam_img:
            self.list_path = os.path.join(self.dir, "datalist", "train_list.txt")
            self.func_transforms = transforms.Compose([transforms.Resize((cfg.crop_size, cfg.crop_size)),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean_vals, std_vals)
                                                       ])
        else:
            if self.train:
                self.list_path = os.path.join(self.dir, "datalist", "train_list.txt")
                if danet:
                    self.func_transforms = transforms.Compose(
                        [transforms.Resize(input_size),  # resize 和 crop 是不一样的。resize不是正方形，crop是正方形
                         transforms.RandomCrop(cfg.crop_size),
                         transforms.RandomHorizontalFlip(),
                         transforms.ColorJitter(0.4, 0.4, 0.4),
                         transforms.ToTensor(),
                         Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
                         transforms.Normalize(mean_vals, std_vals)
                         ])
                else:
                    self.func_transforms = transforms.Compose(
                        [transforms.Resize(input_size),  # resize 和 crop 是不一样的。resize不是正方形，crop是正方形
                         transforms.RandomCrop(cfg.crop_size),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean_vals, std_vals)
                         ])
            else:
                self.list_path = os.path.join(self.dir, "datalist", "test_list.txt")
                if crop:
                    self.func_transforms = transforms.Compose([transforms.Resize(input_size),
                                                               transforms.CenterCrop(cfg.crop_size),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize(mean_vals, std_vals)
                                                               ])
                else:
                    self.func_transforms = transforms.Compose([transforms.Resize((cfg.crop_size, cfg.crop_size)),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize(mean_vals, std_vals)
                                                               ])
                self.read_bbox_list()

        self.read_labeled_image_list()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        # image = Image.open(img_name)  # 因为有些图片不是三通道的，所以需要转为RGB三通道
        image = self.func_transforms(image).float()
        if self.train:
            return image, torch.tensor(self.label_list[idx])
        else:
            return image, torch.tensor(self.label_list[idx]), torch.tensor(self.img_size[idx])

    def read_bbox_list(self):
        with open(os.path.join(self.dir, "datalist", "test_bounding_box.txt"), 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i].split(' ')
                self.bbox_list.append([[float(line[0]), float(line[1]), float(line[2]), float(line[3])]])  # [[xmin, ymin, xmax, ymax]]
                self.img_size.append([float(line[4]), float(line[5])])  # [height, width]

    def read_labeled_image_list(self):
        with open(self.list_path, 'r') as f:
            for line in f:
                image, label = line.strip("\n").split(';')
                self.image_list.append(os.path.join(self.dir, "images", image.strip()))
                self.label_list.append(int(label.strip()))

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
