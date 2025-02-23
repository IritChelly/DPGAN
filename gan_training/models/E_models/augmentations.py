from torchvision.datasets import STL10
import torchvision.transforms.functional as tvf
from torchvision import transforms
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
from torch import nn
from PIL import Image


class Augment(nn.Module):
    def __init__(self, target_size, nc, device, dataset_name):
        # target_size: original image size, e.g: (32,32)
        # This function defines the augmentation functions.

        super().__init__()
        self.device = device
        size = target_size[0]
        s = 1
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)  # for cifar, stl. Taken from Simclr.
        # color_jitter = transforms.ColorJitter(brightness=0.1 * s, contrast=0.5 * s, saturation=0.2 * s, hue=0.4 * s)
        # color_jitter = transforms.ColorJitter(brightness=0.8 * s, contrast=0.8 * s, saturation=0.8 * s, hue=0.2 * s)  
                                    # another option: (brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)

        random_resized_rotation_right = WrapWithRandomParams(lambda angle: ResizedRotation(angle, target_size), [(0.0, 80.0)])
        random_resized_rotation_left = WrapWithRandomParams(lambda angle: ResizedRotation(angle, target_size), [(280.0, 360.0)])
        random_resized_rotation = WrapWithRandomParams(lambda angle: ResizedRotation(angle, target_size), [(0.0, 360.0)])

        if dataset_name == 'mnist':
            self.randomize_transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.RandomApply([transforms.RandomResizedCrop(size=size, scale=(0.9, 1.0), ratio=(0.3, 2.0))], p=0.33),
                                    transforms.RandomApply([transforms.RandomChoice([random_resized_rotation_right, random_resized_rotation_left])]),
                                    transforms.RandomApply([color_jitter], p=0.8),
                                    transforms.RandomGrayscale(p=0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, ), (0.5, ))])
        else:  # Applicable for cifar and stl:
            self.randomize_transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.RandomResizedCrop(size=size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomApply([color_jitter], p=0.8),
                                        transforms.RandomGrayscale(p=0.2),
                                        GaussianBlur(kernel_size=int(0.1 * size)),
                                        transforms.ToTensor()])
        # else:
        #     self.randomize_transform = transforms.Compose([
        #                             transforms.ToPILImage(),
        #                             #transforms.RandomApply([transforms.RandomResizedCrop(size=size, scale=(0.95, 1.0), ratio=(0.3, 2.0))], p=0.33),
        #                             #transforms.RandomChoice([
        #                             #    transforms.RandomHorizontalFlip(p=0.5),  # create mirror image - not relevant for mnist
        #                             #    transforms.Lambda(random_rotate)
        #                             #]),
        #                             transforms.RandomHorizontalFlip(p=0.5),
        #                             #transforms.RandomApply([random_resized_rotation], p=0.33),
        #                             transforms.RandomApply([transforms.RandomChoice([random_resized_rotation_right, random_resized_rotation_left])]),
        #                             transforms.RandomApply([color_jitter], p=0.8),
        #                             #transforms.RandomGrayscale(p=0.2),
        #                             #GaussianBlur(nc, kernel_size=int(0.1 * size)),
        #                             transforms.ToTensor(),
        #                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #                             transforms.Lambda(lambda x: x + 1. / 128 * torch.rand(x.size()))])


    def forward(self, X):
        # X is a tensor in the shape: (batch_size, 3, img_sz, img_sz)

        # Apply first augmentation:
        X_aug_1_lst = [torch.unsqueeze(self.randomize_transform(x_.cpu() * 0.5 + 0.5), 0) for x_ in X]  # random.seed(idx) # for debug
        X_aug_1 = torch.cat(X_aug_1_lst, dim=0).to(self.device)  # shape: (batch_size, 3, img_sz, img_sz)

        # Apply second augmentation:
        X_aug_2_lst = [torch.unsqueeze(self.randomize_transform(x_.cpu() * 0.5 + 0.5), 0) for x_ in X]  # random.seed(idx + 1) # for debug
        X_aug_2 = torch.cat(X_aug_2_lst, dim=0).to(self.device)  # shape: (batch_size, 3, img_sz, img_sz)

        # # DEBUG:
        # idx = 0  # pick an image to view
        # orig_img = Image.fromarray((np.squeeze(np.moveaxis(X[idx].detach().cpu().numpy(), 0, -1)) * 255).astype(np.uint8))
        # img_aug1 = Image.fromarray((np.squeeze(np.moveaxis(X_aug_1[idx].detach().cpu().numpy(), 0, -1)) * 255).astype(np.uint8))
        # img_aug2 = Image.fromarray((np.squeeze(np.moveaxis(X_aug_2[idx].detach().cpu().numpy(), 0, -1)) * 255).astype(np.uint8))
        # orig_img.save("orig_img.jpeg")
        # img_aug1.save("img_aug1.jpeg")
        # img_aug2.save("img_aug2.jpeg")
        # 1/0
        # # END DEBUG
        
        # Shape of returned tensors: (batch_size, 3, img_sz, img_sz)
        return X_aug_1, X_aug_2



class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


# class GaussianBlur(object):
#     """blur a single image on CPU"""
#     def __init__(self, nc, kernel_size):
#         radias = kernel_size // 2
#         kernel_size = radias * 2 + 1
#         self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
#                                 stride=1, padding=0, bias=False, groups=3)
#         self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
#                                 stride=1, padding=0, bias=False, groups=3)
#         self.k = kernel_size
#         self.r = radias

#         self.blur = nn.Sequential(
#             nn.ReflectionPad2d(radias),
#             self.blur_h,
#             self.blur_v
#         )

#         self.pil_to_tensor = transforms.ToTensor()
#         self.tensor_to_pil = transforms.ToPILImage()

#     def __call__(self, img):
#         img = self.pil_to_tensor(img).unsqueeze(0)

#         sigma = np.random.uniform(0.1, 2.0)
#         x = np.arange(-self.r, self.r + 1)
#         x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
#         x = x / x.sum()
#         x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

#         self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
#         self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

#         with torch.no_grad():
#             img = self.blur(img)
#             img = img.squeeze()

#         img = self.tensor_to_pil(img)

#         return img


class WrapWithRandomParams():
    def __init__(self, constructor, ranges):
        self.constructor = constructor
        self.ranges = ranges
    
    def __call__(self, image):
        randoms = [float(np.random.uniform(low, high)) for _, (low, high) in zip(range(len(self.ranges)), self.ranges)]
        return self.constructor(*randoms)(image)


class ResizedRotation():
    def __init__(self, angle, output_size=(96, 96)):
        self.angle = angle
        self.output_size = output_size
        
    def angle_to_rad(self, ang): return np.pi * ang / 180.0
        
    def __call__(self, image):
        w, h = image.size
        new_h = int(np.abs(w * np.sin(self.angle_to_rad(90 - self.angle))) + np.abs(h * np.sin(self.angle_to_rad(self.angle))))
        new_w = int(np.abs(h * np.sin(self.angle_to_rad(90 - self.angle))) + np.abs(w * np.sin(self.angle_to_rad(self.angle))))
        img = tvf.resize(image, (new_w, new_h))
        img = tvf.rotate(img, self.angle)
        img = tvf.center_crop(img, self.output_size)
        return img


def random_rotate(image):
    if random.random() > 0.5:
        return tvf.rotate(image, angle=random.choice((0, 90, 180, 270)))
    return image









# ----------- OLD randomize_transforms function:
# random_resized_rotation = WrapWithRandomParams(lambda angle: ResizedRotation(angle, target_size), [(0.0, 360.0)])

# self.randomize_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomResizedCrop(target_size, scale=(1/3, 1.0), ratio=(0.3, 2.0)),
#     transforms.RandomChoice([
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.Lambda(random_rotate)
#     ]),
#     transforms.RandomApply([
#         random_resized_rotation
#     ], p=0.33),
#     transforms.RandomApply([
#         transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
#     ], p=0.8),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.ToTensor()
# ])
# --------- END


# # For debug:
# # data_dir = '/vildata/meitarr/Datasets/STL10/STL10'
# # stl10_unlabeled = STL10(data_dir, split="unlabeled")
# # idx = 123
# # orig_img = stl10_unlabeled[idx][0]  # this is a PIL image object of size (96,96,3)
# # random_resized_rotation = WrapWithRandomParams(lambda angle: ResizedRotation(angle), [(0.0, 360.0)])
# # img = random_resized_rotation(tvf.resize(orig_img, (96, 96)))
# # img.save("tmp.png")


# data_dir = '/vildata/meitarr/Datasets/STL10/STL10'
# stl10_unlabeled = STL10(data_dir, split="unlabeled")
# idx = 123
# orig_img = stl10_unlabeled[idx][0]  # this is a PIL image object of size (96,96,3)
# augment = Augment((96, 96), None)
# x_aug1, x_aug2 = augment(orig_img) 
# x_aug1_ = np.moveaxis(x_aug1.numpy(), 0, -1)
# x_aug2_ = np.moveaxis(x_aug2.numpy(), 0, -1)
# x_aug1_im = Image.fromarray((x_aug1_ * 255).astype(np.uint8))
# x_aug2_im = Image.fromarray((x_aug2_ * 255).astype(np.uint8))

# orig_img.save("orig_img.jpeg")
# x_aug1_im.save("x_aug1_im.jpeg")
# x_aug2_im.save("x_aug2_im.jpeg")

# print(x_aug1_im)
# print(x_aug2_im)

# 1/0

# data_dir = '/vildata/meitarr/Datasets/STL10/STL10'
# stl10_unlabeled = STL10(data_dir, split="unlabeled")
# idx = 123
# orig_img = stl10_unlabeled[idx][0]   # this is a PIL image object of size (96,96,3)
# orig_img_ = np.array(orig_img)
# img_sz = 96
# augment = Augment((img_sz, img_sz), None)
# x_aug1, x_aug2 = augment(orig_img_) 
# 1/0

# x_aug1_ = x_aug1.numpy()
# x_aug2_ = x_aug2.numpy()
# x_aug1_im = Image.fromarray(x_aug1_)
# x_aug2_im = Image.fromarray(x_aug2_)

# orig_img.save("orig_img.jpeg")
# x_aug1_im.save("x_aug1_im.jpeg")
# x_aug2_im.save("x_aug2_im.jpeg")