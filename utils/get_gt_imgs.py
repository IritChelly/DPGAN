
import os
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms, datasets
import numpy as np
import random
import torch.utils.data as data


class StackedMNIST(data.Dataset):
    def __init__(self, data_dir, transform, batch_size=100000):
        super().__init__()
        self.channel1 = datasets.MNIST(data_dir,
                                       transform=transform,
                                       train=True,
                                       download=True)
        self.channel2 = datasets.MNIST(data_dir,
                                       transform=transform,
                                       train=True,
                                       download=True)
        self.channel3 = datasets.MNIST(data_dir,
                                       transform=transform,
                                       train=True,
                                       download=True)
        self.indices = {
            k: (random.randint(0,
                               len(self.channel1) - 1),
                random.randint(0,
                               len(self.channel1) - 1),
                random.randint(0,
                               len(self.channel1) - 1))
            for k in range(batch_size)
        }

    def __getitem__(self, index):
        index1, index2, index3 = self.indices[index]
        x1, y1 = self.channel1[index1]
        x2, y2 = self.channel2[index2]
        x3, y3 = self.channel3[index3]
        return torch.cat([x1, x2, x3], dim=0), y1 * 100 + y2 * 10 + y3

    def __len__(self):
        return len(self.indices)


def get_images(root, N):
    if False and os.path.exists(root + '.txt'):
        with open(os.path.exists(root + '.txt')) as f:
            files = f.readlines()
            random.shuffle(files)
            return files
    else:
        all_files = []
        for i, (dp, dn, fn) in enumerate(os.walk(os.path.expanduser(root))):
            for j, f in enumerate(fn):
                if j >= 1000:
                    break     # don't get whole dataset, just get enough images per class
                if f.endswith(('.png', '.webp', 'jpg', '.JPEG')):
                    all_files.append(os.path.join(dp, f))
        random.shuffle(all_files)
        return all_files


def pt_to_np(imgs):
    '''normalizes pytorch image in [-1, 1] to [0, 255]'''
    return (imgs.permute(0, 2, 3, 1).mul_(0.5).add_(0.5).mul_(255)).clamp_(0, 255).numpy()


def get_transform(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_gt_samples(dataset, nimgs=50000):

    if dataset == 'cifar':
        data = datasets.CIFAR10(paths[dataset], transform=get_transform(sizes[dataset]))
        images = []
        for x, y in tqdm(data):
            images.append(x)

        return pt_to_np(torch.stack(images))

    elif dataset == 'stacked_mnist':
        data = StackedMNIST(paths[dataset], transform=transforms.Compose([
                                   transforms.Resize(32),
                                   transforms.CenterCrop(32),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, ), (0.5, ))
                               ]))
        images = []
        j = 0
        for x, y in tqdm(data):
            images.append(x)
            j = j + 1
            if j >= nimgs:
                break

        return pt_to_np(torch.stack(images))
    
    elif dataset == 'mnist':
        data = datasets.MNIST(paths[dataset], transform=transforms.Compose([
                                   transforms.Resize(32),
                                   transforms.CenterCrop(32),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, ), (0.5, ))
                               ]))
        images = []
        j = 0
        for x, y in tqdm(data):
            images.append(x)
            j = j + 1
            if j >= nimgs:
                break

        return pt_to_np(torch.stack(images))

    else:
        transform = get_transform(sizes[dataset])
        all_images = get_images(paths[dataset],nimgs)
        images = []
        for file_path in tqdm(all_images[:nimgs]):
            images.append(transform(Image.open(file_path).convert('RGB')))

        return pt_to_np(torch.stack(images))


paths = {
    'imagenet': 'data/ImageNet',
    'places': 'data/Places365',
    'cifar': 'data/CIFAR',
    'stacked_mnist': 'data/MNIST',
    'mnist': 'data/MNIST'
}

sizes = {'imagenet': 128, 'places': 128, 'cifar': 32, 'stacked_mnist' : 32, 'mnist' : 32}

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Save a batch of ground truth train set images for evaluation')
    parser.add_argument('--cifar', action='store_true')
    parser.add_argument('--imagenet', action='store_true')
    parser.add_argument('--places', action='store_true')
    parser.add_argument('--stacked_mnist',action='store_true')
    parser.add_argument('--mnist',action='store_true')
    args = parser.parse_args()

    os.makedirs('output', exist_ok=True)

    if args.cifar:
        cifar_samples = get_gt_samples('cifar', nimgs=50000)
        np.savez('output/cifar_gt_imgs.npz', fake=cifar_samples, real=cifar_samples)
    if args.imagenet:
        imagenet_samples = get_gt_samples('imagenet', nimgs=50000)
        np.savez('output/imagenet_gt_imgs.npz', fake=imagenet_samples, real=imagenet_samples)
    if args.places:
        places_samples = get_gt_samples('places', nimgs=50000)
        np.savez('output/places_gt_imgs.npz', fake=places_samples, real=places_samples)
    if args.stacked_mnist:
        stacked_mnist_samples = get_gt_samples('stacked_mnist', nimgs=50000)
        np.savez('output/stacked_mnist_gt_imgs.npz', fake=stacked_mnist_samples, real=stacked_mnist_samples)
    if args.mnist:
        mnist_samples = get_gt_samples('mnist', nimgs=50000)
        np.savez('output/mnist_gt_imgs.npz', fake=mnist_samples, real=mnist_samples)
    



