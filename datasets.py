import os
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image


def load_volume_and_tag_list(volume_root, volume_split, length, train, gen_attn):

    tag_root = volume_root.replace('imageVolume', 'imageTag')

    if train:
        volume_list = sorted(os.listdir(volume_root))[volume_split[-1]:]  # training
        tag_list = sorted(os.listdir(tag_root))[volume_split[-1]:]  # training
    else:
        if gen_attn:
            volume_list = sorted(os.listdir(volume_root))[:volume_split[-1]]  # validation & testing
            tag_list = sorted(os.listdir(tag_root))[:volume_split[-1]]  # validation & testing
        else:
            volume_list = sorted(os.listdir(volume_root))[volume_split[0]:volume_split[-1]]  # validation
            tag_list = sorted(os.listdir(tag_root))[volume_split[0]:volume_split[-1]]  # validation
    
    sub_volume_list = []
    sub_tag_list = []
    sub_volume_idx_list = []
    for i in range(len(volume_list)):
        volume_name = volume_list[i]
        tag_name = tag_list[i]
        num_plane = int(tag_name.split('_')[-1].split('.')[0])

        if gen_attn:
            num_sub_volume = int(np.ceil(num_plane / length))
        else:
            num_sub_volume = num_plane // length
            
        for sub_volume_idx in range(num_sub_volume):
            sub_volume_list.append(volume_name)
            sub_tag_list.append(tag_name)
            sub_volume_idx_list.append(sub_volume_idx)

    return sub_volume_list, sub_tag_list, sub_volume_idx_list


class VolumeDataset(Dataset):
    def __init__(self, root, volume_split, length,
                 train=True, gen_attn=False,
                 transform=None):
        self.volume_name_list, self.tag_name_list, self.volume_idx_list = load_volume_and_tag_list(root, volume_split, length, train, gen_attn)
        self.volume_root = root
        self.tag_root = root.replace('imageVolume', 'imageTag')
        self.length = length
        self.transform = transform
        self.gen_attn = gen_attn
    
    def __getitem__(self, idx):
        volume_name, tag_name, volume_idx = self.volume_name_list[idx], self.tag_name_list[idx], self.volume_idx_list[idx]
        volume = np.load(os.path.join(self.volume_root, volume_name))
        tag = np.load(os.path.join(self.tag_root, tag_name))
        num_plane = tag.shape[0]

        if self.gen_attn:
            plane_idx = self.length * volume_idx if self.length * volume_idx + self.length <= num_plane else num_plane - self.length
        else:
            plane_idx = np.random.randint(num_plane - self.length + 1)
        
        sub_volume_name = '{}_{}_{}-{}'.format(volume_name.split('_')[0], volume_name.split('_')[1], plane_idx, plane_idx + self.length)
        sub_volume, sub_tag = volume[:, :, plane_idx:plane_idx + self.length], tag[plane_idx:plane_idx + self.length]
        
        img_list = []
        for i in range(self.length):
            plane = sub_volume[:, :, i]
            img = PIL.Image.fromarray(plane).convert("RGB")
            if self.transform:
                img = self.transform(img)
            img_list.append(img)

        sub_volume = torch.stack(img_list)
        label = torch.from_numpy(sub_tag)

        if self.gen_attn:
            f_sub_volume = torch.flip(sub_volume, [-1])
            return [sub_volume, f_sub_volume], label, sub_volume_name

        return sub_volume, label, sub_volume_name
    
    def __len__(self):
        return len(self.volume_name_list)


def build_dataset(is_train, args, gen_attn=False):
    transform = build_transform(is_train, args)
    dataset = None
    nb_classes = None

    if is_train:
        subset = 'train'
    else:
        if gen_attn:
            subset = 'generate attention map'
        else:
            subset = 'evaluate'

    if args.data_set == 'Volume':
        dataset = VolumeDataset(root=args.data_path, volume_split=args.volume_split, length=args.length,
                               train=is_train, gen_attn=gen_attn,
                               transform=transform)
        nb_classes = 1
    
    print('length of {} set:'.format(subset), len(dataset))

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im and not args.gen_attention_maps:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
