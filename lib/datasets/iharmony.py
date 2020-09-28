import os
import argparse
import torch
from skimage import io, transform
import numpy as np 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
import warnings

def load_image_names(path, dataset, train=True):
    full_path = os.path.join(
        path,
        dataset,
        f'{dataset}_{"train" if train else "test"}.txt'
    )
    with open(full_path, 'r') as image_list:
        X, Y, M = [], [], []
        for image_name in image_list.readlines():
            X.append(make_path(image_name.strip('\n'), dataset, path, 'comp'))
            Y.append(make_path(image_name.strip('\n'), dataset, path, 'real'))
            M.append(make_path(image_name.strip('\n'), dataset, path, 'mask'))
        return X, Y, M


def make_path(image_name, dataset, dataset_path, ptype='real'):
    if ptype == 'real':
        name, _, comp = image_name.split('_')
        _, ext = comp.split('.')
        return os.path.join(dataset_path, dataset, 'real_images', name + '.' + ext)
    elif ptype == 'mask':
        name, mask, comp = image_name.split('_')
        _, ext = comp.split('.')
        return os.path.join(dataset_path, dataset, 'masks', f'{name}_{mask}.{"png"}')
    elif ptype == 'comp':
        return os.path.join(dataset_path, dataset, 'composite_images', image_name)


def parse_args():
    parser = argparse.ArgumentParser('iHarmony Dataloader')
    parser.add_argument('--dataset_path')
    return parser.parse_args()



warnings.filterwarnings("ignore")
# plt.ion()

# args = parse_args()


def open_file(path):
    with open(path, 'r') as f:
        return [make_path(line, 'HCOCO') for line in f.readlines()]


class HarmonisationDataset(Dataset):
    def __init__(self, data_path, dataset, train=True):
        self.data_path = data_path
        self.comps, self.real, self.masks = load_image_names(data_path, dataset, train=True)

        process = [
            transforms.ToPILImage(),
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor()
        ]
        self.transform_train = transforms.Compose(process)

    def __len__(self):
        return len(self.comps)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        seed = np.random.randint(2147483647)
        comp_path = self.comps[idx]
        real_path = self.real[idx]
        comp = io.imread(comp_path)
        real = io.imread(real_path)
        # sample = {'comp': comp, 'real': real}
        # sample = [comp, real]

        random.seed(seed)
        torch.manual_seed(seed)
        comp = self.transform_train(comp)
        random.seed(seed)
        torch.manual_seed(seed)
        real = self.transform_train(real)
        # if self.transform:
        #     sample = self.transform(sample)

        return {'comp':comp, 'real':real}

if __name__ == "__main__":
    h = HarmonisationDataset('E:/image_harmonization', 'HCOCO', train=True)

    dl = DataLoader(h, batch_size=8, shuffle=True, num_workers=8)
    for i in iter(dl):
        print(i['real'].shape)