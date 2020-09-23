import pathlib
import socket
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms


def GET_HARMONIZATION_PATH():
    machine_name = socket.gethostname()
    if machine_name == 'Dolhasz':
        data_path = 'E:/image_harmonization'
    elif machine_name == 'DESKTOP-8K2EUHF':
        data_path = 'H:/image_harmonization'
    elif machine_name == 'dolhasz-laptop':
        data_path = 'D:/image_harmonization'
    elif machine_name == 'neuron':
        data_path = '/home/dolhasz/Image_Harmonization_Dataset'
    elif machine_name == 'neuromancer':
        data_path = '/home/dolhasz/Image_Harmonization_Dataset'
    elif machine_name == 'serrano':
        data_path = '/home/dolhasz/image_harmonization/'
    return data_path

datasets = ('HCOCO', 'HAdobe5k', 'HFlickr', 'Hday2night')


def load_all_image_paths(path, train=True):
    X, Y, M = [], [], []
    for d in datasets:
        x, y, m = load_image_names(path, d, train)
        X.extend(x)
        Y.extend(y)
        M.extend(m)
    return X, Y, M


def load_image_names(path, dataset, train=True):
    fpath = f'{dataset}_{"train" if train else "test"}.txt'
    full_path = path / dataset / fpath
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
        return dataset_path / dataset / 'real_images' / (name + '.' + ext)
    elif ptype == 'mask':
        name, mask, comp = image_name.split('_')
        _, ext = comp.split('.')
        return dataset_path / dataset / 'masks' / f'{name}_{mask}.{"png"}'
    elif ptype == 'comp':
        return dataset_path / dataset / 'composite_images' / image_name


class iHarmonyLoader(data.Dataset):
    def __init__(self, dataset, train, resize=(224,224)):
        self.basepath = pathlib.Path(GET_HARMONIZATION_PATH())
        if dataset == 'all':
            data = load_all_image_paths(self.basepath, train)
        else:
            data = load_image_names(self.basepath, dataset, train)
        self.X, self.Y, self.M = data
        self.X_transforms = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5)
        ]) 
        self.Y_transforms = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ]) 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = Image.open(self.X[idx])
        Y = Image.open(self.Y[idx])

        if self.X_transforms is not None:
            X = self.X_transforms(X)
        if self.Y_transforms is not None:
            Y = self.Y_transforms(Y)

        return X, Y



if __name__ == "__main__":
    train_gen = iHarmonyLoader('all', train=True)

    X, Y = train_gen[0]
    print(X)
    print(np.min(X.numpy()))

    transforms.ToPILImage()(X).show()
    transforms.ToPILImage()(Y).show()

    