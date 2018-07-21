import torch as t
from torch.utils import data
import os
from PIL import Image
import numpy as np
from torchvision import transforms as T
from torch.utils.data import DataLoader




transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean = [.5, .5, .5], std = [.5, .5, .5])
])


class Pulse(data.Dataset):
    """docstring for  """

    def __init__(self, root, transforms = transform):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        if 'txt_0' in img_path.split('/')[-1]:
            label = 0
        elif 'txt_1' in img_path.split('/')[-1]:
            label = 1
        elif 'txt_2' in img_path.split('/')[-1]:
            label = 2
        elif 'txt_3' in img_path.split('/')[-1]:
            label = 3
        '''
        pil_img = Image.open(img_path)
        array = np.asarray(pil_img)
        data = t.from_numpy(array)
        return data, label
        '''
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)

'''
dataset = Pulse('/Volumes/MIngHardisk/spec5/kaiser', transforms = transform)
img, label = dataset[0]
for img, label in dataset:
    print(img.size(), img.float().mean(), label)

'''

'''
class NewPulse(Pulse):
    def __init__(self, index):
        try:
            return super(NewPulse, self).__getitem__(index)
        except:
            new_index = random.randint(0, len(self) - 1)
            return self[new_index] #随机替代损坏图片

from torch.utils.data.dataloader import default_collate


def my_collate_fn(batch):
    batch = list(filter(lambda x:x[0] is not None, batch))
    return default_collate(batch)

dataset = NewPulse('/Volumes/MIngHardisk/spec5/test', transforms = transform)



dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=False)

'''