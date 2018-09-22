from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image, ImageFile
from torchvision import transforms
from utils import *


class Airbus(Dataset):
    def __init__(self, filename, root_dir, transform=None, train=False):
        df = pd.read_csv(os.path.join(root_dir, filename))
        self.train = train
        self.fk_frame = df
        self.imgnames = df.ImageId.unique()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.imgnames)

    def __getitem__(self, idx):
        if self.train:
            img_name = os.path.join(self.root_dir, 'train/', self.imgnames[idx])
        else:
            img_name = os.path.join(self.root_dir, 'test/', self.imgnames[idx])
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(img_name)
        image = image.convert('RGB')
        jpg_to_tensor = transforms.ToTensor()
        tensor_to_pil = transforms.ToPILImage()
        image = tensor_to_pil(jpg_to_tensor(image))

        if self.train:
            rle = self.fk_frame.loc[self.fk_frame.ImageId == self.imgnames[idx], 'EncodedPixels']
            masks = masks_as_image(rle)
            if masks.sum() > 0 :
                label = 1
            else:
                label = 0
        else:
            masks = np.zeros(shape=image.shape, dtype=np.uint16)
            label = 0

        if image is not None:
            sample = {'image': image, 'masks': masks, 'labels': label}

        if self.transform:
            image = self.transform(image)
            masks = tensor_to_pil(jpg_to_tensor(np.dstack([masks]*3)))
            masks = self.transform(masks)*255.0
            sample = {'image': image, 'masks': masks, 'labels': label}

        return sample