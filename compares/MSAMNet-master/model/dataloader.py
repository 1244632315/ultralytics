from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import  torch
import os
import sep

class MultiframeDataset(Dataset):
    def __init__(self, dataset_dir, img_ids, input_size=512, num_frame=5, suffix='.png', test=False):
        super(MultiframeDataset, self).__init__()
        self._items = img_ids
        self.masks = dataset_dir+'/'+'masks'
        self.images = dataset_dir+'/'+'images'
        self.base_size = input_size
        self.num_frame = num_frame
        self.suffix = suffix
        self.test = test
              
    def __len__(self):
        return len(self._items)
    
    def __getitem__(self, index):
        base_size = self.base_size
        seq_id, img_id = self._items[index].split("/")
        imgs_path = self.images +'/'+ seq_id
        label_path = self.masks +'/'+ self._items[index] + self.suffix

        if self.test:
            mask = torch.zeros((1, base_size, base_size))
        else:
            mask = Image.open(label_path).convert('L')
            mask = mask.resize((base_size, base_size), Image.NEAREST)
            mask = np.array(mask, dtype=np.float32)
            mask = np.expand_dims(mask, axis=0) / 255.0
            mask = torch.from_numpy(mask)
        images = []
        
        for id in range(0, self.num_frame):
            img = Image.open(imgs_path +'/%d' % (max(int(img_id) - id, 1))+ self.suffix).convert('L')
            img  = img.resize ((base_size, base_size), Image.BILINEAR)
            img = np.array(img, dtype=np.float32)
            img = img / 255.0

            images.append(img)
 
        frames = torch.from_numpy(np.array(images[::-1])) # [T, H, W]
   
        return frames, mask

class MSODataset(Dataset):
    def __init__(self, dataset_dir, mode, img_ids, input_size, num_frame=5, suffix='.tif'):
        super(MSODataset, self).__init__()
        self._items = img_ids
        self.masks = dataset_dir+'/'+mode+'/masks'
        self.images = dataset_dir+'/'+mode+'/images'
        self.input_size = input_size
        self.num_frame = num_frame
        self.suffix = suffix

    @staticmethod
    def _normalize_frame(img):
        img = np.array(img, dtype=np.float32)
        bkg = sep.Background(img)
        img = img - bkg
        img[img < 0] = 0
        std = float(img.std())
        if std < 1e-6:
            std = 1.0
        img = (img - float(img.mean())) / std
        return img
        
    def __len__(self):
        return len(self._items)
    
    def __getitem__(self, index):
        seq_id, img_id = self._items[index].split("/")
        image_data = []
        imgs_path = self.images +'/'+ seq_id
        img_path = self.images +'/'+ self._items[index] +'.tif'
        img = Image.open(img_path)
        img = self._normalize_frame(img)
        image_data.append(img)
        

        label_path = self.masks +'/'+ self._items[index] +'.png'
        mask = Image.open(label_path)
        mask = np.array(mask, dtype=np.float32)
        mask = np.expand_dims(mask, axis=0) / 255.0

        for id in range(1, self.num_frame):
            img_his_path = imgs_path +'/%05d.tif' % (int(img_id) - id)
            if not os.path.exists(img_his_path):
                for i in range(1, self.num_frame):
                    img_his_path = imgs_path +'/%05d.tif' % (int(img_id) - id + i)
                    if os.path.exists(img_his_path):
                        break
            img_his = Image.open(img_his_path)
            img_his = self._normalize_frame(img_his)
            image_data.append(img_his)
            
        images = torch.from_numpy(np.array(image_data[::-1]))
        mask = torch.from_numpy(mask)

        return images, mask
