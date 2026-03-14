import numpy
from PIL import Image
from torch.utils.data import Dataset
import random
from pathlib import Path
from torchvision import transforms
import torch
import json

class Fmow(Dataset):
    def __init__(self, dataroot, l_res=64, h_res=256):
        super().__init__()
        self.l_res = l_res
        self.h_res = h_res
        self.dataroot = dataroot # list
        self.randomcrop = transforms.RandomCrop(h_res)
        self.sresize = transforms.Compose([transforms.Resize(l_res, interpolation=transforms.InterpolationMode.BICUBIC),
                                           transforms.Resize(h_res, interpolation=transforms.InterpolationMode.BICUBIC),])
        self.hflip = transforms.RandomHorizontalFlip()
        self.totensor_normalize = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5]),])
        self.orige_img = []
        for root in dataroot:
            for file in Path(root).rglob('*'):
                if file.suffix == '.jpg' and file.name.endswith('rgb.jpg') and not file.name.endswith('msrgb.jpg'):
                    self.orige_img.append(file)
            self.datalen = len(self.orige_img)
    
    def __len__(self):
        return self.datalen
    
    def transormer_augment(self,img,min_max=(-1, 1)):
        ret_img = self.totensor_normalize(img)
        # ret_img = self.normalize(img)
        # ret_img = img * (min_max[1] - min_max[0]) + min_max[0] 
        return ret_img
    
    def __getitem__(self, index):
        img = Image.open(self.orige_img[index]).convert("RGB")
        img = self.hflip(img)
        width, height = img.size
        if min(width, height) < 256:
            img = img.resize((256, 256))
        img_HR = self.randomcrop(img)
        img_LR = self.sresize(img_HR) # 256-64-256
        img_HR = self.transormer_augment(img_HR)
        img_LR = self.transormer_augment(img_LR)
        
        return {'HR': img_HR, 'LR': img_LR}
    
def tensor2img(tensor):
 
    if tensor.ndimension() != 3:
        raise ValueError("Input tensor must have 3 dimensions (C, H, W).")

    tensor = (tensor + 1) / 2

    tensor = tensor.clamp(0, 1)  
    tensor = tensor * 255
    tensor = tensor.to(torch.uint8)

    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(img)  
    return img
    
def load_dataset(name,config):
    data_list = config["datalist"]
    fmowdatasets = Fmow(data_list)
    return fmowdatasets
    
if __name__ == '__main__':
    with open("../config/pretrain_64_256_26M.json", 'r') as file:
        dataset_config_name = json.load(file)
    fmowdatasets = load_dataset(
            "fmow",
            dataset_config_name['datasets']
        )
    print(len(fmowdatasets))
    result = fmowdatasets[2]
    LR = tensor2img(result['LR']).save("Lr.jpg")
    HR = tensor2img(result['HR']).save("Hr.jpg")