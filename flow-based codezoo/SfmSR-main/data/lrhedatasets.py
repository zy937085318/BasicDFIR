import numpy
import os
from PIL import Image
from torch.utils.data import Dataset
import random
from pathlib import Path
from torchvision import transforms
import torch
import json

class Lrhrdataset(Dataset):
    def __init__(self, hrdataroot,srdataroot,type, l_res=64, h_res=256):
        super().__init__()
        self.l_res = l_res
        self.h_res = h_res
        self.type = type
        self.hrdataroot = hrdataroot # srt
        self.srdataroot = srdataroot # srt
        self.hflip = transforms.RandomHorizontalFlip()
        self.totensor_normalize = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5]),])
        self.imgpairs = []
        
        if not isinstance(hrdataroot, list):
            hrdataroot = [hrdataroot]  
        if not isinstance(srdataroot, list):
            srdataroot = [srdataroot]      
        for i in range(len(hrdataroot)):
            hr_files = os.listdir(hrdataroot[i])
            for hr_filename in hr_files:
            
                lr_filename = hr_filename
                
                hr_path = os.path.join(hrdataroot[i], hr_filename)
                lr_path = os.path.join(srdataroot[i], lr_filename)
            
                if os.path.exists(lr_path):
            
                    self.imgpairs.append((hr_path, lr_path))
                    
            self.datalen = len(self.imgpairs)
    
    def __len__(self):
        return self.datalen
    
    def transormer_augment(self,img,min_max=(-1, 1)):
        ret_img = self.totensor_normalize(img)
        # ret_img = self.normalize(img)
        # ret_img = img * (min_max[1] - min_max[0]) + min_max[0] 
        return ret_img
    
    def __getitem__(self, index):
        img_HR,img_LR = self.imgpairs[index]
        img_HR,img_LR = Image.open(img_HR).convert("RGB"), Image.open(img_LR).convert("RGB"),
        # if self.type == "train":
        #     img_LR = self.hflip(img_LR)
        #     img_HR = self.hflip(img_HR)
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
    hrdata_list = config["datasets"]["datalist"]
    lrdata_list = config["datasets"]["datalistlr"]
    dataset_type = config["train_type"]
    datasets = Lrhrdataset(hrdata_list,lrdata_list,dataset_type)
    return datasets
    
if __name__ == '__main__':
    with open("../config/potsdamtrain_64_256_26M.json", 'r') as file:
        dataset_config = json.load(file)
    datasets = load_dataset(
            "lrnrdatasets",
            dataset_config
        )
    print(len(datasets))
    result = datasets[2]
    LR = tensor2img(result['LR']).save("Lr.jpg")
    HR = tensor2img(result['HR']).save("Hr.jpg")