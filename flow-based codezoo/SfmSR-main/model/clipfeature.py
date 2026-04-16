import torch
import clip
from torchvision.transforms import Resize,transforms


def alingclip(tensor1):
    tensor = (tensor1 + 1) / 2
    clip_normalize = transforms.Compose([
    Resize((224, 224)),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], 
        std=[0.26862954, 0.26130258, 0.27577711] )])
    tensor_normalized = torch.stack([clip_normalize(t) for t in tensor])
    return tensor_normalized

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.to(torch.float32)
lr_tensor = torch.randn(2, 3, 256, 256).to(device=device,dtype=torch.float32)  

lr_tensor_resized = alingclip(lr_tensor)

with torch.no_grad():
    clip_features = model.encode_image(lr_tensor_resized)  

print(clip_features.shape)  