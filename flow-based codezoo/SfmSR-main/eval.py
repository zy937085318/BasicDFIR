import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import numpy as np

from PIL import Image
import torchvision.transforms as transforms
from model import DiTTransformer2DModel,FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from typing import Dict, List, Optional, Tuple, Union
import json
from data import Fmow,load_dataset
from torch.utils.data import DataLoader, SubsetRandomSampler,SequentialSampler
import utils.metrics as Metrics
import torch.nn.functional as F
import lpips
import clip
from torchvision.transforms import Resize,transforms
import matplotlib.pyplot as plt
import time

def normalize_tensor(tensor):
    return (tensor - 0.5) * 2  

def sample_to_tensor(sample):
    if isinstance(sample, torch.Tensor):
        return sample
    
    elif isinstance(sample, np.ndarray):
        tensor = torch.from_numpy(sample).float()  
        if tensor.ndimension() == 3 and tensor.shape[2] in [3, 4]: 
            tensor = tensor.permute(2, 0, 1) 
        return normalize_tensor(tensor)
    
    elif isinstance(sample, Image.Image):
        transform = transforms.ToTensor()
        tensor = transform(sample)
        return normalize_tensor(tensor)
    
    else:
        raise TypeError("Input sample must be a torch.Tensor, numpy.ndarray, or PIL.Image.")


class FlowPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation based on a Transformer backbone instead of a UNet.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        transformer ([`DiTTransformer2DModel`]):
            A class conditioned `DiTTransformer2DModel` to denoise the encoded image latents.
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        transformer: DiTTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler)

        # create a imagenet -> id dictionary for easier use
        self.labels = {}


    @torch.no_grad()
    def __call__(
        self,
        sample,
        condition = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        aim_sample = None
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 250):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels

        sample = sample_to_tensor(sample)
        oringe_sample = sample.clone()
        if aim_sample is not None:
            aim_sample = sample_to_tensor(aim_sample)
        latent_model_input =sample
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        timelist = np.linspace(0,self.scheduler.initstep-1,num_inference_steps)
        for t in self.progress_bar(timelist):
            timesteps = t
            timesteps = torch.tensor(timesteps, dtype=torch.long,device=latent_model_input.device)
            # predict noise model_output
            flow_pred = self.transformer(
                latent_model_input, timestep=timesteps,condition = condition
            ).sample
            # compute previous image: x_t -> x_t-1
            # if aim_sample is not None:
            #     # flow_preds = aim_sample - latent_model_input
            #     flow_preds = aim_sample - oringe_sample
            # loss1=F.mse_loss(flow_pred,flow_preds)
            # print(f"loss:{loss1}")
            # visualize_difference_histogram(flow_pred, flow_preds,t)

            latent_model_input = self.scheduler.step(flow_pred, t, latent_model_input,num_inference_steps).prev_sample
        samples = latent_model_input
        samples = (samples / 2 + 0.5).clamp(0, 1)

        if output_type == "tensor":
            return ImagePipelineOutput(images=samples)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
  
def visualize_difference_histogram(flow_pred, flow_preds,t):

    difference = (flow_pred - flow_preds).to(torch.float32)
    difference_np = difference.cpu().detach().numpy()

    plt.figure(figsize=(10, 6))
    plt.hist(difference_np.flatten(), bins=50, color='blue', alpha=0.7)
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.title('Difference Histogram')
    plt.savefig(f"./fig/lossnet/loss{t}.jpg")
    # plt.show()  
    
def batchtensor2img(tensor):
 
    if tensor.ndimension() != 4:
        raise ValueError("Input tensor must have 3 dimensions (B, C, H, W).")

    tensor = (tensor + 1) / 2

    tensor = tensor.clamp(0, 1)  
    tensor = tensor * 255
    tensor = tensor.to(torch.uint8)
    imgs = []
    img = tensor.permute(0, 2, 3, 1).cpu().numpy()
    for i in img:
        i = Image.fromarray(i)
        imgs.append(i)
    return imgs

# folder1 = "data/hr_256"
# folder2 = "data/sr_64_256"
# oringe_psnr_ssim(folder1,folder2)
def oringe_psnr_ssim(folder1,folder2):
    total_psnr = 0.0
    total_ssim = 0.0
    num_images = 0
    for filename in os.listdir(folder1):
        if filename.endswith(".tif"):
       
            hrimg_path = os.path.join(folder1, filename)
            lrimg_path = os.path.join(folder2, filename)
            
         
            hrimg = np.array(Image.open(hrimg_path))
            lrimg = np.array(Image.open(lrimg_path))
            
         
            psnr = Metrics.calculate_psnr(hrimg, lrimg)
            ssim = Metrics.calculate_ssim(hrimg, lrimg)
            print(f"PSNR: {psnr}")
            print(f"SSIM: {ssim}")
           
            total_psnr += psnr
            total_ssim += ssim
            num_images += 1
        
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    print(f"Average PSNR: {avg_psnr}")
    print(f"Average SSIM: {avg_ssim}")
    return avg_psnr,avg_ssim

def calculate_lpips(img1, img2, use_gpu=True):
    # model = PerceptualSimilarity.PerceptualLoss(model='net-lin', net='alex', use_gpu=use_gpu)
    # d = model.forward(img1, img2, normalize=True)
    # return d.detach().item()
    allloss = 0
    num = 0
    loss_fn_alex = lpips.LPIPS(net='alex').to('cuda')  # best forward scores
    transf = transforms.ToTensor()
    for i in range(len(img1)):
        test1 = transf(img1[i]).to(torch.float32).to('cuda')
        test_HR = transf(img2[i]).to(torch.float32).to('cuda')
        lpips_metrc = loss_fn_alex(test1, test_HR)
        num = num + 1
        allloss = allloss + lpips_metrc
    lpipsloss = allloss/num
    return float(lpipsloss)



def bilinearsamper(imgtensor,scale_factor = 1/4):
    downsampled_imgs = F.interpolate(imgtensor, scale_factor=scale_factor, mode='bilinear', align_corners=False)

    upsampled_imgs = F.interpolate(downsampled_imgs, size=imgtensor.shape[2:], mode='bilinear', align_corners=False)
    return upsampled_imgs

def alingclip(tensor1):
    tensor = (tensor1 + 1) / 2
    clip_normalize = transforms.Compose([
    Resize((224, 224)),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], 
        std=[0.26862954, 0.26130258, 0.27577711] )])
    tensor_normalized = torch.stack([clip_normalize(t) for t in tensor])
    return tensor_normalized

if __name__=="__main__":
    save_fig = True
    # with open("./config/potsdamtest_64_256_26M.json", 'r') as file:
    #     dataset_config = json.load(file)
    # dataset = load_dataset(
    #         "lrnrdatasets",
    #         dataset_config
    #     )
    # with open("./config/Torontotest_64_256_26M.json", 'r') as file:
    #     dataset_config = json.load(file)
    # dataset = load_dataset(
    #         "lrnrdatasets",
    #         dataset_config
    #     )
    # with open("./config/Torontotest_32_256_26M.json", 'r') as file:
    #     dataset_config = json.load(file)
    # dataset = load_dataset(
    #         "lrnrdatasets",
    #         dataset_config
    #     )
    with open("./config/potsdamtest_32_256_26M.json", 'r') as file:
        dataset_config = json.load(file)
    dataset = load_dataset(
            "lrnrdatasets",
            dataset_config
        )
    # with open("./config/tinytest_64_256_26M.json", 'r') as file:
    #     dataset_config = json.load(file)
    # dataset = load_dataset(
    #         "fmow",
    #         dataset_config
    #     )
    scheduler_initstep = 20
    dtype = torch.float16
    clipmodel, _ = clip.load("ViT-B/32", device="cuda")
    clipmodel = clipmodel.to(torch.float32)
    modelweight = "weight/net"
    model = DiTTransformer2DModel.from_pretrained(modelweight).to(dtype)
    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=scheduler_initstep)  # unditional 100
    flowpipe = FlowPipeline(model,noise_scheduler).to('cuda')
    N = 1
    dataloader = DataLoader(dataset, batch_size=N)
    total_psnr = 0.0
    total_ssim = 0.0
    total_psnrnet = 0.0
    total_ssimnet = 0.0
    num_images = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch {batch_idx+1}")
            hr_tensor = batch['HR'].to('cuda').to(dtype)
            hr_imgs = batchtensor2img(hr_tensor)
            lr_tensor = batch['LR'].to('cuda').to(dtype)
            lr_clip = clipmodel.encode_image(alingclip(lr_tensor).to(torch.float32)).to(dtype)
            lr_tensor = lr_tensor.to(dtype)
            # lr_tensor = bilinearsamper(hr_tensor)
            lr_imgs = batchtensor2img(lr_tensor)
            # sr_imgs = flowpipe(lr_tensor,condition = lr_tensor,aim_sample=hr_tensor,num_inference_steps=10,output_type="pil").images
            # condition = None 
            stime = time.time()
            sr_imgs = flowpipe(lr_tensor,condition=lr_clip,aim_sample=hr_tensor,num_inference_steps=5, output_type="pil").images # SRx4 3BEST
            print(f"time:{time.time()-stime}")
            for i in range(len(lr_imgs)):
                hr_img = np.array(hr_imgs[i])
                lr_img = np.array(lr_imgs[i])
                sr_img = np.array(sr_imgs[i])
                psnr = Metrics.calculate_psnr(lr_img,hr_img)
                psnrnet = Metrics.calculate_psnr(sr_img,hr_img)
                ssim = Metrics.calculate_ssim(lr_img,hr_img)
                ssimnet = Metrics.calculate_ssim(sr_img,hr_img)
                total_psnr += psnr
                total_ssim += ssim
                total_psnrnet += psnrnet
                total_ssimnet += ssimnet
                num_images += 1
                if save_fig:
                    hr_imgs[i].save(f"./fig/test/hr{i}.jpg")    
                    lr_imgs[i].save(f"./fig/test/lr{i}.jpg")    
                    sr_imgs[i].save(f"./fig/test/sr{i}.jpg")    
            break

    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    print(f"Average PSNR: {avg_psnr}")
    print(f"Average SSIM: {avg_ssim}")   
    avg_psnrnet = total_psnrnet / num_images
    avg_ssimnet = total_ssimnet / num_images
    print(f"Average PSNRNET: {avg_psnrnet}")
    print(f"Average SSIMNET: {avg_ssimnet}") 
    pls1 = calculate_lpips(lr_imgs,hr_imgs)
    plsnet = calculate_lpips(sr_imgs,hr_imgs)
    print(f"Average LPILS: {pls1}")
    print(f"Average LPILSNET: {plsnet}") 

    