import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
# from diffusers import DiTTransformer2DModel,DiTPipeline # UNet2DModel,DDPMPipeline, DDPMScheduler,
from model import DiTTransformer2DModel,FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from typing import Dict, List, Optional, Tuple, Union
import json
from data import Fmow,load_dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import clip
from torchvision.transforms import Resize,transforms

from torch.utils.tensorboard import SummaryWriter

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
        timelist = np.linspace(0,self.scheduler.initstep,num_inference_steps)[:-1]
        for t in self.progress_bar(timelist):
            timesteps = t
            timesteps = torch.tensor(timesteps, dtype=torch.long,device=latent_model_input.device)
            # predict noise model_output

            flow_pred = self.transformer(
                latent_model_input, timestep=timesteps,condition = condition
            ).sample
            # compute previous image: x_t -> x_t-1
            if aim_sample is not None:
                flow_pred = aim_sample - oringe_sample
                

            latent_model_input = self.scheduler.step(flow_pred, t, latent_model_input,num_inference_steps).prev_sample

        samples = latent_model_input
        samples = samples.clamp(0, 1)

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
    config = "./config/tinytest_64_256_26M.json"
    modelweight = "weight/net"
    device = "cuda"
    # model = DiTTransformer2DModel.from_pretrained(modelweight).to(torch.bfloat16)
    clipmodel, _ = clip.load("ViT-B/32", device=device)
    scheduler_initstep = 20
    clipmodel = clipmodel.to(torch.float32)
    model = DiTTransformer2DModel(
            sample_size=256, 
            in_channels=3, # conditional 6
            condition="clip",
            out_channels=3,
            patch_size = 2,
            num_layers=8,
            attention_head_dim=24,
        ).to(torch.bfloat16)
    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=scheduler_initstep)
    flowpipe = FlowPipeline(model,noise_scheduler).to('cuda')
    with open(config, 'r') as file:
        dataset_config_name = json.load(file)
        dataset = load_dataset(
            "fmow",
            dataset_config_name
        )
    N = 2
    # indices = torch.randperm(len(dataset)).tolist()[:N]
    indices = torch.randint(low=0, high=len(dataset), size=(N,)).tolist()
    sampler = SubsetRandomSampler(indices)
    dataloader = DataLoader(dataset, batch_size=N, sampler=sampler)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch {batch_idx+1}")
            lr_tensor = batch['LR'].to('cuda').to(torch.bfloat16)
            lr_clip = clipmodel.encode_image(alingclip(lr_tensor).to(torch.float32)).to(torch.bfloat16)
            lr_tensor = lr_tensor.to(torch.bfloat16)
            
            lr_imgs = batchtensor2img(lr_tensor)
            hr_tensor = batch['HR'].to('cuda').to(torch.bfloat16)
            hr_imgs = batchtensor2img(hr_tensor)
            # sr_images = flowpipe(lr_tensor,aim_sample=hr_tensor,output_type="pil").images
            # sr_images = flowpipe(lr_tensor,num_inference_steps=100,output_type="pil").images
            sr_images = flowpipe(lr_tensor,condition=lr_clip,num_inference_steps=20,output_type="pil").images #  conditional
        # for index,i in enumerate(sr_images):
        #     i.save(f"SRindex{index}.jpg")
        # for index,i in enumerate(hr_imgs):
        #     i.save(f"HRindex{index}.jpg")
        # for index,i in enumerate(lr_imgs):
        #     i.save(f"LRindex{index}.jpg")
        
    # tensorboard test 
    # writer = SummaryWriter('logs')
    # lr_tensor = (lr_tensor + 1) / 2 
    
    # hr_tensor = (hr_tensor + 1) / 2   
    # sr output="tensor"
    # writer.add_images('1 Images', lr_tensor, 0)  
    # writer.add_images('2 Images', hr_tensor, 0)  
    # writer.add_images('3 Images', sr_images, 0)  
    # writer.close()
    
        # model = DiTTransformer2DModel(
    #         sample_size=256, # 32
    #         in_channels=3,
    #         patch_size = 2,
    #         num_layers=8,
    #         attention_head_dim=24,
    #     ).to(torch.bfloat16)
    # noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=100)
            