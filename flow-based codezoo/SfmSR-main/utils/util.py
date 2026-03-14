import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import diffusers
from thop import profile
from model import DiTTransformer2DModel 

def count_parameters(model):
    all_parameters = sum(p.numel() for p in model.parameters())
    trainabel_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return all_parameters/ 1e6, trainabel_parameters/ 1e6


if __name__ == "__main__":
    modelweight = "weight/net"
    model = DiTTransformer2DModel.from_pretrained(modelweight).to("cuda")
    print(count_parameters(model))
    

    input_tensor = torch.randn(1, 3, 256, 256).to("cuda")
    timestep = torch.tensor(1, dtype=torch.long,device="cuda")

    flops, params = profile(model, inputs=(input_tensor,None,timestep))
    
    import time
    stime = time.time()
    model(input_tensor,None,timestep)
    print(f"time: {time.time()-stime}")  
    
    print(f"FLOPs: {flops / 1e9:.2f} G")  
    print(f"Params: {params / 1e6:.2f} M")  
    # model = MyModel()

    def get_model_memory(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size() 
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()  
        total_size = param_size + buffer_size  
        return total_size / (1024 ** 2) 

    memory_mb = get_model_memory(model)
    print(f"Model Memory: {memory_mb:.2f} MB")