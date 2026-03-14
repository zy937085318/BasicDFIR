max_batch=25
batch_size_ip=4

### CelebA
dataset=celeba
eval_split=test

method=restora_flow
model_type=ot
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} steps_ode 64 correction_steps 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} steps_ode 64 correction_steps 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} steps_ode 128 correction_steps 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} steps_ode 128 correction_steps 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


method=pnp_flow
model_type=ot
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} steps_pnp 100 alpha 0.8 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} steps_pnp 100 alpha 0.5 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} steps_pnp 100 alpha 0.3 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} steps_pnp 100 alpha 0.01 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


method=ot_ode
model_type=ot
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} steps_ode 100 start_time 0.3 gamma gamma_t max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} steps_ode 100 start_time 0.1 gamma gamma_t max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} steps_ode 100 start_time 0.1 gamma constant max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} steps_ode 100 start_time 0.1 gamma constant max_batch ${max_batch} batch_size_ip ${batch_size_ip}


method=flow_priors
model_type=ot
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} start_time 0.0 lmbda 100 eta 0.01  max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} start_time 0.0 lmbda 10000 eta 0.01  max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} start_time 0.0 lmbda 10000 eta 0.1  max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} start_time 0.0 lmbda 10000 eta 0.01  max_batch ${max_batch} batch_size_ip ${batch_size_ip}


method=ddnm
model_type=ddpm
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} sampling_steps 100 jump_length 1 jump_n_sample 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} sampling_steps 100 jump_length 1 jump_n_sample 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} sampling_steps 100 jump_length 1 jump_n_sample 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} sampling_steps 100 jump_length 1 jump_n_sample 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


method=repaint
model_type=ddpm
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} timesteps 250 jump_length 10 jump_n_sample 10 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} timesteps 250 jump_length 10 jump_n_sample 10 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} timesteps 250 jump_length 10 jump_n_sample 10 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


method=d_flow
model_type=ot
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} lmbda 0.001 alpha 0.1 LBFGS_iter 3 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} lmbda 0.001 alpha 0.1 LBFGS_iter 9 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} lmbda 0.001 alpha 0.1 LBFGS_iter 10 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} lmbda 0.01 alpha 0.1 LBFGS_iter 20 max_batch ${max_batch} batch_size_ip ${batch_size_ip}

########################################################################################################################
########################################################################################################################

### COCO
dataset=coco
eval_split=val

method=restora_flow
model_type=ot
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} steps_ode 64 correction_steps 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} steps_ode 64 correction_steps 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} steps_ode 128 correction_steps 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} steps_ode 128 correction_steps 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


method=pnp_flow
model_type=ot
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} steps_pnp 100 alpha 0.8 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} steps_pnp 100 alpha 0.5 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} steps_pnp 100 alpha 0.3 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} steps_pnp 100 alpha 0.01 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


method=ot_ode
model_type=ot
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} steps_ode 100 start_time 0.3 gamma gamma_t max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} steps_ode 100 start_time 0.1 gamma gamma_t max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} steps_ode 100 start_time 0.1 gamma constant max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} steps_ode 100 start_time 0.1 gamma constant max_batch ${max_batch} batch_size_ip ${batch_size_ip}


method=flow_priors
model_type=ot
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} start_time 0.0 lmbda 100 eta 0.01  max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} start_time 0.0 lmbda 10000 eta 0.01  max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} start_time 0.0 lmbda 10000 eta 0.1  max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} start_time 0.0 lmbda 10000 eta 0.01  max_batch ${max_batch} batch_size_ip ${batch_size_ip}


method=ddnm
model_type=ddpm
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} sampling_steps 100 jump_length 1 jump_n_sample 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} sampling_steps 100 jump_length 1 jump_n_sample 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} sampling_steps 100 jump_length 1 jump_n_sample 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} sampling_steps 100 jump_length 1 jump_n_sample 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


method=repaint
model_type=ddpm
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} timesteps 250 jump_length 10 jump_n_sample 10 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} timesteps 250 jump_length 10 jump_n_sample 10 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} timesteps 250 jump_length 10 jump_n_sample 10 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


method=d_flow
model_type=ot
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} lmbda 0.001 alpha 0.1 LBFGS_iter 3 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} lmbda 0.001 alpha 0.1 LBFGS_iter 9 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} lmbda 0.001 alpha 0.1 LBFGS_iter 10 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} lmbda 0.01 alpha 0.1 LBFGS_iter 20 max_batch ${max_batch} batch_size_ip ${batch_size_ip}

########################################################################################################################
########################################################################################################################

## AFHQ-Cat
dataset=afhq_cat
eval_split=test

method=restora_flow
model_type=ot
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} steps_ode 64 correction_steps 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} steps_ode 64 correction_steps 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} steps_ode 256 correction_steps 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} steps_ode 128 correction_steps 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip}

method=pnp_flow
model_type=ot
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} steps_pnp 100 alpha 0.8 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} steps_pnp 100 alpha 0.5 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} steps_pnp 500 alpha 0.01 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} steps_pnp 200 alpha 0.01 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


method=ot_ode
model_type=ot
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} steps_ode 100 start_time 0.3 gamma gamma_t max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} steps_ode 100 start_time 0.1 gamma gamma_t max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} steps_ode 100 start_time 0.1 gamma constant max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} steps_ode 100 start_time 0.1 gamma constant max_batch ${max_batch} batch_size_ip ${batch_size_ip}


method=flow_priors
model_type=ot
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} start_time 0.0 lmbda 100 eta 0.01  max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} start_time 0.0 lmbda 10000 eta 0.01  max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} start_time 0.0 lmbda 10000 eta 0.1  max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} start_time 0.0 lmbda 10000 eta 0.01  max_batch ${max_batch} batch_size_ip ${batch_size_ip}


method=ddnm
model_type=ddpm
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} sampling_steps 100 jump_length 1 jump_n_sample 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} sampling_steps 100 jump_length 1 jump_n_sample 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} sampling_steps 100 jump_length 1 jump_n_sample 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} sampling_steps 100 jump_length 1 jump_n_sample 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


method=repaint
model_type=ddpm
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} timesteps 250 jump_length 10 jump_n_sample 10 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} timesteps 250 jump_length 10 jump_n_sample 10 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} timesteps 250 jump_length 10 jump_n_sample 10 max_batch ${max_batch} batch_size_ip ${batch_size_ip}


method=d_flow
model_type=ot
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem denoising method ${method} lmbda 0.001 alpha 0.1 LBFGS_iter 3 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem box_inpainting method ${method} lmbda 0.01 alpha 0.1 LBFGS_iter 9 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem superresolution method ${method} lmbda 0.001 alpha 0.1 LBFGS_iter 20 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
python main.py --opts dataset ${dataset} eval_split ${eval_split} model_type ${model_type} problem random_inpainting method ${method} lmbda 0.001 alpha 0.1 LBFGS_iter 20 max_batch ${max_batch} batch_size_ip ${batch_size_ip}
