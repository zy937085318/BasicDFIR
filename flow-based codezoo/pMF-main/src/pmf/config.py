# Configuration for Pixel MeanFlow (pMF)

import os
import yaml

class Config:
    def __init__(self, config_dict=None):
        if config_dict:
            self.update_from_dict(config_dict)
        else:
            # Defaults
            self.image_size = 256
            self.patch_size = 16
            self.in_channels = 3
            self.hidden_size = 1152
            self.depth = 28
            self.num_heads = 16
            self.mlp_ratio = 4.0
            self.class_dropout_prob = 0.1
            self.num_classes = 1001
            self.learn_sigma = False

            self.global_batch_size = 1024
            self.micro_batch_size = 32
            self.learning_rate = 1e-4
            self.weight_decay = 0.01
            self.num_epochs = 160
            self.mixed_precision = "fp16"
            self.gradient_accumulation_steps = 1

            self.t_min = 0.0
            self.t_max = 1.0

            self.use_muon = True
            self.muon_lr = 0.02
            self.muon_momentum = 0.95
            self.adam_lr = 1e-4
            self.adam_beta1 = 0.9
            self.adam_beta2 = 0.95

            self.lambda_perc = 1.0
            self.perc_threshold = 0.6

            # Sampling defaults
            self.sampling_dist = "logit_normal"
            self.logit_normal_loc = 0.0
            self.logit_normal_scale = 0.8
            self.uniform_prob = 0.1

            self.cfg_training = False
            self.cfg_scale_min = 1.0
            self.cfg_scale_max = 7.0

            # (pMF): 论文 Appendix A 表述“维护多个 EMA decay 并在推理选择最佳”，但未在正文中给出 decay 列表；此处提供常见默认值，需与官方实现/表8对齐。
            self.ema_decays = [0.9999]
            # (pMF): 论文 Table 8/实现细节中应包含精确 warmup 与 schedule；此处仅保留占位默认值，需补齐后对齐。
            self.warmup_steps = 5000
            self.warmup_ratio = 0.03

            self.data_path = os.environ.get("IMAGENET_PATH", "/data2/private/huangcheng/data/imagenet-1k-256x256-modelscope")
            self.output_dir = "results"
            self.checkpoint_dir = "checkpoints"
            self.log_dir = "logs"

            self.seed = 42

    def update_from_dict(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                self.update_from_dict(v)
            else:
                if hasattr(self, k):
                    setattr(self, k, v)
                else:
                    # Allow new keys from yaml (e.g. nested structure flattened or specific keys)
                    setattr(self, k, v)

def load_config(path=None):
    if path is None:
        path = os.environ.get("PMF_CONFIG")
    if path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates = [
            os.path.join(base_dir, "configs", "pMF-B-16.yaml"),
            os.path.join(base_dir, "configs", "config.yaml"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), candidates[0])

    if os.path.exists(path):
        with open(path, "r") as f:
            yaml_config = yaml.safe_load(f)

        config = Config()
        config.update_from_dict(yaml_config)
        return config
    else:
        print(f"Warning: {path} not found. Using defaults.")
        return Config()

def get_config(path=None):
    return load_config(path)
