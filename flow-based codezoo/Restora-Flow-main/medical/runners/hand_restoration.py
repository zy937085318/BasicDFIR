import torch
from generative_model.unet import create_model
from generative_model.sampling_xray import Flow, GaussianDiffusion
from dataset import XrayHand38ChannelsImageGenerator


class HandRestoration:
    def __init__(self, model_type, experiment_name="full"):
        self.model_type = model_type
        self.experiment_name = experiment_name

        self.input_size = 256
        self.in_channels = self.out_channels = 38
        self.num_features = 192

        self.dataset = XrayHand38ChannelsImageGenerator(
            input_size=self.input_size,
            heatmap_sigma=8,
            experiment_name=experiment_name,
            train=False,
        )

        if model_type == "flow":
            self.model_path = "model_checkpoints/xray_hand/flow/full/model.pt"
        else:   # ddpm
            self.model_path = "model_checkpoints/xray_hand/ddpm/full/model.pt"

        self.model = None

    def load_model(self):
        unet = create_model(
            self.input_size,
            self.num_features,
            1,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            dims=2,
        ).cuda()

        if self.model_type == "flow":
            self.model = Flow(
                unet,
                image_size=self.input_size,
                channels=self.out_channels,
                ode_steps=32,
            ).cuda()
        else:
            self.model = GaussianDiffusion(
                unet,
                image_size=self.input_size,
                channels=self.out_channels,
                timesteps=50,
            ).cuda()

        state = torch.load(self.model_path)
        self.model.load_state_dict(state["ema"])

    def run(self, problem, method, degradation, sigma_y, params, output_folder):
        flow_methods = {
            "Restora-Flow": "solve_ip_restora_flow",
            "OT-ODE": "solve_ip_ot_ode",
            "PnP-Flow": "solve_ip_pnp_flow",
            "Flow-Priors": "solve_ip_flow_priors",
            "D-Flow": "solve_ip_d_flow",
        }

        diffusion_methods = {
            "RePaint": "solve_ip_repaint",
            "DDNM": "solve_ip_ddnm",
        }

        valid_methods = (
            flow_methods if self.model_type == "flow" else diffusion_methods
        )

        if method not in valid_methods:
            allowed = ", ".join(valid_methods.keys())
            raise ValueError(
                f"Method '{method}' is not supported for model type '{self.model_type}'. "
                f"Allowed methods: {allowed}"
            )

        solve_name = valid_methods[method]
        solve_fn = getattr(self.model, solve_name)

        return solve_fn(
            self.dataset,
            problem,
            degradation,
            sigma_y,
            params,
            output_folder=output_folder,
        )
