import pytest
import torch
from pmf.config import Config
from pmf.pixel_mean_flow import PixelMeanFlow
from pmf.dit import DiT


@pytest.fixture
def config():
    c = Config()
    c.image_size = 32
    c.patch_size = 4
    c.hidden_size = 32
    c.depth = 1
    c.num_heads = 4
    c.num_classes = 11
    c.lambda_perc = 0.0
    return c


@pytest.fixture
def model(config):
    return DiT(
        input_size=config.image_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        num_classes=config.num_classes,
        class_dropout_prob=config.class_dropout_prob,
        learn_sigma=config.learn_sigma,
    )


@pytest.fixture
def pmf(model, config):
    return PixelMeanFlow(model, config)

def test_config_has_sampling_params(config):
    assert hasattr(config, "sampling_dist")
    assert hasattr(config, "logit_normal_loc")
    assert hasattr(config, "logit_normal_scale")
    assert hasattr(config, "uniform_prob")


def test_config_has_cfg_params(config):
    assert hasattr(config, "cfg_training")
    assert hasattr(config, "cfg_scale_min")
    assert hasattr(config, "cfg_scale_max")

@pytest.mark.parametrize("sampling_dist", ["uniform", "logit_normal"])
def test_sample_t_r_bounds(pmf, config, sampling_dist):
    config.sampling_dist = sampling_dist
    t, r = pmf.sample_t_r(256, torch.device("cpu"))
    assert t.shape == (256,)
    assert r.shape == (256,)
    assert torch.all(t >= 0) and torch.all(t <= 1)
    assert torch.all(r >= 0) and torch.all(r <= t)


def test_logit_normal_sampling_extremes(pmf, config):
    config.sampling_dist = "logit_normal"
    config.logit_normal_scale = 1.0
    config.uniform_prob = 0.0

    config.logit_normal_loc = 100.0
    t_hi, _ = pmf.sample_t_r(256, torch.device("cpu"))
    assert t_hi.mean() > 0.9

    config.logit_normal_loc = -100.0
    t_lo, _ = pmf.sample_t_r(256, torch.device("cpu"))
    assert t_lo.mean() < 0.1

def test_sample_t_r_cfg_shapes_and_constraints(pmf, config):
    config.cfg_scale_min = 1.0
    config.cfg_scale_max = 2.0
    t, r, w, interval = pmf.sample_t_r_cfg(128, torch.device("cpu"))
    assert t.shape == (128,)
    assert r.shape == (128,)
    assert w.shape == (128,)
    assert interval.shape == (128, 2)
    assert torch.all(w >= config.cfg_scale_min) and torch.all(w <= config.cfg_scale_max)
    assert torch.all(interval[:, 0] >= 0) and torch.all(interval[:, 1] <= 1)
    assert torch.all(interval[:, 0] <= interval[:, 1])
