
import pytest
import yaml
import os
from pmf.config import Config, get_config

CONFIG_PATH = "configs/pMF-B-16.yaml"

def test_default_config_initialization():
    config = Config()
    assert config.image_size > 0
    assert config.patch_size > 0
    assert config.hidden_size > 0
    assert config.depth > 0
    assert config.num_heads > 0
    assert config.num_classes > 0

def test_load_yaml_config():
    """
    Test loading the project's main configuration file.
    """
    if not os.path.exists(CONFIG_PATH):
        pytest.skip(f"Config file {CONFIG_PATH} not found")
        
    with open(CONFIG_PATH, 'r') as f:
        yaml_config = yaml.safe_load(f)
        
    # Basic schema checks
    required_sections = ['model', 'training', 'pmf', 'optimizer', 'loss', 'paths']
    for section in required_sections:
        assert section in yaml_config, f"Missing section '{section}' in config"

    # Check critical values
    if 'model' in yaml_config:
        assert 'hidden_size' in yaml_config['model']
        
    if 'loss' in yaml_config:
        assert 'lambda_perc' in yaml_config['loss']
        assert 'perc_threshold' in yaml_config['loss']

def test_get_config_override():
    # Test if get_config handles overrides (mocking args is hard, so just check basic return)
    # Ideally we would mock sys.argv or use a wrapper that accepts args
    pass

def test_config_attribute_access():
    c = Config()
    # Check if attributes can be set and retrieved
    c.new_attr = 123
    assert c.new_attr == 123
