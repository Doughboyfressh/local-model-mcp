import importlib.util
from pathlib import Path

import pytest
import torch

torch.set_num_threads(1)


@pytest.fixture
def workspace(tmp_path):
    path = Path(__file__).parents[1] / "examples" / "make_tiny_model.py"
    spec = importlib.util.spec_from_file_location("tiny_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_tiny(tmp_path / "lab")
