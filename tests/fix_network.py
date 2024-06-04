import rootutils
import torch
from rich.console import Console


rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
import os

import numpy as np

from src.data.components import BURGERS
from src.models.components import CAMO
from src.models.components.dataclass_config import (
    CrossAttentionEncoderConfig,
    InputEncoderConfig,
    NormlizationConfig,
    ParamInitConfig,
    PosEncodingConfig,
    PropagatorDecoderConfig,
    QueryEncoderConfig,
)


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.multiprocessing.set_sharing_strategy("file_system")
torch.autograd.set_detect_anomaly(True)


console = Console()


def get_activation(name):
    def hook(module, input, output):
        print("layer name", name)
        console.print(f"=> layer_name: {name}")
        console.print(f"{output}")

    return hook


dataset = BURGERS(root="/home/jc/dev/CAMO/data")


x, y, input_pos, query_pos = dataset.__getitem__(0)
x = torch.cat((x, input_pos), dim=-1)

# add first dim for inputs
x = torch.unsqueeze(x, dim=0)
input_pos = torch.unsqueeze(input_pos, dim=0)
query_pos = torch.unsqueeze(query_pos, dim=0)
print(input_pos.shape)

# TODO: sync configs between CAMO and OFormer
model = CAMO(
    input_encoder_config=InputEncoderConfig(
        attn_type="galerkin",
        in_features=2,
        out_features=96,
        d_model=96,
        nhead=1,
        num_layers=4,
        scale=[8, 4, 4, 1],
        norm_first=False,
        pos_config=PosEncodingConfig(min_freq=1 / 2048),
        init_config=ParamInitConfig(
            method="orthogonal",
            gain=1 / 96,
            diagonal_weight=1 / 96,
        ),
    ),
    query_encoder_config=QueryEncoderConfig(),
    crossattention_encoder_config=CrossAttentionEncoderConfig(),
    propagator_decoder_config=PropagatorDecoderConfig(),
)

for name, layer in model.named_modules():
    layer.register_forward_hook(get_activation(name))

# excute forward check
console.print(f"x: {x}")
pred = model(x, input_pos, query_pos)

# TODO: 1. init in encoder
