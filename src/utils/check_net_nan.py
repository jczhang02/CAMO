from typing import Any, List

import numpy as np
import rich
import torch
from rich.console import Console


__all__ = ["check_net_value_rich", "check_net_grad"]


def check_net_value_rich(net: torch.nn.Module) -> None:
    """print the name, parameter value and grad of all network layer.

    Parameters
    ----------
    net : torch.nn.Module
        The network needed to be checked.

    """
    name_list: List[str] = []
    value_list: List[Any] = []
    grad_list: List[Any] = []

    console = Console()

    for name, p in net.named_parameters():
        name_list.append(name)
        value_list.append(p.detach().cpu().numpy() if p is not None else [0])
        grad_list.append(p.grad.detach().cpu().numpy() if p.grad is not None else [np.nan])

    for i in range(len(name_list)):
        style_v = "bold green"
        style_g = "bold green"

        if np.max(value_list[i]).item() - np.min(value_list[i]).item() < 1e-6:
            style_v = "bold red"
        if np.isnan(value_list[i]).any():
            style_v = "underline red"
        if np.isnan(grad_list[i]).any():
            style_g = "underline red"

        console.print(
            f"value {name_list[i]}: {np.min(value_list[i]):.3e} ~ {np.max(value_list[i]):.3e}", style=style_v
        )
        console.print(f"grad  {name_list[i]}: {np.min(grad_list[i]):.3e} ~ {np.max(grad_list[i]):.3e}", style=style_g)

    return


def check_net_grad(net) -> None:
    for name, parms in net.named_parameters():
        rich.print(
            "-->name:",
            name,
            "-->grad_requirs:",
            parms.requires_grad,
            "--weight",
            torch.mean(parms.data),
            " -->grad_value:",
            torch.mean(parms.grad),
        )
