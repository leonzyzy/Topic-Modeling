import torch

def last_true_index(x: torch.Tensor, dim: int = 1, default: int = -1) -> torch.Tensor:
    """
    Returns the last index of True along the specified dimension in a 2D boolean tensor.

    Args:
        x (torch.Tensor): A 2D boolean tensor.
        dim (int): The dimension to search along (default is 1, i.e., per row).
        default (int): Value to return if no True is found in a row/column.

    Returns:
        torch.Tensor: 1D tensor of indices indicating the last True per row/column.
                      If no True is found, returns `default`.
    """
    if x.dtype != torch.bool:
        raise ValueError("Input tensor must be of boolean dtype.")

    rev_x = x.flip(dims=[dim])
    last_true_from_end = rev_x.float().argmax(dim=dim)
    size = x.size(dim)
    last_true = size - 1 - last_true_from_end

    # Mask for rows/cols with no True
    no_true_mask = x.any(dim=dim) == False
    last_true[no_true_mask] = default

    return last_true
