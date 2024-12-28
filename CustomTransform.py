
import os

import torch
import torch.distributed


def run():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"initializing rank: {rank}/{world_size} - waiting for other node(s) to join")

    # initialize the process group
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    print(f"initialized rank: {rank}/{world_size} - all nodes joined")


if __name__ == "__main__":
    run()
