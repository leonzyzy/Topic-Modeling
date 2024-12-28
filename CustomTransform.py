
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
torchrun --nnodes 2 --nproc_per_node 2 --node_rank 0 --master_addr <IP ADDRESS OF NODE 0> --master_port 29332 multi_node_test.py
