# VENDORED VERBATIM from dino_wm/distributed_fn/__init__.py — needed by vqvae.py's Quantize.
from .distributed import (
    get_rank,
    get_local_rank,
    is_primary,
    synchronize,
    get_world_size,
    all_reduce,
    all_gather,
    reduce_dict,
    data_sampler,
    LOCAL_PROCESS_GROUP,
)
from .launch import launch
