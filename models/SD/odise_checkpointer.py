# ------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# To view a copy of this license, visit
# https://github.com/facebookresearch/detectron2/blob/main/LICENSE
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import os.path as osp
from collections import defaultdict
from typing import List
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts
from fvcore.common.checkpoint import Checkpointer

# from odise.utils.file_io import PathManager


def _longest_common_prefix(names: List[str]) -> str:
    """
    ["abc.zfg", "abc.zef"] -> "abc."
    """
    names = [n.split(".") for n in names]
    m1, m2 = min(names), max(names)
    ret = []
    for a, b in zip(m1, m2):
        if a == b:
            ret.append(a)
        else:
            # break for the first non-matching element
            # Fixing BUG in detectron2
            break
    ret = ".".join(ret) + "." if len(ret) else ""
    return ret


def group_by_prefix(names):
    grouped_names = defaultdict(list)

    for name in names:
        grouped_names[name.split(".")[0]].append(name)

    return grouped_names

class LdmCheckpointer(Checkpointer):
    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(
            model=model, save_dir=save_dir, save_to_disk=save_to_disk, **checkpointables
        )
        # self.path_manager = PathManager

    def _load_model(self, checkpoint):
        # rename the keys in checkpoint
        checkpoint["model"] = checkpoint.pop("state_dict")
        return super()._load_model(checkpoint)
