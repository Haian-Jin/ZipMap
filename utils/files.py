import os
import os.path as osp
import glob
import re
from typing import Optional
from omegaconf import DictConfig, ListConfig


def get_all_sequences(dataset_cfg: DictConfig, sort_by_seq_name: bool = True):
    if isinstance(dataset_cfg.ls_all_seqs, str):
        # if ls_all_seqs is a string, it is the root path of sequences
        seq_list = [d for d in os.listdir(dataset_cfg.ls_all_seqs) if osp.isdir(osp.join(dataset_cfg.ls_all_seqs, d))]
    elif isinstance(dataset_cfg.ls_all_seqs, ListConfig):
        # if ls_all_seqs is a ListConfig, it is the ListConfig of sequence names
        seq_list = dataset_cfg.ls_all_seqs
    else:
        raise ValueError(f"Unknown ls_all_seqs type: {type(dataset_cfg.ls_all_seqs)}, ls_all_seqs is {dataset_cfg.ls_all_seqs}, which should be a string or a ListConfig")
    return sorted(seq_list) if sort_by_seq_name else seq_list

def _natural_key(path: str):
    """Return a key for natural sorting of filenames.

    Splits the basename by digit groups and converts digit parts to integers
    so that filenames like '2.png' < '10.png'.
    """
    name = osp.basename(path)
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def list_imgs_a_sequence(dataset_cfg: DictConfig, seq: Optional[str] = None):
    subdir = dataset_cfg.img.path.format(seq=seq)  # string include {seq}
    ext = dataset_cfg.img.ext
    filelist = sorted(glob.glob(f"{subdir}/*.{ext}"), key=_natural_key)
    return filelist

def list_depths_a_sequence(dataset_cfg: DictConfig, seq: Optional[str] = None):
    subdir = dataset_cfg.depth.path.format(seq=seq)  # string include {seq}
    ext = dataset_cfg.depth.ext
    filelist = sorted(glob.glob(f"{subdir}/*.{ext}"), key=_natural_key)
    return filelist