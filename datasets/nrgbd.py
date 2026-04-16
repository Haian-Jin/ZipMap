import os.path as osp
import os 
import numpy as np
import torch
import cv2
import imageio.v2
import torchvision.transforms as tvf

from typing import Optional, Union, List
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm
import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from models.vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from datasets.utils.cropping import resize_image, resize_image_depth_and_intrinsic

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
to_tensor = tvf.ToTensor()

class NRGBD(Dataset):
    def __init__(
        self,
        NRGBD_DIR: str,
        load_img_size: int = 518,
        cache_file: str = "data/dataset_cache/nrgbd_mv_recon_cache.npy",
    ):
        
        self.NRGBD_DIR = NRGBD_DIR
        if NRGBD_DIR == None:
            raise NotImplementedError
        print(f"NRGBD_DIR is {NRGBD_DIR}")

        if osp.exists(cache_file):
            print(f"[NRGBD] Loading from cache file: {cache_file}")
            self.metadata = np.load(cache_file, allow_pickle=True).item()
            self.sequence_list = sorted(list(self.metadata.keys()))
        else:
            print(f"[NRGBD] Cache file not found, loading from {NRGBD_DIR}")

            self.sequence_list = sorted(
                [d for d in os.listdir(NRGBD_DIR) if osp.isdir(osp.join(NRGBD_DIR, d))]
            )
            
            self.metadata = {}
            for seq in tqdm(self.sequence_list):
                seq_path = osp.join(NRGBD_DIR, seq)
                
                list_seqs_func = lambda subdir, prefix, suffix: sorted(
                    [d for d in os.listdir(osp.join(seq_path, subdir)) if d.endswith(suffix)],
                    key=lambda file: int(file.split('.')[0][len(prefix):])
                )
                seq_rgbs = list_seqs_func(subdir='images', prefix='img', suffix='.png')
                seq_depths = list_seqs_func(subdir='depth', prefix='depth', suffix='.png')

                seq_pose_file = osp.join(seq_path, 'poses.txt')
                with open(seq_pose_file, 'r') as f:
                    seq_poses_rawlines = [line.strip() for line in f.readlines()]

                seq_poses = []
                lines_per_matrix = 4
                for i in range(0, len(seq_poses_rawlines), lines_per_matrix):
                    pose_floats = [
                        [float(x) for x in line.split()]
                        for line in seq_poses_rawlines[i : i + lines_per_matrix]
                    ]
                    seq_poses.append(pose_floats)

                # return np.array(seq_poses, dtype=np.float32)

                if not (len(seq_rgbs) == len(seq_depths) == len(seq_poses)):
                    raise ValueError(f"Sequence {seq} has mismatched number of RGB, depth, and pose files: {len(seq_rgbs)}, {len(seq_depths)}, {len(seq_poses)}")

                if not (int(seq_rgbs[0].split('.')[0][len('img'):]) == int(seq_depths[0].split('.')[0][len('depth'):]) == 0):
                    raise ValueError(f"First frame of sequence {seq} has number of RGB and depth != 0: {seq_rgbs[0]}, {seq_depths[0]}, {seq_poses[0]}")
                if not (int(seq_rgbs[-1].split('.')[0][len('img'):]) == int(seq_depths[-1].split('.')[0][len('depth'):]) == len(seq_rgbs) - 1):
                    raise ValueError(f"Last frame of sequence {seq} has number of RGB and depth != N-1: {seq_rgbs[-1]}, {seq_depths[-1]}, N={len(seq_rgbs)}")
                
                # camera_pose[:, 1:3] *= -1.0
                seq_poses = np.array(seq_poses, dtype=np.float32)
                seq_poses[:, :, 1:3] *= -1.0  # gl to cv
                seq_extrs = closed_form_inverse_se3(seq_poses)[:, :3, :]
                self.metadata[seq] = seq_extrs

            os.makedirs(osp.dirname(cache_file), exist_ok=True)
            np.save(cache_file, self.metadata)

        self.load_img_size = load_img_size
        print(f"[NRGBD] Data size: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def get_seq_framenum(self, index: Optional[int] = None, sequence_name: Optional[str] = None):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        return self.metadata[sequence_name].shape[0]

    def __getitem__(self, idx_N):
        """Fetch item by index and a dynamic variable n_per_seq."""

        # Different from most pytorch datasets,
        # here we not only get index, but also a dynamic variable n_per_seq
        # supported by DynamicBatchSampler

        index, n_per_seq = idx_N
        sequence_name = self.sequence_list[index]
        metadata = self.metadata[sequence_name]
        ids = np.random.choice(len(metadata), n_per_seq, replace=False)
        return self.get_data(index=index, ids=ids)

    def get_data(
            self,
            index: Optional[int] = None,
            sequence_name: Optional[str] = None,
            ids: Union[List[int], np.ndarray, None] = None,
        ):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name: str = self.sequence_list[index]
        seq_extrinsics: np.ndarray = self.metadata[sequence_name]  # (N, 3, 4)
        seq_len: int = seq_extrinsics.shape[0]

        if ids is None:
            ids = np.arange(seq_len).tolist()
        elif isinstance(ids, np.ndarray):
            assert ids.ndim == 1, f"ids should be a 1D array, but got {ids.ndim}D"
            ids = ids.tolist()

        fx, fy, cx, cy = 554.2562584220408, 554.2562584220408, 320, 240  # hard code

        image_paths: list      = [""] * len(ids)
        images: list           = [0]  * len(ids)
        depths: list           = [0]  * len(ids)
        extrinsics: np.ndarray = seq_extrinsics[ids]  # (N, 3, 4) -> (S, 3, 4)
        intrinsics: np.ndarray = np.tile(
            np.array(
                [
                    [fx, 0,  cx],
                    [0,  fy, cy],
                    [0,  0,  1 ]
                ],
                dtype=np.float32
            ),
            reps=(len(ids), 1, 1)
        )  # (S, 3, 3)

        for id_index, id in enumerate(ids):
            
            impath    = osp.join(self.NRGBD_DIR, sequence_name, "images", f"img{id}.png")
            depthpath = osp.join(self.NRGBD_DIR, sequence_name, "depth", f"depth{id}.png")

            rgb_image: Image.Image = Image.open(impath)
            depthmap: np.ndarray   = imageio.v2.imread(depthpath)
            assert depthmap.shape == (480, 640), f"Depth map shape {depthmap.shape} does not match expected (480, 640)"
            rgb_image: Image.Image = resize_image(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
            depthmap[depthmap > 10] = 0    # to far, invalid
            depthmap[depthmap < 1e-3] = 0  # to near, invalid

            rgb_image, depthmap, intrinsics[id_index] = resize_image_depth_and_intrinsic(
                image=rgb_image,
                depth_map=depthmap,
                intrinsic=intrinsics[id_index],
                output_width=self.load_img_size, # finally width = 518, height = 388
            )

            image_paths[id_index] = impath
            images[id_index]      = to_tensor(rgb_image)
            depths[id_index]      = depthmap

        depths = np.array(depths)  # (S, H, W)
        pointclouds = unproject_depth_map_to_point_map(
            depth_map=depths[..., None],
            intrinsics_cam=intrinsics,
            extrinsics_cam=extrinsics
        )

        batch = {"seq_id": sequence_name, "seq_len": seq_len, "ind": torch.tensor(ids)}
        batch['image_paths'] = image_paths  # list of str
        batch['images']      = torch.stack(images, dim=0)
        batch['pointclouds'] = pointclouds  # in numpy
        batch['valid_mask']  = depths > 1e-4
        # batch["extrs"] = extrinsics
        # batch["intrs"] = intrinsics
        # batch["w"] = metadata["w"]
        # batch["h"] = metadata["h"]

        return batch
