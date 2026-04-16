# ZipMap Evaluation
Our evaluation code is heavily built on top of the repository [ZhouTimeMachine/recons_eval](https://github.com/ZhouTimeMachine/recons_eval) provided by the authors of [Pi3](https://github.com/yyfz/Pi3). We sincerely thank the authors for their open-source contributions. Our modifications include:
- We incorporate our model, and some baseline models (Cut3R, TTT3R) into the framework.
- We change the resolutions for models with patch size 14 (Pi3, VGGT, Our ZipMap) from 512 to 518
- We include the runtime measurement code for all methods, which is not included in the original repository. We also provide a script to run the evaluation with multiple GPUs.




<Details>
<summary><strong>Original ReadMe from "ZhouTimeMachine/recons_eval"</strong></summary>

# Evaluation for Reconstruction

A unified evaluation framework for 3D reconstruction, used in [π³](https://arxiv.org/abd/2507.13347).

This repo includes unofficial inference of some popular methods (e.g. [VGGT](https://github.com/facebookresearch/vggt), [MoGe](https://github.com/microsoft/MoGe)), if authors of these methods have any concerns about our implementation, please feel free to pull request or issue. (pull request is welcome!)

## Evaluation Overview

- Monocular Depth Estimation
- Video Depth Estimation
- Relative Camera Pose Estimation
- Multi-view Reconstruction (Point Map Estimation)
- Runtime Measurement


The root config file of all evaluations is `configs/eval.yaml`, however you don't need to edit it

- All main hyperparameters you need are in `configs/evaluation/xxxxx.yaml`
- Sometimes you may want to change the dataset config in `configs/data/xxxxx.yaml`, or the model config in `configs/model/xxxxx.yaml`

## Dataset Preparation

- **Depth Estimation**: We follow [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) to prepare Sintel, Bonn, KITTI and NYU-v2.
- **Camera Pose Estimation**
    - Angular: We follow [VGGT](https://github.com/facebookresearch/vggt/blob/evaluation/evaluation/README.md) to prepare Co3Dv2, and we afford [our script](datasets/preprocess/prepare_re10k.sh) for RealEstate10k preprocessing.
    - Distance: We follow [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) to prepare Sintel, TUM-dynamics and ScanNetv2.
- **Point Map Estimation**: We follow [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to prepare 7-Scenes, Neural-NRGBD and DTU. We afford [our script](datasets/preprocess/prepare_eth3d.sh) for ETH3D preprocessing.

> We provide reference-only preprocessing scripts under `datasets/preprocess`. Please ensure you have obtained the necessary licenses from the original dataset providers before proceeding.

## 1. Monocular Depth Estimation

See [monodepth/README.md](monodepth/README.md) for more details.

```bash
python monodepth/infer.py
# torchrun --nnodes=1 --nproc_per_node=8 monodepth/infer_mp.py  # accelerate with multi gpus
python monodepth/eval.py
```

## 2. Video Depth Estimation

configs in `configs/evaluation/videodepth.yaml`, see [videodepth/README.md](videodepth/README.md) for more details.

```bash
python videodepth/infer.py
# torchrun --nnodes=1 --nproc_per_node=8 videodepth/infer_mp.py  # accelerate with multi gpus
python videodepth/eval.py
```

## 3. Relative Camera Pose Estimation

configs in `configs/evaluation/relpose-angular.yaml`, see [relpose/README.md](relpose/README.md) for more details.

### 3.1 Angular Metrics

```bash
# python relpose/sampling.py  # to generate seq-id-maps under datasets/seq-id-maps, which is provided in this repo
python relpose/eval_angle.py
# torchrun --nnodes=1 --nproc_per_node=8 videodepth/eval_angle_mp.py   # accelerate with multi gpus
```

### 3.2 Distance Metrics

```bash
python relpose/eval_dist.py
# torchrun --nnodes=1 --nproc_per_node=8 videodepth/eval_dist_mp.py  # accelerate with multi gpus
```

## 4. Multi-view Reconstruction (Point Map Estimation)

See [mv_recon/README.md](mv_recon/README.md) for more details.

```bash
# python mv_recon/sampling.py  # to generate seq-id-maps under datasets/seq-id-maps, which is provided in this repo
python mv_recon/eval.py
# torchrun --nnodes=1 --nproc_per_node=8 mv_recon/eval_mp.py  # accelerate with multi gpus
```

## 5. Runtime Measurement

```bash
python speed_test/eval.py
```

## Acknowledgement

Our work mainly builds upon:
- [Recons_eval](https://github.com/ZhouTimeMachine/recons_eval)
- [Pi3](https://github.com/yyfz/Pi3)
- [DUSt3R](https://github.com/naver/dust3r)
- [MonST3R](https://github.com/Junyi42/monst3r)
- [Spann3R](https://github.com/HengyiWang/spann3r)
- [CUT3R](https://github.com/CUT3R/CUT3R)
- [MoGe](https://github.com/microsoft/MoGe)
- [VGGT](https://github.com/facebookresearch/vggt)

## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{wang2025pi3,
      title={$\pi^3$: Scalable Permutation-Equivariant Visual Geometry Learning}, 
      author={Yifan Wang and Jianjun Zhou and Haoyi Zhu and Wenzheng Chang and Yang Zhou and Zizun Li and Junyi Chen and Jiangmiao Pang and Chunhua Shen and Tong He},
      year={2025},
      eprint={2507.13347},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.13347}, 
}
```

## License

This project is licensed under CC BY-NC-SA 4.0 License. See the LICENSE file and https://creativecommons.org/licenses/by-nc-sa/4.0/ for details.

</Details>