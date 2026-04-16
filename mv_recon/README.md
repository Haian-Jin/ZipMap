# Multi-view Reconstruction (Point Map Estimation)

See configs in `configs/evaluation/mv_recon.yaml`.

```bash
# python mv_recon/sampling.py  # to generate seq-id-maps under datasets/seq-id-maps, which is provided in this repo
python mv_recon/eval.py
# torchrun --nnodes=1 --nproc_per_node=8 mv_recon/eval_mp.py  # accelerate with multi gpus
```

## Eval and Save Point Maps

Assume `eval_models` only have `pi3`, `eval.py` will generate folders like:

```
recons-eval
├── ...
├── outputs
|   └── mv_recon
|       ├── hydra (runtime configs)
|       └── pi3
|           ├── 7scenes-dense
|           |   ├── _seq_metrics
|           |   |   ├── sequence_1.csv
|           |   |   ├── sequence_2.csv
|           |   |   └── ...
|           |   ├── _seq_metrics.csv
|           |   ├── sequence_1-gt.ply
|           |   ├── sequence_1-pred.ply
|           |   ├── sequence_1.png
|           |   └── ...
|           ├── 7scenes-sparse
|           ├── DTU
|           ├── ETH3D
|           ├── NRGBD-dense
|           └── NRGBD-sparse
└── ...
```

## Use Your Own Sampling Strategy

seq-id-map records the mapping from sequence name to image ids of this sequence. We only use a subset of images in each sequence as model input, for example, we sample key frames every 5 frames in DTU and ETH3D. And sometimes we may want to sample randomly, precomputing a seq-id-map decouples deterministic evaluation from random sampling.

You can edit [configs/data/mv_recon.yaml](configs/data/mv_recon.yaml) and [mv_recon/sampling.py](mv_recon/sampling.py) to generate new seq-id-maps under [datasets/seq-id-maps](datasets/seq-id-maps). We have provided all necessary seq-id-maps for our evaluation settings.

If you want to use your own sampling strategy, you need to generate seq-id-maps before running `eval.py`, and specify it in `configs/data/mv_recon.yaml`.