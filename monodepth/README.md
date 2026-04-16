# Monocular Depth Estimation

See configs in `configs/evaluation/monodepth.yaml`.

```bash
python monodepth/infer.py
# torchrun --nnodes=1 --nproc_per_node=8 monodepth/infer_mp.py  # accelerate with multi gpus
python monodepth/eval.py
```

## Infer to Generate .npy & .png Depth Files

Assume `eval_models` only have `pi3`, `infer.py` will generate folders like:

```
recons-eval
├── ...
├── outputs
|   └── monodepth
|       ├── hydra (runtime configs)
|       └── pi3
|           ├── bonn
|           |   ├── sequence_1
|           |   |   ├── xxxxxx.npy
|           |   |   ├── xxxxxx.png
|           |   |   └── ...
|           |   ├── sequence_2
|           |   └── ...
|           ├── kitti
|           ├── nyu-v2
|           └── sintel
└── ...
```

## Eval with Generated Depth Files

After `infer.py` finishes, you can run `eval.py` to evaluate the results.

Then the monodepth metrics will be generated in `outputs/monodepth/{dataset_name}-metric.csv`.

```
recons-eval
├── ...
├── outputs
|   └── monodepth
|       ├── hydra (runtime configs)
|       ├── pi3
|       |   ├── bonn
|       |   |   ├── sequence_1
|       |   |   |   ├── xxxxxx.npy
|       |   |   |   ├── xxxxxx.png
|       |   |   |   └── ...
|       |   |   ├── sequence_2
|       |   |   └── ...
|       |   ├── kitti
|       |   ├── nyu-v2
|       |   └── sintel
|       ├── bonn-metric.csv
|       ├── kitti-metric.csv
|       ├── nyu-v2-metric.csv
|       └── sintel-metric.csv
└── ...
```