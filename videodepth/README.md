# Video Depth Estimation

See configs in `configs/evaluation/videodepth.yaml`.

```bash
python videodepth/infer.py
# torchrun --nnodes=1 --nproc_per_node=8 videodepth/infer_mp.py  # accelerate with multi gpus
python videodepth/eval.py              # align=scale&shift by default
python videodepth/eval.py align=scale  # override with align=scale
```

## Infer to Generate .npy & .png Depth Files

Assume `eval_models` only have `pi3`, `infer.py` will generate folders like:

```
recons-eval
├── ...
├── outputs
|   └── videodepth
|       ├── hydra (runtime configs)
|       └── pi3
|           ├── bonn
|           |   ├── sequence_1
|           |   |   ├── _time.json
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

Then the videodepth metrics will be generated in `outputs/videodepth/{dataset_name}-metric-{align}.csv`. For example, if `align=scale`, then

```
recons-eval
├── ...
├── outputs
|   └── videodepth
|       ├── hydra (runtime configs)
|       ├── pi3
|       |   ├── bonn
|       |   |   ├── sequence_1
|       |   |   |   ├── _time.json
|       |   |   |   ├── xxxxxx.npy
|       |   |   |   ├── xxxxxx.png
|       |   |   |   └── ...
|       |   |   ├── sequence_2
|       |   |   └── ...
|       |   ├── kitti
|       |   └── sintel
|       ├── bonn-metric-scale.csv
|       ├── kitti-metric-scale.csv
|       └── sintel-metric-scale.csv
└── ...
```