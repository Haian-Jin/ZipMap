# Reference: https://github.com/facebookresearch/vggt/blob/evaluation/evaluation/README.md
# The scripts provided here are for reference only. Please ensure you have obtained the necessary licenses from the original dataset providers before proceeding.

# firstly, Download the Co3Dv2 dataset from [the official repository](https://github.com/facebookresearch/co3d)
# download it to `data/co3dv2/data`
mkdir -p data/co3dv2/data
mkdir -p data/co3dv2/annotations

# generate annotations
python preprocess_co3d.py --category all --co3d_v2_dir data/co3dv2/data --output_dir data/co3dv2/annotations