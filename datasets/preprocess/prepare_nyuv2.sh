# Reference: https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md, https://github.com/Junyi42/monst3r/blob/main/data/download_nyuv2.sh
# The scripts provided here are for reference only. Please ensure you have obtained the necessary licenses from the original dataset providers before proceeding.

mkdir -p data/nyu-v2
cd data/nyu-v2

wget https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/main/data/val-000000.tar -O val-000000.tar
wget https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/main/data/val-000001.tar -O val-000001.tar

# unzip all
find . -name "*.tar" -exec tar -xvf {} \;
rm *.tar
cd ../..

# prepare the dataset for depth evaluation
python datasets/preprocess/prepare_nyuv2.py
