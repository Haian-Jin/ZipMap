# Reference: https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md, https://github.com/Junyi42/monst3r/blob/main/data/download_bonn.sh
# The scripts provided here are for reference only. Please ensure you have obtained the necessary licenses from the original dataset providers before proceeding.

mkdir -p data/bonn
cd data/bonn

wget https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_dataset.zip
unzip rgbd_bonn_dataset.zip
rm rgbd_bonn_dataset.zip

cd ../..

python datasets/preprocess/prepare_bonn.py
