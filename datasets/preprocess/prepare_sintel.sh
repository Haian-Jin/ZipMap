# Reference: https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md, https://github.com/Junyi42/monst3r/blob/main/data/download_sintel.sh
# The scripts provided here are for reference only. Please ensure you have obtained the necessary licenses from the original dataset providers before proceeding.

mkdir sintel     # if pwd is project root, `mkdir -p data/sintel`
cd sintel        # if pwd is project root, `cd data/sintel` 

# images
wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip  # --no-proxy in original script
unzip MPI-Sintel-training_images.zip
rm MPI-Sintel-training_images.zip

# depth & cameras
wget http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip  # --no-proxy in original script
unzip MPI-Sintel-depth-training-20150305.zip
rm MPI-Sintel-depth-training-20150305.zip

cd ..            # if pwd is project root, `cd ../..`