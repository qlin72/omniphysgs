# conda create -n omniphysgs python=3.11.9
# conda activate omniphysgs
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
pip install -r settings/requirements.txt

pip install -e third_party/gaussian-splatting/submodules/diff-gaussian-rasterization/
pip install -e third_party/gaussian-splatting/submodules/simple-knn/