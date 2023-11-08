conda create -n envpool_env python=3.9
conda activate envpool_env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install envpool==0.8.2
pip install tensorboard
pip install opencv-python==4.8.0.76
conda install -c conda-forge gym-box2d
pip install
pip install "gym[accept-rom-license, atari]"

