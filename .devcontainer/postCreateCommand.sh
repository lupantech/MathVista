git config --global safe.directory '*'
git config --global core.editor "code --wait"
git config --global pager.branch false

# Activate conda by default
echo "source /home/vscode/miniconda3/bin/activate" >> ~/.zshrc
echo "source /home/vscode/miniconda3/bin/activate" >> ~/.bashrc

# Use environment by default
echo "conda activate mathvista" >> ~/.zshrc
echo "conda activate mathvista" >> ~/.bashrc

# Activate conda on current shell
source /home/vscode/miniconda3/bin/activate

# Create and activate mathvista environment
conda create -n mathvista python=3.10 -y
conda activate mathvista

# Install Nvidia Cuda Compiler
CUDA_VERSION=11.8
conda install -y -c nvidia cuda=$CUDA_VERSION cuda-nvcc=$CUDA_VERSION

pip install --upgrade pip
pip install -r requirements.txt

echo "postCreateCommand.sh completed!"
