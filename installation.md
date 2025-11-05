# Installation Instruction

We recommend setting up the environment using Miniconda or Anaconda. We have tested the code on Linux with Python 3.10, PyTorch 2.3.1, and CUDA 11.8, but it should also work in other environments. If you have trouble, feel free to open an issue.

### Step 1: create an environment
Clone this repo:
```shell
git clone https://github.com/your-repo/PinPoint3D.git
cd PinPoint3D
```

Create and activate conda environment:
```shell
conda create -n pinpoint3d python=3.10
conda activate pinpoint3d
```
### Step 2: Install PyTorch
For CUDA 12.4 (tested):
```shell
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
```
### Step 3: install Minkowski
3.1 Prepare:
```shell
conda install openblas-devel -c anaconda
```
3.2 Install:
```shell
# adjust your cuda path accordingly!
export CUDA_HOME=/usr/local/cuda

# For pip >= 23.0, use this command:
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
    --config-settings="--blas_include_dirs=${CONDA_PREFIX}/include" \
    --config-settings="--blas=openblas"

# If you get "no such option" error, downgrade pip:
pip install "pip<23.0"
# Then use the old install method:
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
    --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
    --install-option="--blas=openblas"
```

#### For CUDA 12.0+:
There are known compatibility issues. Try the following workarounds:
```shell
# Option 1: Install from source with modifications
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas=openblas
```

**References for CUDA 12+ issues:**
- https://github.com/NVIDIA/MinkowskiEngine/issues/594
- https://github.com/NVIDIA/MinkowskiEngine/issues/543  
- https://github.com/NVIDIA/MinkowskiEngine/issues/596

If you encounter issues, refer to the [official installation guide](https://github.com/NVIDIA/MinkowskiEngine#installation).
### Step 4: Install other required packages
```shell
# Essential packages (tested versions)
pip install numpy==2.2.5 matplotlib==3.10.1 scipy==1.15.2
pip install open3d==0.19.0
pip install wandb==0.19.8  # For experiment tracking
pip install plyfile==1.1  # For PLY file I/O
```

### Step 5: Verify installation
Test if everything is installed correctly:
```python
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import MinkowskiEngine as ME; print('MinkowskiEngine installed successfully')"
python -c "import numpy, matplotlib, wandb, open3d; print('All packages installed')"
```

### Troubleshooting

**1. CUDA version mismatch:**
- Ensure PyTorch CUDA version matches your system CUDA version
- Use `nvidia-smi` to check system CUDA version

**2. MinkowskiEngine compilation errors:**
- Make sure CUDA_HOME is set correctly: `export CUDA_HOME=/usr/local/cuda`
- Check gcc version is compatible (gcc 7-11 recommended)
- For CUDA 12+, refer to the MinkowskiEngine issues linked above

**3. Memory errors during training:**
- Reduce batch size in training arguments
- Use gradient checkpointing if available

**4. Import errors:**
- Ensure you've activated the correct conda environment: `conda activate pinpoint3d`
- Check if all packages are installed in the current environment: `pip list`
