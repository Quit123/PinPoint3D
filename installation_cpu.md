# Installation Instruction (CPU-only version)

We recommend setting up the environment using Miniconda or Anaconda. We have tested the code on Linux with Python 3.10, PyTorch 2.3.1 (CPU), but it should also work in other environments. 

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

### Step 2: Install PyTorch (CPU version)
```shell
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
```
### Step 3: Install Minkowski Engine (CPU version)

3.1 Install dependencies:
```shell
conda install -c intel mkl mkl-include
```

3.2 Install MinkowskiEngine from source:
```shell
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas=mkl --cpu_only
cd ..
```

**Alternative method** (if above fails):
```shell
pip install MinkowskiEngine --no-deps
```

If you encounter issues, refer to MinkowskiEngine's [official CPU-only compilation guide](https://nvidia.github.io/MinkowskiEngine/quick_start.html#cpu-only-compilation).

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
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import MinkowskiEngine as ME; print('MinkowskiEngine installed successfully')"
python -c "import numpy, matplotlib, wandb, open3d; print('All packages installed')"
```

### Performance Note
⚠️ **Training on CPU is extremely slow!** A single epoch that takes minutes on GPU may take hours on CPU. Consider:
- Using a smaller dataset for testing
- Reducing the number of points per scene
- Using lower batch sizes
- Cloud GPU services (Google Colab, AWS, etc.) for actual training
