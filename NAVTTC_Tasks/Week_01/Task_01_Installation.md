# Task 1 - Installation

Install these (links from the course file):
- Anaconda3: https://www.anaconda.com/
- PyTorch: https://pytorch.org/get-started/locally/
- TensorFlow 2.0: https://www.tensorflow.org/install
- VSCode: https://code.visualstudio.com/
- PyCharm (Community Edition): https://www.jetbrains.com/pycharm/

Commands:
```
conda create -n navttc python=3.10 -y
conda activate navttc
pip install numpy pandas matplotlib seaborn scikit-learn scipy
pip install tensorflow
```
For PyTorch copy the command for your system from its Get Started page.

Check the installation works:
```
python -c "import numpy, pandas; print(numpy.__version__, pandas.__version__)"
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import torch; print(torch.__version__)"
```
