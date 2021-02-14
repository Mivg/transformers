#!/usr/bin/env bash
conda create -n scaling python=3.8 -y
source activate scaling
pip install transformers==4.3.2 tokenizers==0.10.1 datasets==1.2.1
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html  # alternative for cuda 10.1: pip install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboard==2.0.0 tensorflow==2.0.0 tensorflow-estimator==2.0.0 tensorflow_hub datasets==1.0.0
pip3 install --upgrade tensorflow-gpu  # for Python 3.n and GPU