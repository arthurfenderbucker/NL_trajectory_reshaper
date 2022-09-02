### Setup isntruction for devices requiring cuda 11
e.g. RTX3090

```
conda create --name py38_cu11 --file setup_cuda11.txt python=3.8
conda activate py38_cu11
```

Install [torch](https://pytorch.org/) e.g.
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```


Install [TensorFlow with pip](https://www.tensorflow.org/install/pip) e.g.
```
conda install -c conda-forge cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
python3 -m pip install tensorflow
```

add ```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/```
to your .bashrc or run it after activating the environment allow the tf to locate the libcudnn.so.8

install the rest of the pip:
```
pip install -r setup_cuda11_pip.txt
```