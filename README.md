# NL_trajectory_reshaper
Reshaping Robot Trajectories Using Natural Language Commands: A Study of Multi-Modal Data Alignment Using Transformers


## setup

[install anaconda](https://docs.anaconda.com/anaconda/install/linux/)

environment setup
```
conda create --name py38 --file spec-file.txt python=3.8
conda activate py38
```
install CLIP
```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

download model
```
gdown --folder https://drive.google.com/drive/folders/1HQNwHlQUOPMnbPE-3wKpIb6GMBz5eqDg?usp=sharing -O models/.
```
download syntetic dataset
```
gdown --folder https://drive.google.com/drive/folders/1_bhWWa9upUWwUs7ln8jaWG_bYxtxuOCt?usp=sharing -O data/.
```

## Running visual demo


```
python modify_draw.py
```

---
## ROS setup:

> **IMPORTANT:** make sure that conda is not being initialized in your .bashrc file, otherwise you might face conficts between the python versions 

[install ROS melodic](http://wiki.ros.org/melodic/Installation/Ubuntu)

[manually install CVbridge](https://cyaninfinite.com/ros-cv-bridge-with-python-3/)

## Running with ROS
terminal 1
```
roscore
```
terminal 2
```
roscd NL_trajectory_reshaper/src
source $HOME/cvbridge_build_ws/install/setup.bash --extend
python modify_draw.py --ros true
```


## Other relevant files
overview of the project
[model_overview.ipynb](model_overview.ipynb)


plots and ablasion studies
[plots_and_ablasion_exp.ipynb](plots_and_ablasion_exp.ipynb)

generate syntetic dataset
[data_generator/data_generator.py](data_generator/data_generator.py)

