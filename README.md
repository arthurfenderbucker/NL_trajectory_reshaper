> **IMPORTANT:** :exclamation:repo under maintenance:exclamation:

# NL trajectory reshaper

Reshaping Robot Trajectories Using Natural Language Commands: A Study of Multi-Modal Data Alignment Using Transformers


![iterative NL interactions over a trajectory](./docs/media/interactions.gif)

_example of multiple iterative interactions over the initial erroneous
trajectory (red)_


## setup
<sub>_tested on Ubuntu 18.04 and 20.04_</sup>

[install anaconda](https://docs.anaconda.com/anaconda/install/linux/)

Environment setup
```
conda create --name py38 --file spec-file.txt python=3.8
conda activate py38
```
Install CLIP + opencv
```
pip install ftfy regex tqdm dqrobotics
pip install git+https://github.com/openai/CLIP.git
pip install opencv-python
```


Download models

```
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1HQNwHlQUOPMnbPE-3wKpIb6GMBz5eqDg?usp=sharing -O models/.
```
Download syntetic dataset
```
gdown --folder https://drive.google.com/drive/folders/1_bhWWa9upUWwUs7ln8jaWG_bYxtxuOCt?usp=sharing -O data/.
```

## Running the visual demo

```
cd src
python modify_draw.py
```

**How to use:**

1) press 'o' to load the original trajectory
2) press 'm' to modify the trajectory using our model for the given input on top.
3) press 't' to set a different interaction text.
4) press 'u' to update the trajctory setting the modified traj as the original one



---
## ROS setup:

> **IMPORTANT:** make sure that conda isn't initialized in your .bashrc file, otherwise, you might face conflicts between the python versions 

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


plots and ablation studies
[plots_and_ablasion_exp.ipynb](plots_and_ablasion_exp.ipynb)

generate syntetic dataset
[data_generator/data_generator.py](data_generator/data_generator.py)

