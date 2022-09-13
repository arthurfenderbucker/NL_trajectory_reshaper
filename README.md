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
pip install ftfy regex tqdm dqrobotics rospkg similaritymeasures
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
> **NOTE:** this is the catkin config that I used to intall CVbridge with the Anaconda </br>
```catkin config -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python -DPYTHON_INCLUDE_DIR=$CONDA_PREFIX/include/python3.8 -DPYTHON_LIBRARY=$CONDA_PREFIX/lib/libpython3.8.so -DSETUPTOOLS_DEB_LAYOUT=OFF```

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

---
## coppelia_simulator + ROS + anaconda setup
install coppelia simulator
https://www.coppeliarobotics.com/helpFiles/en/ros1Tutorial.htm
add ```export COPPELIASIM_ROOT_DIR=~/path/to/coppeliaSim/folde``` to your ~/.bashrc

```
cd <ros_workspace>/src
git clone https://github.com/CoppeliaRobotics/ros_bubble_rob
git clone --recursive https://github.com/CoppeliaRobotics/simExtROS.git sim_ros_interface
cd <ros_workspace>
```

```
catkin config -DPYTHON_EXECUTABLE=$HOME/anaconda3/envs/py38/bin/python -DPYTHON_INCLUDE_DIR=$HOME/anaconda3/envs/py38/include/python3.8 -DPYTHON_LIBRARY=$HOME/anaconda3/envs/py38/lib/libpython3.8.so -DSETUPTOOLS_DEB_LAYOUT=OFF

catkin config --install
catkin build
```

## Other relevant files
overview of the project
[model_overview.ipynb](model_overview.ipynb)


plots and ablation studies
[plots_and_ablasion_exp.ipynb](plots_and_ablasion_exp.ipynb)

generate syntetic dataset
[data_generator/data_generator.py](data_generator/data_generator.py)

