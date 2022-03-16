# NL_trajectory_reshaper
Reshaping Robot Trajectories Using Natural Language Commands: A Study of Multi-Modal Data Alignment Using Transformers


## setup


environment setup
```
#create anew conda environment:
conda create --name py38 --file spec-file.txt python=3.8
conda activate py38
#install CLIP
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

## Relevant files
overview of the project
```model_overview.ipynb```

plots and ablasion studies
```plots_and_ablasion_exp.ipynb```

generate syntetic dataset
```data_generator/data_generator.py```

## generate syntetic dataset




## ROS setup:
TODO


