# On terminal:
# pip install ultralytics  # install packages
# pip list  # list all packages installed in venv

# Import necessary libraries
from pathlib import Path

# import ultralytics
# ultralytics.checks() #checking setup

from ultralytics import YOLO

print(Path.cwd())

####################################################################################################
## YoloV8 model training
####################################################################################################


# define yaml file
yaml_filename = 'my-config.yaml'

# yaml specifications:
# train: datasets/train/images  # 1: all paths are relative to project's working directory
# val: datasets/valid/images    # 2: a 'datasets' subfolder must be created inside working directory
# nc: 1                         # 3: both train and val must be located inside 'datasets' subfolder
# names: ['bees']


# Full path of yaml file
full_path = Path(yaml_filename).resolve()

# Load a model
model = YOLO('yolov8n.pt') # load a pretrained model (recommended for training)

results = model.train(data= str(full_path), epochs=30, patience=5, batch=16, imgsz=640)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
