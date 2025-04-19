# Dossard Finder: Identify Your Race Photos by Bib Number🏃‍♀️

Welcome to the final notebook of this project! This tool allows you to automatically find and extract photos containing your race bib number from a large set of images — perfect for marathon, triathlon, or race photographers and participants.

🔁 **Workflow Overview**\
This notebook uses your trained Faster R-CNN model to detect dossards (race bibs) in images, then applies OCR (text recognition) to read the bib numbers.

Before you begin:
- Connect to a GPU runtime via Runtime > Change runtime type > GPU.
- Mount your Google Drive to access your dataset and save results.
- Provide under User input
  - A folder with photos (uploaded to Google Drive)
  - Your trained model
  - The bib number(s) you want to search for

Based on that, the notebook will:
1. Load the images into a dataset
2. For each image:
  - Detect dossards using your fine-tuned Faster R-CNN model
  - Read the number(s) using EasyOCR — a lightweight, deep learning-based OCR tool that works well on real-world scene text
  - Store all results (image name, predicted numbers, scores, etc.) in a CSV file for reference
3. Finally, it compares the predicted numbers to your target bib number(s) (you can allow mismatches if needed) and copies matching photos into a new folder for easy access.

⚡ **Speed Things Up**\
If you've already run the detection and OCR step before and just want to search for a different number, you can skip directly to the last step by setting: `labels_exist = True`. This will load the existing CSV instead of reprocessing all images — saving a lot of time!

**Note:** This notebook requires a GPU runtime. Running on CPU is possible but will be very slow for larger image sets.

## User input


```python
# Your (and your friends) race numbers
target_numbers = ["2419", "2420"] # as strings

# input and output directories
image_dir = "/content/drive/MyDrive/Run/"
output_dir = "/content/drive/MyDrive/Run_output/"
path2weights = "/content/drive/MyDrive/Model/weights_mymodel.pt" # path to trained object detection model weights

# If you already labeled the images you can start from the .csv
labels_exists = False
label_path = "/content/drive/MyDrive/images_dossard_labels.csv"

# Advanced options
max_distance = 0  # theshold for target matching: allow x character mismatches between the target numbers and the detected strings
```

## Basic set-up


```python
# Connect to drive
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
# pip install required packages
if not labels_exists:
  try:
    import easyocr
  except ImportError:
    !pip install easyocr
    import easyocr
```

    Collecting easyocr
      Downloading easyocr-1.7.2-py3-none-any.whl.metadata (10 kB)
    Requirement already satisfied: torch in /usr/local/lib/python3.11/dist-packages (from easyocr) (2.6.0+cu124)
    Requirement already satisfied: torchvision>=0.5 in /usr/local/lib/python3.11/dist-packages (from easyocr) (0.21.0+cu124)
    Requirement already satisfied: opencv-python-headless in /usr/local/lib/python3.11/dist-packages (from easyocr) (4.11.0.86)
    Requirement already satisfied: scipy in /usr/local/lib/python3.11/dist-packages (from easyocr) (1.14.1)
    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from easyocr) (2.0.2)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.11/dist-packages (from easyocr) (11.1.0)
    Requirement already satisfied: scikit-image in /usr/local/lib/python3.11/dist-packages (from easyocr) (0.25.2)
    Collecting python-bidi (from easyocr)
      Downloading python_bidi-0.6.6-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.9 kB)
    Requirement already satisfied: PyYAML in /usr/local/lib/python3.11/dist-packages (from easyocr) (6.0.2)
    Requirement already satisfied: Shapely in /usr/local/lib/python3.11/dist-packages (from easyocr) (2.1.0)
    Collecting pyclipper (from easyocr)
      Downloading pyclipper-1.3.0.post6-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (9.0 kB)
    Collecting ninja (from easyocr)
      Downloading ninja-1.11.1.4-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl.metadata (5.0 kB)
    Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (3.18.0)
    Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (4.13.2)
    Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (3.4.2)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (3.1.6)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (2025.3.2)
    Collecting nvidia-cuda-nvrtc-cu12==12.4.127 (from torch->easyocr)
      Downloading nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cuda-runtime-cu12==12.4.127 (from torch->easyocr)
      Downloading nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cuda-cupti-cu12==12.4.127 (from torch->easyocr)
      Downloading nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cudnn-cu12==9.1.0.70 (from torch->easyocr)
      Downloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cublas-cu12==12.4.5.8 (from torch->easyocr)
      Downloading nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cufft-cu12==11.2.1.3 (from torch->easyocr)
      Downloading nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-curand-cu12==10.3.5.147 (from torch->easyocr)
      Downloading nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cusolver-cu12==11.6.1.9 (from torch->easyocr)
      Downloading nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cusparse-cu12==12.3.1.170 (from torch->easyocr)
      Downloading nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
    Requirement already satisfied: nvidia-cusparselt-cu12==0.6.2 in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (0.6.2)
    Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (2.21.5)
    Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (12.4.127)
    Collecting nvidia-nvjitlink-cu12==12.4.127 (from torch->easyocr)
      Downloading nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Requirement already satisfied: triton==3.2.0 in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (3.2.0)
    Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.11/dist-packages (from torch->easyocr) (1.13.1)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy==1.13.1->torch->easyocr) (1.3.0)
    Requirement already satisfied: imageio!=2.35.0,>=2.33 in /usr/local/lib/python3.11/dist-packages (from scikit-image->easyocr) (2.37.0)
    Requirement already satisfied: tifffile>=2022.8.12 in /usr/local/lib/python3.11/dist-packages (from scikit-image->easyocr) (2025.3.30)
    Requirement already satisfied: packaging>=21 in /usr/local/lib/python3.11/dist-packages (from scikit-image->easyocr) (24.2)
    Requirement already satisfied: lazy-loader>=0.4 in /usr/local/lib/python3.11/dist-packages (from scikit-image->easyocr) (0.4)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch->easyocr) (3.0.2)
    Downloading easyocr-1.7.2-py3-none-any.whl (2.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.9/2.9 MB[0m [31m32.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl (363.4 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m363.4/363.4 MB[0m [31m4.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (13.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m13.8/13.8 MB[0m [31m117.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (24.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m24.6/24.6 MB[0m [31m91.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (883 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m883.7/883.7 kB[0m [31m57.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl (664.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m664.8/664.8 MB[0m [31m1.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl (211.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m211.5/211.5 MB[0m [31m5.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl (56.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m56.3/56.3 MB[0m [31m12.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl (127.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m127.9/127.9 MB[0m [31m7.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl (207.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m207.5/207.5 MB[0m [31m5.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (21.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m21.1/21.1 MB[0m [31m95.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading ninja-1.11.1.4-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (422 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m422.8/422.8 kB[0m [31m32.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pyclipper-1.3.0.post6-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (969 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m969.6/969.6 kB[0m [31m60.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading python_bidi-0.6.6-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (292 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m292.9/292.9 kB[0m [31m30.8 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: python-bidi, pyclipper, nvidia-nvjitlink-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, ninja, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12, easyocr
      Attempting uninstall: nvidia-nvjitlink-cu12
        Found existing installation: nvidia-nvjitlink-cu12 12.5.82
        Uninstalling nvidia-nvjitlink-cu12-12.5.82:
          Successfully uninstalled nvidia-nvjitlink-cu12-12.5.82
      Attempting uninstall: nvidia-curand-cu12
        Found existing installation: nvidia-curand-cu12 10.3.6.82
        Uninstalling nvidia-curand-cu12-10.3.6.82:
          Successfully uninstalled nvidia-curand-cu12-10.3.6.82
      Attempting uninstall: nvidia-cufft-cu12
        Found existing installation: nvidia-cufft-cu12 11.2.3.61
        Uninstalling nvidia-cufft-cu12-11.2.3.61:
          Successfully uninstalled nvidia-cufft-cu12-11.2.3.61
      Attempting uninstall: nvidia-cuda-runtime-cu12
        Found existing installation: nvidia-cuda-runtime-cu12 12.5.82
        Uninstalling nvidia-cuda-runtime-cu12-12.5.82:
          Successfully uninstalled nvidia-cuda-runtime-cu12-12.5.82
      Attempting uninstall: nvidia-cuda-nvrtc-cu12
        Found existing installation: nvidia-cuda-nvrtc-cu12 12.5.82
        Uninstalling nvidia-cuda-nvrtc-cu12-12.5.82:
          Successfully uninstalled nvidia-cuda-nvrtc-cu12-12.5.82
      Attempting uninstall: nvidia-cuda-cupti-cu12
        Found existing installation: nvidia-cuda-cupti-cu12 12.5.82
        Uninstalling nvidia-cuda-cupti-cu12-12.5.82:
          Successfully uninstalled nvidia-cuda-cupti-cu12-12.5.82
      Attempting uninstall: nvidia-cublas-cu12
        Found existing installation: nvidia-cublas-cu12 12.5.3.2
        Uninstalling nvidia-cublas-cu12-12.5.3.2:
          Successfully uninstalled nvidia-cublas-cu12-12.5.3.2
      Attempting uninstall: nvidia-cusparse-cu12
        Found existing installation: nvidia-cusparse-cu12 12.5.1.3
        Uninstalling nvidia-cusparse-cu12-12.5.1.3:
          Successfully uninstalled nvidia-cusparse-cu12-12.5.1.3
      Attempting uninstall: nvidia-cudnn-cu12
        Found existing installation: nvidia-cudnn-cu12 9.3.0.75
        Uninstalling nvidia-cudnn-cu12-9.3.0.75:
          Successfully uninstalled nvidia-cudnn-cu12-9.3.0.75
      Attempting uninstall: nvidia-cusolver-cu12
        Found existing installation: nvidia-cusolver-cu12 11.6.3.83
        Uninstalling nvidia-cusolver-cu12-11.6.3.83:
          Successfully uninstalled nvidia-cusolver-cu12-11.6.3.83
    Successfully installed easyocr-1.7.2 ninja-1.11.1.4 nvidia-cublas-cu12-12.4.5.8 nvidia-cuda-cupti-cu12-12.4.127 nvidia-cuda-nvrtc-cu12-12.4.127 nvidia-cuda-runtime-cu12-12.4.127 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.2.1.3 nvidia-curand-cu12-10.3.5.147 nvidia-cusolver-cu12-11.6.1.9 nvidia-cusparse-cu12-12.3.1.170 nvidia-nvjitlink-cu12-12.4.127 pyclipper-1.3.0.post6 python-bidi-0.6.6
    


```python
# basic imports
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
import pandas as pd
import numpy as np
import os
import random
import copy
import torch
import cv2
import re

# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')
```


```python
# Creat output directory
os.makedirs(output_dir, exist_ok=True)
```

## Dataset Preparation & Dataloaders 📦
In this notebook section, we prepare our images for processing by the Faster-RCNN model by defining a custom PyTorch Dataset class and creating dataloaders. See "Dossard_Detection_with_FAster-RCNN.ipynb" for further details in the pre-processing steps.


```python
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.functional as TF


# Helper function to pad the image to a square
def pad_to_square(img, boxes, pad_value=0):
    w, h = img.size
    dim_diff = np.abs(h-w)
    pad1 = dim_diff // 2 # Floor division
    pad2 = dim_diff - pad1

    if h >= w: # Add padding where needed
        left, top, right, bottom = pad1, 0, pad2, 0
    else:
        left, top, right, bottom = 0, pad1, 0, pad2
    padding = (left, top, right, bottom)

    # Pad image
    img_padded = TF.pad(img, padding=padding, fill=pad_value)

    # Recalculate bbox coordinates
    if boxes is not None:
        if h >= w: # shift x_coords
            for box in boxes:
                box[0] += pad1
                box[2] += pad1
        else: # shift y_coords
            for box in boxes:
                box[1] += pad1
                box[3] += pad1

    return img_padded, boxes


# Helper function to flip images horizontally
def hflip(image, boxes):
    image_flipped = TF.hflip(image)
    w, h = image.size

    boxes_flipped = copy.deepcopy(boxes)
    if boxes is not None:
        for i, box in enumerate(boxes_flipped):
            box[0] = w - boxes[i][2] # x_min because x_max and visa versa
            box[2] = w - boxes[i][0]

    return image_flipped, boxes_flipped


# Helper function to resize image and bounding boxes
def img_resize(image, boxes, target_size=(224, 224)):
    w, h = image.size
    image_resized = TF.resize(image, target_size)

    if boxes is not None:
        w_factor, h_factor = target_size[0]/w, target_size[1]/h
        for box in boxes:
            box[0] *= w_factor
            box[1] *= h_factor
            box[2] *= w_factor
            box[3] *= h_factor

    return image_resized, boxes


# "transformer" function to chain transformations
def transformer(image, labels, params):
    if params["pad2square"] == True:
        image, labels = pad_to_square(image, labels, pad_value=0)
    if params["image_resize"] == True:
        image, labels = img_resize(image, labels, target_size=params["target_size"])
    if random.random() < params["p_hflip"]:
        image, labels = hflip(image, labels)
    return image, labels
```


```python
from torch.utils.data import Dataset


# Define custom Dataset
class RunnerDataset(Dataset):
    def __init__(self, img_dir, box_dir=None, return_img_name=False, transform=None, transform_params=None):
        self.return_img_name = return_img_name # __getitem__ should return img_name or not True/False
        self.transform = transform
        self.transform_params = transform_params
        self.classes = ['dossard'] # Only 1 class

        # Get image and box file names
        self.img_dir = img_dir
        self.box_dir = box_dir

        valid_extensions = ('.jpg', '.jpeg', '.png')
        self.imgs = [
            image for image in os.listdir(img_dir)
            if image.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(img_dir, image))
            ]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        # Get image
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        # Get boxes
        if self.box_dir:
            box_name = img_name.replace(".png",".txt").replace(".jpg", ".txt").replace(".jpeg", ".txt")
            box_path = os.path.join(self.box_dir, box_name)
            box_file = open(box_path)
            boxes = [line.rstrip().split(',') for line in box_file.readlines()]
            boxes = [list(map(int, bbox)) for bbox in boxes]
        else:
            boxes = None

        # Transform
        if self.transform:
            img, boxes = self.transform(img, boxes, self.transform_params)

        # Everything to tensors
        if boxes is not None and len(boxes) > 0: # Normalize to [0, 1]
          image_width, image_height = img.size
          boxes = torch.tensor(boxes, dtype=torch.float32)
        img = TF.to_tensor(img)

        # Create target dict
        target = {}
        if boxes is not None and len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)  # Important: int64!
            target["boxes"] = boxes
            target["labels"] = labels
            target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
            target["image_id"] = torch.tensor([idx], dtype=torch.int64)
        else:
            # Return empty target if no boxes — still valid
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
            target["image_id"] = torch.tensor([idx], dtype=torch.int64)

        if self.return_img_name == True:
          return img, target, img_name
        else:
          return img, target

```


```python
# Create the dataset
params = {
    "pad2square": True,
    "image_resize": True,
    "target_size": (1000, 1000), # High resolution to facilitate reading of the racenumbers
    "p_hflip": 0,
    }

test_dataset = RunnerDataset(image_dir, box_dir=None, return_img_name=True,
                             transform=transformer, transform_params=params)
```

## Load Faster R-CNN Model with Trained Weights 🧠
In this section, we will load a Faster R-CNN model from `torchvision` using pre-trained weights. This model has been fine-tuned on our dossard dataset, so we’ll now import the custom weights you trained earlier.


```python
import torchvision

# Function to load the model
def get_object_detection_model(num_classes):
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model

if not labels_exists:
  model = get_object_detection_model(num_classes=2) # dossards and background
  model.load_state_dict(torch.load(path2weights, map_location=torch.device('cpu')))
```

    Downloading: "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth" to /root/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
    100%|██████████| 167M/167M [00:00<00:00, 178MB/s]
    

## Dossard Detection & Labeling 🎯

In this section, we’ll run the core of our workflow: detecting dossards in photos and extracting their numbers.

🔍 **Step-by-step:**\
Loop through all images in the provided folder.
For each image:
1. Use the trained Faster R-CNN model to detect dossard bounding boxes.
2. For every detected dossard, apply multiple OCR passes with varying image preprocessing settings (e.g., upscaling, contrast enhancement, binarization).
3. Use EasyOCR to extract the dossard number from each box.
  - Merge similar predictions to reduce redundancy:
  - Remove non-digit characters.
  - Keep only the highest confidence for duplicate detections.
  - Optionally allow small mismatches (e.g., one digit off) to handle OCR inconsistencies.

All predictions are saved in a CSV file named:\
📄 images_dossard_labels.csv

This CSV file includes: Image name, Dossard bounding box coordinates, Detection confidence, OCR result, OCR confidence




```python
# Device agnostic code
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```

    cuda:0
    


```python
# Function to filter detected dossards by confidence score
def filter_by_score(prediction, thresh=0.2):
  keep = prediction["scores"] >= thresh
  indices = keep.nonzero(as_tuple=True)[0]

  final_prediction = {}
  final_prediction['boxes'] = prediction['boxes'][indices]
  final_prediction['scores'] = prediction['scores'][indices]
  final_prediction['labels'] = prediction['labels'][indices]

  return final_prediction
```


```python
# Function for object detection
def predict_dossards(model, reader, img, device, thresh_detection):
  model.eval()
  with torch.no_grad():
    prediction = model([img.to(device)])[0]
  prediction_filtered = filter_by_score(prediction, thresh=thresh_detection)
  return prediction_filtered


# Function to read text in box
def read_box(box, scale=2, bn=False, sharpen=False):

  # Initialize list to save text
  records = []

  # Convert to grayscale
  gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
  # Upscale
  gray_up = cv2.resize(gray, (gray.shape[1]*scale, gray.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
  # Normalize and enhance contrast
  gray_up = cv2.normalize(gray_up, None, 0, 255, cv2.NORM_MINMAX)
  # Ensure correct data type
  if gray_up.dtype != np.uint8:
    gray_up = (gray_up * 255).astype(np.uint8) if gray_up.max() <= 1 else gray_up.astype(np.uint8)

  # Sharpen
  if sharpen:
    gaussian_blur = cv2.GaussianBlur(gray_up, (21,21), sigmaX=2)
    gray_up = cv2.addWeighted(gray_up,7.5,gaussian_blur,-6.5,0)

  # Binarize
  if bn:
    bw = cv2.adaptiveThreshold(gray_up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)

  # Call easyocr to read text
  if not bn:
    result = reader.readtext(gray_up, detail=1)
  else:
    result = reader.readtext(bw, detail=1)

  # Loop over detected text
  for (bbox, text, prob) in result:
    if re.search(r"\d", text): # Check is contains numbers
      records.append({
          "ocr_text": text,
          "ocr_score": prob
      })

  return records


# Function to merge similair predications
def merge_predictions(predictions, score_threshold=0.5, max_digit_diff=1):

    # Filter and clean predictions (numeric only)
    filtered_predictions = []

    for pred in predictions:
        if pred:  # Ignore empty predictions
            # Extract text and score
            ocr_text = pred.get('ocr_text', '').strip()
            ocr_score = pred.get('ocr_score', 0)

            # Only keep numeric ocr_text (ignore non-digits)
            if re.match(r'^\d+$', ocr_text):  # Valid digits only
                filtered_predictions.append({'ocr_text': ocr_text, 'ocr_score': ocr_score})

    # Dictionary to merge predictions
    merged = {}

    for pred in filtered_predictions:
        text = pred['ocr_text']
        score = pred['ocr_score']

        if text in merged:
            # Keep the highest score for the same text
            if score > merged[text]['ocr_score']:
                merged[text] = pred
        else:
            merged[text] = pred

    # Optional: Handle similar predictions (with 1 digit difference)
    final_predictions = []
    seen = set()  # To avoid duplicates

    for text, pred in merged.items():
        # Check for similar text (1 digit difference)
        found_similar = False
        for existing_text in seen:
            if is_similar(text, existing_text, max_digit_diff):
                found_similar = True
                # Keep the highest score prediction if the condition matches
                if pred['ocr_score'] > merged[existing_text]['ocr_score']:
                    merged[existing_text] = pred
                break

        if not found_similar:
            seen.add(text)
            final_predictions.append(pred)
        elif pred['ocr_score'] > score_threshold:
            final_predictions.append(pred)

    return final_predictions


# Compare predictions
def is_similar(text1, text2, max_digit_diff):

    # If the lengths differ too much, they're definitely not similar
    if abs(len(text1) - len(text2)) > max_digit_diff:
        return False

    # Count the digit differences
    diff = sum(1 for a, b in zip(text1, text2) if a != b)
    # Allow for a small difference in digits (e.g., 1-digit difference)
    return diff <= max_digit_diff

```


```python
# Main function to detect and label dossards in images
def detect_and_label_dossards(dataset, model_detection, reader, device, thresh_detection=0.2):

  # Initialize list to save dossard numbers
  records = []

  # Loop over the images in the dataset
  for img, _, img_name in test_dataset:

    # Get predicted dossards per img
    predicted_dossards = predict_dossards(model, reader, img, device, thresh_detection)

    # Img to numpy and transpose
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # from (C, H, W) to (H, W, C)

    # Loop over dossards
    for i, box in enumerate(predicted_dossards["boxes"]):

      # Crop image to box
      box = box.detach().cpu().numpy()
      box = [int(n) for n in box]
      x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
      box_cropped = img[y1:y2, x1:x2, :]

      # Read text in box with different settings
      text_box_1 = read_box(box_cropped, scale=3, bn=False)
      text_box_2 = read_box(box_cropped, scale=6, bn=False)
      text_box_3 = read_box(box_cropped, scale=6, bn=False, sharpen=True)
      text_box_4 = read_box(box_cropped, scale=6, bn=True)

      text_predictions = text_box_1 + text_box_2 + text_box_3 + text_box_4

      merged_text = merge_predictions(text_predictions, score_threshold=0.5, max_digit_diff=1)

      # Save text
      for text_scores in merged_text:
        records.append({
            "image_name": img_name,
            "bbox": [x1, y1, x2, y2],
            "bbox_score": predicted_dossards["scores"][i].detach().cpu().numpy(),
            "ocr_text": text_scores['ocr_text'],
            "ocr_score": text_scores['ocr_score']
            })

  return records


```


```python
# Set-up easyocr reader
if not labels_exists:
  reader = easyocr.Reader(['en'], gpu=True, recog_network='english_g2')  # g2 is better at digits
```

    WARNING:easyocr.easyocr:Downloading detection model, please wait. This may take several minutes depending upon your network connection.
    

    Progress: |██████████████████████████████████████████████████| 100.0% Complete

    WARNING:easyocr.easyocr:Downloading recognition model, please wait. This may take several minutes depending upon your network connection.
    

    Progress: |██████████████████████████████████████████████████| 100.0% Complete


```python
if not labels_exists:
  model.to(device)

  # Call main function
  records = detect_and_label_dossards(test_dataset, model, reader, device, thresh_detection=0.2)
  results_df = pd.DataFrame(records)
```


```python
if not labels_exists:
  # Save results_df
  from pathlib import Path

  out_path = output_dir + "images_dossard_labels.csv"
  filepath = Path(out_path)
  filepath.parent.mkdir(parents=True, exist_ok=True)
  results_df.to_csv(filepath, index=False)
```

## Filter & Save Images 🗂️

Now that we have all dossard predictions stored in a CSV file, we’ll use it to find and extract the images containing the race numbers you're interested in.

✅ What this step does:
1. Loads the "images_dossard_labels.csv" file with all detected dossard numbers.
2. Filters the results to match your target number(s):
  - You can allow for a specific number of mismatches (e.g., 1 digit off, missing digit).
3. Copies the matching images into your specified output directory for easy access.

🔁 If the CSV already exists from a previous run and you set `labels_exists = True`, the notebook skips the detection step and goes straight to this filtering and saving process.


```python
import shutil
from difflib import SequenceMatcher

# Function to check similarity
def is_similar(pred_text, target, max_distance=1):
    pred_digits = ''.join(filter(str.isdigit, str(pred_text)))
    if not pred_digits:
        return False
    # Use simple Levenshtein distance-like logic
    sm = SequenceMatcher(None, pred_digits, target)
    ratio = sm.ratio()
    distance = max(len(pred_digits), len(target)) * (1 - ratio)
    return distance <= max_distance


# Function to Find and copy images containing the target racenumbers
def find_and_save_images(results_df, target_number, image_dir, output_dir, max_distance=1):

  # Get matching image names
  matching_images = set()
  for _, row in results_df.iterrows():
      if is_similar(row['ocr_text'], target_number, max_distance=max_distance):
          matching_images.add(row['image_name'])

  # Copy files
  for img_name in matching_images:
      src_path = os.path.join(image_dir, img_name) # source
      dst_dir = os.path.join(output_dir, target_number) # destination
      os.makedirs(dst_dir, exist_ok=True)
      dst_path = os.path.join(dst_dir, img_name) # destination

      if os.path.exists(src_path):
          shutil.copy(src_path, dst_path)
          print(f"Copied {img_name}")
```


```python
if labels_exists: # Load csv with detected and labeled dossards
  results_df = pd.read_csv(label_path)

for number in target_numbers:
  find_and_save_images(results_df, number, image_dir, output_dir, max_distance=1)
```

    Copied 54439228390_1f6e21dd10_o.jpg
    Copied 54439051369_21a2c3b0e8_o.jpg
    Copied 54439228395_d7364a4166_o.jpg
    Copied 54439051369_21a2c3b0e8_o.jpg
    
