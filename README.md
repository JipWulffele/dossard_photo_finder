# 🏃‍♂️ Dossard Photo Finder

Automatically detect your race bib number in large collections of race photos using deep learning and OCR.

This project was born after running the Grenoble-Vizille trail and wanting to find my own photos among over 1,000 images. Manual browsing? No thanks. This repo helps you detect and extract only the photos that include your bib number.

---

## 🚀 What it does

- Trains a custom object detection model (Faster R-CNN) to detect **dossards** (race bibs).
- Uses OCR (EasyOCR) to read the **bib number** on each dossard.
- Matches against your desired bib number (with optional tolerance for mismatches).
- Copies matching photos to a separate folder — so you can enjoy your glory shots, faster.

---

## 📓 Notebooks & Workflow

The repo is organized around **three notebooks**. Each one tackles a specific part of the pipeline.

### 📘 1. Convert VGG CSV to YOLO-style TXT

**Notebook:** `1_convert_vgg_to_txt.ipynb`

- Converts bounding box annotations from [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/) format to `.txt` files for training.
- VGG uses `x1, y1, width, height` format — this notebook converts it to `[x1, y1, x2, y2]`.
- You’ll mount your Google Drive, provide the input CSV and output directory.
- No GPU needed here.

---

### 📙 2. Train Faster R-CNN for Dossard Detection

**Notebook:** `2_train_faster_rcnn.ipynb`

- Fine-tunes a pre-trained **Faster R-CNN** model from `torchvision.models` to detect dossards.
- Works great with a small dataset (as few as 100 annotated images).
- Requires:
  - Drive paths to images and annotation `.txt` files
  - An output path to save your trained model (`.pth`)
- Requires a GPU (Colab GPU works well).
- Training on 100 images takes ~20 minutes (Colab GPU).

---

### 📕 3. Detect & Find Photos by Bib Number

**Notebook:** `3_detect_and_find_bibs.ipynb`

- Loads your trained model and a folder of race images.
- For each image:
  - Detects dossards using the Faster R-CNN model
  - Reads bib numbers using EasyOCR (run multiple OCR passes with different pre-processing to maximize success)
- Merges OCR predictions smartly (filters noise and near-duplicates).
- Matches against your provided bib number (with optional tolerance).
- Copies the relevant images to a separate output folder.
- Detection + OCR takes ~10 minutes for 1,000 images.
- Can optionally skip detection if CSV of predictions already exists (`labels_exists = True`).

---

## 📸 Example Input
Some example data is included in the example_data/ folder:
- Sample race images
- Sample .txt annotation files
- A demo VGG-format CSV

---

## ⚠️ Limitations & Suggestions
- The model was only trained on ~100 images from a single race — all dossards followed similar formats.
  - You may need to fine-tune again for other events or styles.
- OCR struggles with:
  - Tilted/angled bibs
  - Motion blur or low contrast
- Allowing mismatches (e.g. 1-digit off) leads to some false positives — but still drastically reduces the search space (from 1000s of images to 10s).
- EasyOCR is fast but not perfect — exploring more robust OCR models (or even digit classifiers) would improve performance.
- Could be expanded with face recognition or clothing features to track runners even from the side or back.
- Currently works with local folders in Google Drive. A web interface with image upload and background processing would be a nice next step.

---

## 💻 Running on Colab
This project was developed and tested on Google Colab using the free GPU runtime. Recommended!

---

## 🙌 Credits
Created by a trail running enthusiast with a data science itch to scratch 🏔️
