# Facial Emotion Recognition (FER2013)

CS412 – CS Elective 2 (Affective Computing), Week 7 laboratory: Facial Expression Analysis.

**Author:** Jhey Gulde (CS412 - CS4A), individual work.

A simple end-to-end facial emotion recognition system on the FER2013 dataset (7 classes: angry, disgust, fear, happy, sad, surprise, neutral):

1. **Sanity check**: class folders, image counts and the 48×48 grayscale image spec.
2. **Face detection** (Milestone 1): OpenCV Haar cascade with bounding boxes drawn per class.
3. **Feature extraction** (Milestone 2, Track B): crop the largest face, resize it to 224×224, convert it to 3 channels, and take a 1280-d embedding from a pretrained MobileNetV2 (ImageNet weights, frozen).
4. **Models**: Logistic Regression (standardized features) and Random Forest, with a fixed seed of 42.
5. **Evaluation** (Milestone 3): accuracy, macro-F1, per-class classification reports, 7×7 confusion matrices and misclassified examples.
6. **Deployment**: `predict_emotion(image_path)` returns one of the seven labels for a new image.

## Project layout

```
Week7_Facial_Emotion_Recognition.ipynb   # main notebook
models/haarcascade_frontalface_default.xml
OUTPUTS/                                  # figures, metrics, classification reports
unseen_images/                            # images held out for the predict-function demo
data/                                     # FER2013 (not committed, see below)
```

## How to run

1. Download FER2013 (Kaggle `msambare/fer2013`) and extract it so that you have `data/train/<emotion>/` and `data/test/<emotion>/`.
2. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook and run all cells. The MobileNetV2 weights download automatically on the first run, and the CNN embeddings are cached in `features/` so later runs are fast.
