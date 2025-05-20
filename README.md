# 🤟 Sign Language to Text Converter

This project is a real-time sign language alphabet recognition tool using a webcam. It uses a trained CNN model on the Sign Language MNIST dataset and converts hand signs (A–Z excluding J) into text a

---

## 🖥️ Features

- Live webcam feed with Tkinter GUI
- Predicts American Sign Language (ASL) hand signs in real-time
- Displays predicted letter in GUI
- Optional ROI capture for dataset building
- Can be extended to sentence-building and audio playback


---

## 📦 Requirements

- Python 3.7+
- Install dependencies:

```bash
pip install tensorflow
pip install opencv-python
pip install pillow
pip install numpy
pip install gTTS
```

---

## ▶️ Run the App

Make sure your model file `sl_model.h5` is placed inside a `model/` folder.

```bash
python app.py
```

---

## 📸 Optional: Capture ROI for Custom Dataset

To collect your own training data:
- Position your hand in the green box
- Add a "Capture ROI" button (I can help with this if not already present)
- Images will be saved inside `/captured/` for each gesture

---

## 🧠 Model Info

- Trained on the [Sign Language MNIST dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- Input: 28x28 grayscale images
- Output: 25 classes (A–Z excluding J)

---

## 📌 Limitations

- Not accurate in poor lighting
- Sign J and Z are not supported (require motion)
- Webcam input may not exactly match training data format

---

## ✨ Future Improvements

- Build full sentences
- Add backspace/clear functionality
- Improve accuracy by training with custom webcam data
- Add audio playback for predicted words

---

## 🤝 Contributors

- Built by Akriti Saxena
- Model: TensorFlow / Keras
- UI: Tkinter + OpenCV + PIL
