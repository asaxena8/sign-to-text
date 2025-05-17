import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('model/sl_model.h5')
classes = [chr(i) for i in range(65, 91) if chr(i) != ord('J')]

cap = cv2.VideoCapture(0)
current_letter = ""

def predict_letter(roi):
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        norm = resized.reshape(1, 28, 28, 1).astype("float32") / 255.0
        pred = model.predict(norm)
        idx = np.argmax(pred)
        conf = np.max(pred)
        print(f"[DEBUG] {classes[idx]} ({conf:.2f})")
        return classes[idx] if conf > 0.7 else "-"
    except Exception as e:
        print(f"[ERROR] {e}")
        return "-"

def update_frame():
    global current_letter
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        x1, y1, x2, y2 = 200, 100, 500, 400  # try this
        roi = frame[y1:y2, x1:x2]
        letter = predict_letter(roi)
        current_letter = letter
        letter_label.config(text=f"Predicted: {letter}")
        cv2.imshow("Model Input (ROI)", roi)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        cam_label.imgtk = imgtk  # prevent garbage collection
        cam_label.configure(image=imgtk)

    root.after(10, update_frame)

def on_close():
    cap.release()
    root.destroy()

root = tk.Tk()
root.title("Sign Prediction")
root.geometry("800x650")
root.configure(bg="white")
root.protocol("WM_DELETE_WINDOW", on_close)

title = Label(root, text="Sign Language to Speech", font=("Helvetica", 22, "bold"), bg='white')
title.pack(pady=10)

cam_label = Label(root, bg='black', width=640, height=480)
cam_label.pack()

letter_label = Label(
    root,
    text="Predicted: -",
    font=("Helvetica", 24, "bold"),
    bg="white",
    fg="black"
)
letter_label.pack(pady=10)

update_frame()
root.mainloop()
