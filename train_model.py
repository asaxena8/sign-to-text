import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load dataset
train_df = pd.read_csv("sign_mnist_train.csv")
X = train_df.drop("label", axis=1).values
y = train_df["label"].values

# Preprocessing
X = X.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y = to_categorical(y)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(25, activation="softmax")  # A–Y (excluding J, Z)
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save model
model.save("model/sl_model.h5")
