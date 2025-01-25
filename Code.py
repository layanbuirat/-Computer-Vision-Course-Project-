# Step 1: Import necessary libraries
#haneen odeh & leyan burait 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import random

# Set the seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def generate_csv(data_path, images_csv, labels_csv, image_size=(64, 64)):
    image_data = []  # Initialize the list to store image data
    labels = []      # Initialize the list to store labels

    # Traverse through each folder (representing a class/label)
    for root, _, files in os.walk(data_path):
        label = os.path.basename(root)  # Get the directory name (e.g., user001)

        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  # Include valid image extensions
                image_path = os.path.join(root, file)

                # Read and preprocess the image
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    image = cv2.resize(image, image_size)  # Resize to desired size
                    image_data.append(image.flatten())  # Flatten the image for CSV storage
                    labels.append(label)  # Use the directory name as the label

# Convert to DataFrame and save as CSV
    if image_data and labels:  # Ensure data exists before saving
        pd.DataFrame(image_data).to_csv(images_csv, header=None, index=False)
        pd.DataFrame(labels).to_csv(labels_csv, header=None, index=False)
        print(f"CSV files generated: {images_csv}, {labels_csv}")
    else:
        print("No data found. Please check your dataset.")
    

# Step 3: Data Preprocessing and Splitting with Separate Label CSVs
def split_and_save_data_with_labels(
    images_csv, labels_csv, train_csv, val_csv, test_csv, train_label_csv, val_label_csv, test_label_csv
):

    # Load the CSV files
    images = pd.read_csv(images_csv, header=None).values
    labels = pd.read_csv(labels_csv, header=None).values.flatten()

    # Encode labels into integers
    unique_labels = list(set(labels))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_labels = np.array([label_to_int[label] for label in labels])

    # Split data into training (60%), validation (20%), and testing (20%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(images, int_labels, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Save each set to CSV files
    pd.DataFrame(X_train).to_csv(train_csv, header=None, index=False)
    pd.DataFrame(X_val).to_csv(val_csv, header=None, index=False)
    pd.DataFrame(X_test).to_csv(test_csv, header=None, index=False)

    # Save labels to separate CSV files
    pd.DataFrame(y_train).to_csv(train_label_csv, header=None, index=False)
    pd.DataFrame(y_val).to_csv(val_label_csv, header=None, index=False)
    pd.DataFrame(y_test).to_csv(test_label_csv, header=None, index=False)

    print(f"Training set saved to {train_csv}, Labels saved to {train_label_csv}")
    print(f"Validation set saved to {val_csv}, Labels saved to {val_label_csv}")
    print(f"Testing set saved to {test_csv}, Labels saved to {test_label_csv}")


# PATH OF ORIGINAT DATA SET 
data_path = r"C:\Users\HP\OneDrive - student.birzeit.edu\Desktop\Y4\Computer Vision\Project\isolated_words_per_user"
images_csv = "images.csv"
labels_csv = "labels.csv"
train_csv = "training_set.csv"
val_csv = "validation_set.csv"
test_csv = "test_set.csv"
train_label_csv = "LABELOftrain.csv"
val_label_csv = "LABELOfvalidation.csv"
test_label_csv = "LABELOfTest.csv"


#----------===================================================
# Function to check if all required files exist
def files_exist(*file_paths):
    return all(os.path.exists(file) for file in file_paths)

# Step 1: Generate CSV files if not already generated
if not files_exist(images_csv, labels_csv):
    print("Generating image and label CSV files...")
    generate_csv(data_path, images_csv, labels_csv, image_size=(64, 64))
else:
    print("Image and label CSV files already exist. Skipping generation.")

# Step 2: Split data if not already split
if not files_exist(train_csv, val_csv, test_csv, train_label_csv, val_label_csv, test_label_csv):
    print("Splitting data into training, validation, and testing sets...")    
    split_and_save_data_with_labels(images_csv, labels_csv, train_csv, val_csv, test_csv, train_label_csv, val_label_csv, test_label_csv)
else:
    print("Training, validation, and testing datasets already exist. Skipping splitting.")

# Step 3: Load the preprocessed data
print("Loading datasets333...")
X_train = pd.read_csv(train_csv, header=None).values
X_val = pd.read_csv(val_csv, header=None).values
X_test = pd.read_csv(test_csv, header=None).values
y_train = pd.read_csv(train_label_csv, header=None).values.flatten()
y_val = pd.read_csv(val_label_csv, header=None).values.flatten()
y_test = pd.read_csv(test_label_csv, header=None).values.flatten()

# Reshape images
X_train = X_train.reshape(-1, 64, 64, 1) / 255.0
X_val = X_val.reshape(-1, 64, 64, 1) / 255.0
X_test = X_test.reshape(-1, 64, 64, 1) / 255.0

# Encode labels into one-hot vectors
num_classes = len(np.unique(y_train))
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print("Data loading and preprocessing completed.")


#! ============================================================================
def build_cnn_model(input_shape, num_classes, layer_configs):
    model = Sequential()
    for i, config in enumerate(layer_configs):
        model.add(
            Conv2D(
                filters=config["filters"],
                kernel_size=config["kernel_size"],
                strides=config.get("stride", (1, 1)),  # Default stride is (1, 1)
                padding=config.get("padding", "same"),  # Default padding is "same"
                activation="relu",
                input_shape=input_shape if i == 0 else None,
            )
        )
        if config.get("pooling", False):
            model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))  # Fixed dense neurons at 128
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def grid_search_cnn(X_train, y_train, X_val, y_val, input_shape, num_classes, layer_configs):
    
    best_model = None
    best_val_accuracy = 0

    print(f"Training with layer configurations: {layer_configs}")
    model = build_cnn_model(input_shape, num_classes, layer_configs)

    # EarlyStopping callback to stop training when validation loss stops improving
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # Train the model and capture the history
    history = model.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=64,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[early_stopping]
    )

    # Plot the training history
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Update the best model if validation accuracy improves
    val_accuracy = max(history.history["val_accuracy"])
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model = model

    print(f"Best Validation Accuracy: {best_val_accuracy}")
    return best_model


# Define layer configurations for each network
layer_configs_1 = [
    {"filters": 64, "kernel_size": (5, 5), "stride": (1, 1), "padding": "same", "pooling": True},  # Layer 1
]

layer_configs_2 = [
    {"filters": 16, "kernel_size": (3, 3), "stride": (1, 1), "padding": "same", "pooling": True},  # Layer 1
    {"filters": 32, "kernel_size": (5, 5), "stride": (2, 2), "padding": "same", "pooling": True},  # Layer 2
]

layer_configs_4 = [ 
    {"filters": 32, "kernel_size": (3, 3), "stride": (1, 1), "padding": "same", "pooling": True},  # Layer 1
    {"filters": 64, "kernel_size": (5, 5), "stride": (1, 1), "padding": "same", "pooling": True},  # Layer 2
    {"filters": 128, "kernel_size": (5, 5), "stride": (1, 1), "padding": "same", "pooling": True}, # Layer 3
    {"filters": 256, "kernel_size": (5, 5), "stride": (1, 1), "padding": "same", "pooling": True}, # Layer 4
]

layer_configs_5 = [
     {"filters": 32, "kernel_size": (3, 3), "stride": (1, 1), "padding": "same", "pooling": True},  # Layer 1
    {"filters": 64, "kernel_size": (5, 5), "stride": (1, 1), "padding": "same", "pooling": True},  # Layer 2
    {"filters": 128, "kernel_size": (7, 7), "stride": (2, 2), "padding": "same", "pooling": True}, # Layer 3
    {"filters": 256, "kernel_size": (7, 7), "stride": (1, 1), "padding": "same", "pooling": True}, # Layer 4
    {"filters": 32, "kernel_size": (7, 7), "stride": (1, 1), "padding": "same", "pooling": False},# Layer 5
]


layer_configs_7 = [
    {"filters": 32, "kernel_size": (3, 3), "stride": (1, 1), "padding": "same", "pooling": True},  # Layer 1
    {"filters": 64, "kernel_size": (5, 5), "stride": (1, 1), "padding": "same", "pooling": True},  # Layer 2
    {"filters": 128, "kernel_size": (3, 3), "stride": (1, 1), "padding": "same", "pooling": True}, # Layer 3
    {"filters": 256, "kernel_size": (5, 5), "stride": (1, 1), "padding": "same", "pooling": True}, # Layer 4
    {"filters": 256, "kernel_size": (3, 3), "stride": (1, 1), "padding": "same", "pooling": False},# Layer 5
    {"filters": 128, "kernel_size": (5, 5), "stride": (1, 1), "padding": "same", "pooling": False}, # Layer 6
    {"filters": 128, "kernel_size": (5, 5), "stride": (1, 1), "padding": "same", "pooling": False},# Layer 7
]


# Perform grid search for each network
input_shape = (64, 64, 1)
num_classes = 82

#!================================================= Data Augmentation ======================================================

# Step 2: Duplicate the training data
X_train_dup = np.concatenate([X_train] * 2, axis=0)  # Duplicate image data
y_train_dup = np.concatenate([y_train] * 2, axis=0)  # Duplicate labels

# Reshape X_train_dup for grayscale images
X_train_dup = X_train_dup.reshape(-1, 64, 64, 1)  # For grayscale images

# Apply data augmentation to the duplicated data
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

augmented_images = []
for img in X_train_dup:
    augmented_images.append(datagen.random_transform(img))
X_train_aug = np.array(augmented_images)

# Concatenate augmented and original data
X_train_final = np.concatenate([X_train_dup, X_train_aug], axis=0)
y_train_final = np.concatenate([y_train_dup, y_train_dup], axis=0)  # Ensure duplication matches

# Check shapes
print(f"Shape of X_train_final: {X_train_final.shape}")  # Should be (19544, 64, 64, 1)
print(f"Shape of y_train_final: {y_train_final.shape}")  # Should match X_train_final in samples

# Use sparse categorical crossentropy for segmentation (no need for to_categorical)
grid_search_cnn(X_train_final, y_train_final, X_val, y_val, input_shape, num_classes, layer_configs_4)

#!================
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Grid Search for 1-Layer CNN:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
grid_search_cnn(X_train, y_train, X_val, y_val, input_shape, num_classes, layer_configs_1)
print("="*92)

print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Grid Search for 2-Layer CNN:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
grid_search_cnn(X_train, y_train, X_val, y_val, input_shape, num_classes, layer_configs_2)
print("="*92)

print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Grid Search for 4-Layer CNN:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
grid_search_cnn(X_train, y_train, X_val, y_val, input_shape, num_classes, layer_configs_4)
print("="*92)

print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Grid Search for 5-Layer CNN:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
grid_search_cnn(X_train, y_train, X_val, y_val, input_shape, num_classes, layer_configs_5)
print("="*92)


print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Grid Search for 7-Layer CNN:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
grid_search_cnn(X_train, y_train, X_val, y_val, input_shape, num_classes, layer_configs_7)
print("="*92)



