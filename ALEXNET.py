# Import necessary libraries
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights

# Configuration
IMAGE_SIZE = (128, 128)  # Resized image dimensions
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################################################################################################
# Dataset Preprocessing
#Iterates through each user's folder in the dataset directory.

#Loads each image as grayscale using OpenCV (cv2.imread).

#Resizes the image to a uniform size (IMAGE_SIZE).

#Normalizes pixel values to the range [0, 1].

#Appends the preprocessed image and its label (user name) to the data and labels lists.
#######################################################################################################

class HandwritingDataset(Dataset):
    """
    Custom PyTorch Dataset for handwriting images.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label
    
    """
        
    Load and preprocess images from the dataset path.
    
    Args:
        dataset_path (str): Path to the dataset directory.
        image_size (tuple): Desired size of the images (height, width).
    
    Returns:
        data (np.array): Array of preprocessed images.
        labels (np.array): Array of corresponding labels.
    """


def preprocess_images(dataset_path):
    """
    Load and preprocess images from the dataset path.
    """
    print("Preprocessing images...")
    data = []
    labels = []

    for user in tqdm(os.listdir(dataset_path), desc="Processing users"):
        user_folder = os.path.join(dataset_path, user)
        if not os.path.isdir(user_folder):
            continue

        for img_file in os.listdir(user_folder):
            img_path = os.path.join(user_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image as grayscale
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)  # Resize to uniform dimensions
                img = img / 255.0  # Normalize pixel values to [0, 1]
                data.append(img)
                labels.append(user)

    print(f"Loaded {len(data)} images from dataset.")
    return np.array(data), np.array(labels)

#######################################################################################################
# Augmentation Transforms
#######################################################################################################

# Define realistic augmentations
#    Parse command-line arguments for customizing training configurations and augmentations.
train_transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert to PIL image for augmentation
    transforms.RandomRotation(degrees=10),  # Small rotation
    transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
    transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.8, 1.0)),  # Random crop and resize
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Adjust brightness, contrast, saturation
    transforms.ToTensor(),  # Convert back to Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Validation and test transforms (no augmentation)
val_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#######################################################################################################
# Model Definition (EfficientNetB0)
"""
    Build an EfficientNetB0-based model for writer detection.

    Args:
        num_classes (int): Number of output classes (writers).
        freeze_backbone (bool): Whether to freeze the pretrained layers.

    Returns:
        model (torch.nn.Module): EfficientNetB0 model with a modified final layer.
"""
#######################################################################################################

def build_efficientnet(num_classes):
    """
    Build an EfficientNetB0-based model for writer detection.
    """
    print("Loading EfficientNetB0 model...")
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)  # Load pretrained weights

    # Replace the final classification layer
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # Update final layer
    return model

#######################################################################################################
# Training and Evaluation
#######################################################################################################

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    print("Starting model training...")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_model = None
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Training phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())

        # Validation phase
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model.state_dict()

    model.load_state_dict(best_model)
    print("Training completed.")
    return model, history


# Function to plot Loss vs Epoch and Accuracy vs Epoch
def plot_training_history(history):
    """
    Plot the training and validation loss and accuracy.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Loss vs. Epochs")

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title("Accuracy vs. Epochs")

    plt.tight_layout()
    plt.show()

#######################################################################################################
# Main Execution
'''
Dataset Loading:

Load and preprocess images using preprocess_images.

Encode labels using LabelEncoder.

Data Splitting:

Split the dataset into training, validation, and test sets using train_test_split.

Data Preparation:

Convert grayscale images to 3-channel format (required for EfficientNet).

Create HandwritingDataset and DataLoader objects.

Model Training:

Build the EfficientNet model using build_efficientnet.

Define loss function (criterion) and optimizer (optimizer).

Train the model using train_model.

Plotting:

Visualize training history using plot_training_history.
'''
#######################################################################################################

def main():
    print("Initializing...")
    # Dataset path
    DATASET_PATH = r"C:\Users\hp\Desktop\computer vesion\vision_#2\isolated_words_per_user"

    # Load and preprocess dataset
    print("Loading dataset...")
    images, labels = preprocess_images(DATASET_PATH)

    # Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    NUM_CLASSES = len(np.unique(labels_encoded))

    # Train-validation-test split
    print("Splitting dataset...")
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42
    )
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42
    )

    # Convert grayscale to 3-channel
    train_images = np.repeat(train_images[..., np.newaxis], 3, axis=-1)
    val_images = np.repeat(val_images[..., np.newaxis], 3, axis=-1)
    test_images = np.repeat(test_images[..., np.newaxis], 3, axis=-1)

    # Create Datasets and DataLoaders
    print("Creating DataLoaders...")
    train_dataset = HandwritingDataset(train_images, train_labels, transform=train_transform)
    val_dataset = HandwritingDataset(val_images, val_labels, transform=val_test_transform)
    test_dataset = HandwritingDataset(test_images, test_labels, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build and train model
 #   /* Hyperparameters
#Image Size: The images are resized to 128x128 pixels to ensure uniformity and reduce computational load.

#Batch Size: The model is trained with a batch size of 32, which balances memory usage and training stability.

#Epochs: The model is trained for 30 epochs, allowing sufficient time for the model to learn from the data without overfitting.

#Learning Rate: A learning rate of 0.001 is used with the Adam optimizer, which is a common choice for deep learning tasks.

#Device: The model is trained on a GPU if available (CUDA), otherwise, it falls back to the CPU.

#Data Preprocessing
#The dataset is preprocessed by resizing images to 128x128 and normalizing pixel values to the range [0, 1].

#Grayscale images are converted to 3-channel images to match the input requirements of EfficientNetB0.

#Data augmentation techniques such as random rotation, horizontal flipping, random cropping, and color jittering are applied to the training set to improve generalization.
    print("Building model...")
    model = build_efficientnet(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Starting training...")
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

    # Plot the training history
    print("Plotting training history...")
    plot_training_history(history)

    # Evaluate model
    print("Evaluating model...")
    model.eval()
    test_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
    test_accuracy = test_corrects.double() / len(test_loader.dataset)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Run the main function
if __name__ == "__main__":
    main()
