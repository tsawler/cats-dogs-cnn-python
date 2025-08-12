"""
Cat vs Dog Image Classifier using PyTorch

This program trains a Convolutional Neural Network (CNN) model to classify images of cats and dogs.
Hyperparameters can be specified via command-line arguments.
"""

import argparse
import warnings
import gc
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    DataLoader,
    random_split,
    Subset
)
from torchvision import datasets, transforms
from tqdm import tqdm


def parse_args():
    """
    Parse command line arguments with reasonable defaults.

    This function defines all the parameters that can be controlled from the command line,
    allowing for easy experimentation without modifying the code.
    """
    parser = argparse.ArgumentParser(
        description='Train a cat vs dog classifier')

    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to the dataset directory')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size to resize images (square) - larger values provide more detail but require more memory')
    parser.add_argument('--augmentation', action='store_true',
                        help='Enable data augmentation for training (applies transformations to artificially increase dataset size)')

    # Training hyperparameters - these control the learning process
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training - how many images to process at once')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate - controls how large steps the optimizer takes (higher = faster but potentially unstable)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs - each epoch processes the entire dataset once')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer - helps accelerate in relevant directions and dampen oscillations')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty) - helps prevent overfitting by penalizing large weights')

    # Output parameters
    parser.add_argument('--model_path', type=str, default='cat_dog_classifier.pth',
                        help='Path to save the PyTorch model')
    parser.add_argument('--onnx_path', type=str, default='cat_dog_classifier.onnx',
                        help='Path to save the ONNX model (a format for model interoperability)')

    # Other parameters
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation set split ratio - percentage of data used for validation')
    parser.add_argument('--patience', type=int, default=2,
                        help='Patience for learning rate scheduler - how many epochs to wait before reducing learning rate')

    # Early stopping parameters - helps prevent overfitting by stopping training when performance stops improving
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Number of epochs to wait before stopping training if no improvement')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001,
                        help='Minimum change to qualify as improvement')

    # Inference parameters - for using the model after training
    parser.add_argument('--inference', action='store_true',
                        help='Run inference on a single image instead of training')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to the image file for inference')
    parser.add_argument('--model_file', type=str, default=None,
                        help='Path to the model file (.pth or .onnx) for inference')

    return parser.parse_args()


class EarlyStopping:
    """
    Early stopping mechanism to terminate training when validation loss stops improving.

    This prevents overfitting by monitoring validation loss and stopping training
    when the model starts to memorize the training data instead of learning general patterns.

    Args:
        patience (int): Number of epochs to wait before stopping if no improvement
        min_delta (float): Minimum change to qualify as an improvement
    """

    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience  # How many epochs to wait
        self.min_delta = min_delta  # Minimum change threshold
        self.counter = 0  # Counter for epochs without improvement
        self.best_score = None  # Best score seen so far
        self.early_stop = False  # Flag to indicate if we should stop training

    def __call__(self, val_loss):
        """
        Check if training should be stopped based on validation loss.

        Args:
            val_loss (float): Current validation loss

        Returns:
            bool: True if early stopping should be triggered, False otherwise
        """
        score = -val_loss  # Higher score is better (negative loss)

        if self.best_score is None:
            # First epoch, initialize the best score
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            # Current score isn't better than best score by at least min_delta
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                # We've waited long enough, trigger early stopping
                self.early_stop = True
                print("Early stopping triggered!")
        else:
            # Score improved, reset counter and update best score
            self.best_score = score
            self.counter = 0

        return self.early_stop


import torch
import torch.nn as nn

class CatDogCNN(nn.Module):
    """
    CNN architecture for cat vs dog classification.

    This model uses a series of convolutional layers followed by fully connected layers.
    Each convolutional layer extracts increasingly complex features from the images:
    - First layers detect simple features like edges and textures
    - Middle layers combine these to detect patterns like shapes
    - Later layers detect high-level features like "whiskers" or "ears"

    Args:
        image_size (int): Size of the input images (square)
    """

    def __init__(self, image_size):
        super(CatDogCNN, self).__init__()

        # =============== FIRST CONVOLUTIONAL BLOCK ===============
        # Conv2d: Applies 2D convolution to extract visual features
        # Parameters:
        #   - in_channels=3: RGB color channels (Red, Green, Blue)
        #   - out_channels=32: Number of feature maps/filters to learn
        #   - kernel_size=3: 3x3 sliding window for pattern detection
        #   - padding=1: Add 1 pixel border to keep spatial dimensions same
        # Input shape: [batch, 3, H, W] → Output shape: [batch, 32, H, W]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        # BatchNorm2d: Normalizes feature maps to have mean≈0, std≈1
        # Benefits: Faster training, better gradient flow, acts as regularization
        # Parameter: 32 = number of channels to normalize (matches conv1 output)
        # Shape unchanged: [batch, 32, H, W] → [batch, 32, H, W]
        self.bn1 = nn.BatchNorm2d(32)
        
        # MaxPool2d: Downsampling by taking maximum value in each 2x2 region
        # Parameters:
        #   - kernel_size=2: Size of pooling window (2x2)
        #   - stride=2: Step size (non-overlapping windows)
        # Effect: Reduces spatial dimensions by half
        # Shape change: [batch, 32, H, W] → [batch, 32, H/2, W/2]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # =============== SECOND CONVOLUTIONAL BLOCK ===============
        # More feature maps (64) to detect more complex patterns
        # Input: 32 feature maps from previous block
        # Output: 64 feature maps with same spatial size (due to padding=1)
        # Shape: [batch, 32, H, W] → [batch, 64, H, W]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Batch normalization for 64 channels
        # Shape unchanged: [batch, 64, H, W] → [batch, 64, H, W]
        self.bn2 = nn.BatchNorm2d(64)
        
        # Second pooling layer - another 2x reduction in spatial size
        # Shape: [batch, 64, H, W] → [batch, 64, H/2, W/2]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # =============== THIRD CONVOLUTIONAL BLOCK ===============
        # Even more feature maps (128) for detecting higher-level patterns
        # Input: 64 feature maps, Output: 128 feature maps
        # Shape: [batch, 64, H, W] → [batch, 128, H, W]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization for 128 channels
        # Shape unchanged: [batch, 128, H, W] → [batch, 128, H, W]
        self.bn3 = nn.BatchNorm2d(128)
        
        # Third pooling layer - spatial size reduced by 2 again
        # Shape: [batch, 128, H, W] → [batch, 128, H/2, W/2]
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # =============== FOURTH CONVOLUTIONAL BLOCK ===============
        # Final feature extraction with maximum feature maps (256)
        # Input: 128 feature maps, Output: 256 feature maps
        # Shape: [batch, 128, H, W] → [batch, 256, H, W]
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization for 256 channels
        # Shape unchanged: [batch, 256, H, W] → [batch, 256, H, W]
        self.bn4 = nn.BatchNorm2d(256)
        
        # Fourth and final pooling layer
        # Shape: [batch, 256, H, W] → [batch, 256, H/2, W/2]
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # =============== DIMENSION CALCULATION FOR FC LAYERS ===============
        # Calculate spatial dimensions after all pooling operations
        # Each pooling reduces dimensions by factor of 2
        # After 4 pooling layers: original_size / (2^4) = original_size / 16
        # Floor division (//) ensures integer result (pixel dimensions must be whole numbers)
        feature_size = image_size // 16
        
        # Calculate total number of features for flattening
        # Final feature map has shape: [batch, 256, feature_size, feature_size]
        # When flattened: [batch, 256 * feature_size * feature_size]
        fc_input_size = 256 * feature_size * feature_size

        # =============== FULLY CONNECTED LAYERS ===============
        # First fully connected layer: feature extraction → classification preparation
        # Takes all spatial features and combines them into 512 learned representations
        # Input shape: [batch, fc_input_size] → Output shape: [batch, 512]
        self.fc1 = nn.Linear(fc_input_size, 512)
        
        # Dropout: Regularization technique to prevent overfitting
        # During training: randomly sets 50% of neurons to zero
        # During inference: does nothing (all neurons active)
        # Forces network to not rely too heavily on any single feature
        # Shape unchanged: [batch, 512] → [batch, 512]
        self.dropout = nn.Dropout(0.5)
        
        # Final classification layer: 512 features → 2 class scores
        # Output represents raw scores (logits) for each class: [cat_score, dog_score]
        # Input shape: [batch, 512] → Output shape: [batch, 2]
        self.fc2 = nn.Linear(512, 2)
        
        # ReLU activation function: Rectified Linear Unit
        # Formula: f(x) = max(0, x)
        # Purpose: Introduces non-linearity, allows network to learn complex patterns
        # Zeros out negative values, keeps positive values unchanged
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network.

        This defines how the input flows through the layers to produce an output.
        Each operation transforms the tensor shape and content progressively.

        Args:
            x (Tensor): Input tensor of shape [batch_size, 3, image_size, image_size]
                       Represents a batch of RGB images

        Returns:
            Tensor: Output tensor of shape [batch_size, 2] 
                   Contains logits for each class (cat=0, dog=1)
        """
        
        # =============== CONVOLUTIONAL FEATURE EXTRACTION ===============
        # Process through 4 convolutional blocks sequentially
        # Each block: Convolution → Batch Norm → ReLU → Max Pooling
        for i in range(1, 5):
            # Dynamically get the layers for current block (conv1, bn1, pool1, etc.)
            # getattr(self, f'conv{i}') is equivalent to self.conv1, self.conv2, etc.
            conv = getattr(self, f'conv{i}')  # Get convolution layer
            bn = getattr(self, f'bn{i}')      # Get batch normalization layer  
            pool = getattr(self, f'pool{i}')  # Get max pooling layer

            # Step 1: Apply 2D convolution
            # Slides filters across input to detect patterns/features
            # Shape changes: [batch, in_channels, H, W] → [batch, out_channels, H, W]
            x = conv(x)
            
            # Step 2: Apply batch normalization
            # Normalizes feature maps: subtracts mean, divides by std, applies learned scale/shift
            # Improves training stability and convergence speed
            # Shape unchanged: [batch, channels, H, W] → [batch, channels, H, W]
            x = bn(x)
            
            # Step 3: Apply ReLU activation
            # Element-wise operation: replaces negative values with 0
            # Introduces non-linearity essential for learning complex patterns
            # Shape unchanged: [batch, channels, H, W] → [batch, channels, H, W]
            x = self.relu(x)
            
            # Step 4: Apply max pooling
            # Downsamples by taking maximum value in each 2x2 region
            # Reduces computational load and provides translation invariance
            # Shape changes: [batch, channels, H, W] → [batch, channels, H/2, W/2]
            x = pool(x)

        # After all 4 blocks, tensor shape is approximately:
        # [batch, 256, image_size/16, image_size/16]

        # =============== PREPARE FOR CLASSIFICATION ===============
        # Flatten 4D feature maps into 2D for fully connected layers
        # x.size(0) gets batch dimension (preserve batch size)
        # -1 means "calculate this dimension automatically" 
        # Transforms: [batch, 256, H, W] → [batch, 256*H*W]
        # This converts spatial feature maps into a single feature vector per image
        x = x.view(x.size(0), -1)

        # =============== CLASSIFICATION LAYERS ===============
        # First fully connected layer
        # Linear transformation: x_out = x_in @ weight^T + bias
        # Combines all spatial features into 512 higher-level representations
        # Shape: [batch, fc_input_size] → [batch, 512]
        x = self.fc1(x)
        
        # Apply ReLU activation after first FC layer
        # Introduces non-linearity for the classification stage
        # Shape unchanged: [batch, 512] → [batch, 512]
        x = self.relu(x)
        
        # Apply dropout for regularization
        # Training mode: randomly zeros 50% of features to prevent overfitting
        # Evaluation mode: uses all features (dropout disabled)
        # Shape unchanged: [batch, 512] → [batch, 512]
        x = self.dropout(x)
        
        # Final classification layer
        # Maps 512 features to 2 class scores (cat vs dog)
        # No activation applied - returns raw logits for loss computation
        # Shape: [batch, 512] → [batch, 2]
        x = self.fc2(x)
        
        # Return final classification scores
        # These logits will be used with CrossEntropyLoss during training
        # Or passed through softmax for probability prediction during inference
        return x


def setup_device():
    """
    Set up and return the device for computation (GPU or CPU).

    This function checks for available hardware acceleration options:
    1. NVIDIA GPU (CUDA)
    2. Apple GPU (MPS)
    3. CPU (fallback)

    Using a GPU can significantly speed up neural network training.

    Returns:
        torch.device: Device to use for computation
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA).")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple GPU (MPS).")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device


def load_data(data_dir, image_size, batch_size, val_split, device, augmentation):
    """
    Load and prepare the dataset for training and validation with optional data augmentation.

    Args:
        data_dir (str): Directory containing the dataset
        image_size (int): Size to resize images to
        batch_size (int): Number of images to process at once
        val_split (float): Fraction of data to use for validation
        device (torch.device): Device to use for computation
        augmentation (bool): Whether to enable data augmentation for training

    Returns:
        tuple: (train_loader, val_loader) DataLoader objects for training and validation
    """
    warnings.filterwarnings("ignore", message="Truncated File Read",
                            category=UserWarning, module="PIL.TiffImagePlugin")

    use_pin_memory = (device.type != 'mps')

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if augmentation:
        print("Using data augmentation during training")
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        print("No data augmentation")
        train_transform = val_transform

    try:
        full_dataset = datasets.ImageFolder(root=data_dir)
        print(f"Found {len(full_dataset)} images in total.")
        print(f"Classes: {full_dataset.classes}")

        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        generator = torch.Generator().manual_seed(42)
        train_indices, val_indices = random_split(
            range(len(full_dataset)), [train_size, val_size], generator=generator
        )

        train_subset = Subset(full_dataset, train_indices.indices)
        train_subset.dataset.transform = train_transform

        val_subset = Subset(full_dataset, val_indices.indices)
        val_subset.dataset.transform = val_transform

        print(f"Training set: {len(train_subset)} images")
        print(f"Validation set: {len(val_subset)} images")

    except Exception as e:
        print(f"Error loading dataset from {data_dir}: {e}")
        exit(1)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=use_pin_memory
    )

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs, early_stopping_enabled=False, early_stopping_patience=3,
                early_stopping_min_delta=0.001):
    """
    Train the model and validate it after each epoch.

    This function:
    1. Trains the model on the training data
    2. Validates the model on the validation data
    3. Adjusts the learning rate if needed
    4. Implements early stopping if enabled
    5. Tracks and returns the best model state

    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss)
        optimizer (optim.Optimizer): Optimization algorithm (e.g., SGD)
        scheduler (optim.lr_scheduler): Learning rate scheduler
        device (torch.device): Device to use for computation
        num_epochs (int): Maximum number of training epochs
        early_stopping_enabled (bool): Whether to use early stopping
        early_stopping_patience (int): Number of epochs to wait before stopping
        early_stopping_min_delta (float): Minimum change to qualify as improvement

    Returns:
        tuple: (best_model_state, best_val_accuracy) Best model state and its validation accuracy
    """
    print("\nStarting training...")
    train_losses = []
    val_accuracies = []

    best_val_accuracy = 0.0
    best_model_state = None

    early_stopper = None
    if early_stopping_enabled:
        early_stopper = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta
        )
        print(
            f"Early stopping enabled with patience={early_stopping_patience}")

    for epoch in range(num_epochs):
        # ===== Training phase =====
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)")

        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loader_tqdm.set_postfix(
                loss=running_loss/(train_loader_tqdm.n+1))

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ===== Validation phase =====
        model.eval()
        correct_predictions = 0
        total_samples = 0
        val_running_loss = 0.0

        with torch.no_grad():
            val_loader_tqdm = tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)")
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                val_loader_tqdm.set_postfix(
                    accuracy=f"{100 * correct_predictions / total_samples:.2f}%")

        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * correct_predictions / total_samples
        val_accuracies.append(val_accuracy)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            print(f"New best model: {best_val_accuracy:.2f}% accuracy")

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.2f}%")

        if early_stopping_enabled and early_stopper(avg_val_loss):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print(
        f"\nTraining finished! Best validation accuracy: {best_val_accuracy:.2f}%")
    return best_model_state, best_val_accuracy


def save_model(model, model_path, onnx_path, image_size, device):
    """
    Save the trained model in PyTorch and ONNX formats.

    PyTorch format (.pth) is used within PyTorch applications.
    ONNX format (.onnx) allows the model to be used by different frameworks
    and deployment platforms.

    Args:
        model (nn.Module): The trained model to save
        model_path (str): Path to save the PyTorch model
        onnx_path (str): Path to save the ONNX model
        image_size (int): Size of the input images
        device (torch.device): Device used for computation
    """
    torch.save(model.state_dict(), model_path)
    print(f"PyTorch model saved to {model_path}")

    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                              'output': {0: 'batch_size'}}
            )
        print(f"Model exported to ONNX format at {onnx_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")


def run_inference(image_path, model_file, image_size, device):
    """
    Run inference on a single image using a trained model.

    This function can use either a PyTorch (.pth) or ONNX (.onnx) model
    to classify a single image as cat or dog.

    Args:
        image_path (str): Path to the image file
        model_file (str): Path to the model file
        image_size (int): Size to resize the image to
        device (torch.device): Device to use for computation

    Returns:
        tuple: (prediction, confidence) Class prediction and confidence score
    """
    import numpy as np
    from PIL import Image

    if not Path(image_path).exists() or not Path(model_file).exists():
        print("Error: Image or model file not found")
        return None, None

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

    if Path(model_file).suffix == '.pth':
        model = CatDogCNN(image_size).to(device)
        try:
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.eval()

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                pred_class = ['cat', 'dog'][predicted.item()]
                conf_score = confidence.item()
        except Exception as e:
            print(f"Error with PyTorch model: {e}")
            return None, None

    elif Path(model_file).suffix == '.onnx':
        try:
            import onnxruntime as ort
            ort_session = ort.InferenceSession(model_file)
            ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.cpu().numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)
            outputs = ort_outputs[0]

            def softmax(x):
                exp_x = np.exp(x - np.max(x))
                return exp_x / exp_x.sum()

            probabilities = softmax(outputs[0])
            predicted = np.argmax(probabilities)
            pred_class = ['cat', 'dog'][predicted]
            conf_score = float(probabilities[predicted])
        except Exception as e:
            print(f"Error with ONNX model: {e}")
            return None, None
    else:
        print("Error: Unsupported model format")
        return None, None

    print(
        f"\nInference Results:\nImage: {image_path}\nPrediction: {pred_class}\nConfidence: {conf_score:.2%}")
    return pred_class, conf_score


def run_training_and_cleanup(args, device):
    """
    Orchestrates the complete machine learning training pipeline while ensuring proper resource management.
    
    This function wraps the entire training process in a try-finally block to guarantee that
    system resources (especially GPU memory and multiprocessing workers) are properly released,
    preventing memory leaks and zombie processes that could accumulate over multiple training runs.
    
    MACHINE LEARNING WORKFLOW OVERVIEW:
    The function implements a typical ML training pipeline:
    1. Data Loading & Preprocessing
    2. Model Architecture Definition
    3. Training Configuration (loss, optimizer, scheduler)
    4. Training Loop Execution
    5. Model Persistence (saving)
    6. Resource Cleanup
    
    WHY THIS FUNCTION EXISTS:
    Machine learning training involves several resources that can cause problems if not properly managed:
    - GPU memory can accumulate unused tensors (memory leaks)
    - DataLoader worker processes can become "zombies" if not properly terminated
    - Large models and datasets consume significant RAM
    - Multi-GPU setups require careful synchronization
    
    This function ensures all these resources are properly cleaned up regardless of whether
    training succeeds, fails, or is interrupted.
    
    Args:
        args: Parsed command-line arguments containing all training configuration
              (learning rate, batch size, number of epochs, data paths, etc.)
        device (torch.device): Computational device (CPU, CUDA GPU, or Apple MPS)
                              determined by setup_device() function
    
    Returns:
        bool: True if DataLoader workers were used (needed for cleanup decision in main()),
              False otherwise
    """
    
    # ===== INITIALIZATION AND SAFETY FLAG =====
    # Track whether we're using multiprocessing workers in DataLoaders
    # This information is crucial for deciding cleanup strategy later
    # Workers (separate processes) require more aggressive cleanup than single-threaded loading
    using_workers = False
    
    try:
        # ===== STEP 1: DATA LOADING AND PREPROCESSING =====
        # This is often the most memory-intensive part of the pipeline
        # Creates two DataLoader objects that handle:
        # - Loading images from disk in batches
        # - Applying transformations (resize, normalize, augmentation)
        # - Shuffling training data while keeping validation data in order
        # - Managing worker processes for parallel data loading
        print("Loading and preprocessing dataset...")
        train_loader, val_loader = load_data(
            args.data_dir,        # Directory containing image folders (e.g., /data/cats/, /data/dogs/)
            args.image_size,      # Target size to resize all images (e.g., 256x256 pixels)
            args.batch_size,      # How many images to process simultaneously (affects memory usage)
            args.val_split,       # What fraction of data to use for validation (e.g., 0.2 = 20%)
            device,               # Where to store the data (GPU memory vs RAM)
            args.augmentation     # Whether to apply random transformations to increase dataset variety
        )
        
        # Mark that we're using worker processes for data loading
        # Workers are separate Python processes that load data in parallel
        # This significantly speeds up training but requires careful cleanup
        using_workers = True
        print("✓ Data loading completed successfully")

        # ===== STEP 2: MODEL ARCHITECTURE INSTANTIATION =====
        # Create the neural network model and move it to the appropriate device
        # The model consists of:
        # - Convolutional layers for feature extraction (detecting edges, textures, shapes)
        # - Pooling layers for dimensionality reduction
        # - Fully connected layers for classification decisions
        print("Initializing neural network model...")
        model = CatDogCNN(args.image_size).to(device)
        
        # Display the model architecture for debugging and verification
        # This helps ensure the model has the expected structure and parameter count
        print("\nModel Architecture:")
        print(model)
        
        # Calculate and display total number of trainable parameters
        # This gives insight into model complexity and memory requirements
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")

        # ===== STEP 3: TRAINING CONFIGURATION SETUP =====
        # Configure the three essential components for training:
        
        # LOSS FUNCTION (Criterion):
        # CrossEntropyLoss is standard for multi-class classification
        # It combines softmax activation with negative log-likelihood loss
        # Mathematically: loss = -log(softmax(predicted_class_score))
        # Lower loss = better predictions
        print("Setting up training components...")
        criterion = nn.CrossEntropyLoss()
        
        # OPTIMIZER:
        # SGD (Stochastic Gradient Descent) updates model weights to minimize loss
        # Key parameters:
        # - lr (learning rate): How big steps to take when updating weights
        # - momentum: Helps accelerate convergence and avoid local minima  
        # - weight_decay: L2 regularization to prevent overfitting
        optimizer = optim.SGD(
            model.parameters(),           # All model weights and biases to optimize
            lr=args.learning_rate,        # Step size for weight updates
            momentum=args.momentum,       # Momentum factor for smoother convergence
            weight_decay=args.weight_decay # Regularization strength
        )
        
        # LEARNING RATE SCHEDULER:
        # Automatically reduces learning rate when validation loss stops improving
        # This helps fine-tune the model in later stages of training
        # 'min' mode: reduce LR when validation loss stops decreasing
        # factor=0.1: multiply LR by 0.1 when reducing
        # patience=args.patience: wait this many epochs before reducing
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',              # Monitor for decreasing validation loss
            factor=0.1,              # Multiply LR by this factor when reducing  
            patience=args.patience   # Epochs to wait before reducing LR
        )
        print("✓ Training configuration completed")

        # ===== STEP 4: TRAINING LOOP EXECUTION =====
        # This is the core machine learning process where the model learns to classify images
        # The training loop alternates between:
        # 1. Forward pass: Feed images through model to get predictions
        # 2. Loss calculation: Compare predictions to true labels
        # 3. Backward pass: Calculate gradients (how to adjust weights)
        # 4. Weight update: Apply gradients to improve model
        # 5. Validation: Test model on unseen data to check generalization
        print("Starting training process...")
        print(f"Training for maximum {args.num_epochs} epochs...")
        if args.early_stopping:
            print(f"Early stopping enabled: will stop if no improvement for {args.early_stopping_patience} epochs")
        
        best_model_state, best_val_accuracy = train_model(
            model,                               # The neural network to train
            train_loader,                        # DataLoader with training images
            val_loader,                          # DataLoader with validation images  
            criterion,                           # Loss function to minimize
            optimizer,                           # Algorithm for updating weights
            scheduler,                           # Learning rate adjustment strategy
            device,                              # Hardware to use (CPU/GPU)
            args.num_epochs,                     # Maximum number of training cycles
            args.early_stopping,                 # Whether to stop early if no improvement
            args.early_stopping_patience,        # How many epochs to wait before stopping
            args.early_stopping_min_delta       # Minimum improvement threshold
        )
        print("✓ Training process completed")

        # ===== STEP 5: BEST MODEL RESTORATION =====
        # During training, we save the model state with the best validation accuracy
        # This prevents overfitting by using the model that generalizes best to unseen data
        # rather than the model from the final epoch (which might be overfitted)
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"✓ Restored best model state (validation accuracy: {best_val_accuracy:.2f}%)")
        else:
            print("⚠ Warning: No best model state saved, using final epoch model")
        
        # ===== STEP 6: MODEL PERSISTENCE =====
        # Save the trained model in two formats:
        # 1. PyTorch (.pth): Native format, includes full model state
        # 2. ONNX (.onnx): Universal format for deployment across different frameworks
        print("Saving trained model...")
        save_model(
            model,              # The trained neural network
            args.model_path,    # Path for PyTorch format file
            args.onnx_path,     # Path for ONNX format file  
            args.image_size,    # Image dimensions (needed for ONNX export)
            device              # Device used for computation
        )
        print("✓ Model saved successfully")

        print("\n" + "="*50)
        print("🎉 ALL TRAINING OPERATIONS COMPLETED SUCCESSFULLY! 🎉")
        print("="*50)
        
        # ===== IMMEDIATE CLEANUP FOR SMOOTH OPERATION =====
        # Explicitly delete large objects to free memory before the finally block
        # This is especially important when running multiple training sessions
        # or when system memory is limited
        print("Performing immediate cleanup of training objects...")
        del train_loader    # Free DataLoader and associated worker processes
        del val_loader      # Free validation DataLoader  
        del model           # Free model parameters and buffers from memory
        print("✓ Immediate cleanup completed")
        
        # Return success status with worker information
        return using_workers

    except Exception as e:
        # ===== ERROR HANDLING =====
        # If anything goes wrong during training, log the error details
        # The finally block will still execute to clean up resources
        print(f"\n❌ ERROR DURING TRAINING: {e}")
        print("Proceeding with cleanup and resource deallocation...")
        
        # Even on failure, return the worker status for proper cleanup
        return using_workers
        
    finally:
        # ===== GUARANTEED RESOURCE CLEANUP =====
        # This block ALWAYS executes, regardless of success or failure
        # It's crucial for preventing resource leaks and system stability
        
        print("Executing comprehensive resource cleanup...")
        
        # DEVICE-SPECIFIC MEMORY CLEANUP:
        # Different hardware accelerators require different cleanup strategies
        
        if device.type == 'cuda':
            # NVIDIA GPU cleanup
            # CUDA can accumulate "zombie" tensors in GPU memory even after Python
            # objects are deleted. empty_cache() forces immediate GPU memory deallocation
            print("Clearing CUDA GPU memory cache...")
            torch.cuda.empty_cache()
            print("✓ CUDA memory cleared")
            
        elif device.type == 'mps':
            # Apple Silicon GPU cleanup  
            # MPS (Metal Performance Shaders) on Apple Silicon requires more aggressive
            # cleanup. We run garbage collection twice because Apple's implementation
            # sometimes needs multiple passes to fully release GPU memory
            print("Performing Apple MPS GPU memory cleanup...")
            gc.collect()    # First garbage collection pass
            gc.collect()    # Second pass for stubborn MPS memory leaks
            print("✓ MPS memory cleared")
        
        # SYSTEM-WIDE MEMORY CLEANUP:
        # Force Python's garbage collector to run immediately
        # This ensures that all Python objects (especially large NumPy arrays
        # and PyTorch tensors) are properly deallocated from system RAM
        print("Running system garbage collection...")
        gc.collect()
        print("✓ System memory cleanup completed")
        
        print("✓ All resource cleanup operations completed successfully")
    
    # This return statement should never be reached due to the return statements
    # in the try and except blocks, but it's here for completeness
    return False


def main():
    """
    Main function to train the model or run inference.
    
    This function:
    1. Parses command line arguments
    2. Sets up the device (CPU/GPU)
    3. Either runs inference on a single image or trains a new model
    4. Handles cleanup to prevent resource leaks
    """
    args = parse_args()
    device = setup_device()

    if args.inference:
        if not args.image_path:
            print("Error: --image_path is required for inference")
            exit(1)

        if args.model_file:
            model_file = args.model_file
        elif Path(args.model_path).exists():
            model_file = args.model_path
            print(f"Using default PyTorch model: {model_file}")
        elif Path(args.onnx_path).exists():
            model_file = args.onnx_path
            print(f"Using default ONNX model: {model_file}")
        else:
            print("Error: No trained model found")
            exit(1)

        run_inference(args.image_path, model_file, args.image_size, device)
        return
    else:
        # Run training in a separate function to ensure proper resource cleanup
        using_workers = run_training_and_cleanup(args, device)
        
        # If we used worker processes, force a clean exit
        if using_workers:
            print("Forcing clean exit...")
            os._exit(0)  # Immediately terminate the process without cleanup


if __name__ == "__main__":
    main()