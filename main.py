#!/usr/bin/env python3
"""
Cat vs Dog Image Classifier using PyTorch

This program trains a Convolutional Neural Network (CNN) model to classify images of cats and dogs.
Hyperparameters can be specified via command-line arguments.
"""

import argparse
import os
import warnings

import torch
import torch.nn as nn  # Neural network module with building blocks for ML models
import torch.optim as optim  # Optimization algorithms like SGD, Adam, etc.
from torch.utils.data import DataLoader  # Helps load data in batches for efficient training
from torchvision import datasets, transforms  # Vision-specific utilities and datasets
from tqdm import tqdm  # Progress bar for tracking training


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

        # First convolutional block
        # Conv2d: Applies 2D convolution to extract visual features
        # 3 input channels (RGB), 32 output feature maps, 3x3 kernel
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # BatchNorm2d: Normalizes the outputs of the convolutional layer
        # Helps with faster and more stable training
        self.bn1 = nn.BatchNorm2d(32)
        # MaxPool2d: Reduces spatial dimensions by taking maximum in 2x2 regions
        # This reduces computation and helps with translation invariance
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block - more feature maps for more complex patterns
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional block - even more feature maps
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fourth convolutional block - final feature extraction
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate dimensions for fully connected layer
        # After 4 pooling layers (each reducing dimensions by half), the size is divided by 16
        feature_size = image_size // 16
        fc_input_size = 256 * feature_size * feature_size

        # Fully connected layers for classification
        # Takes flattened feature maps and outputs class probabilities
        self.fc1 = nn.Linear(fc_input_size, 512)
        # Dropout: Randomly zeroes some elements during training
        # This prevents overfitting by making the network more robust
        self.dropout = nn.Dropout(0.5)
        # Final layer outputs 2 values (one per class: cat, dog)
        self.fc2 = nn.Linear(512, 2)
        # ReLU activation: max(0, x) - introduces non-linearity
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network.
        
        This defines how the input flows through the layers to produce an output.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, 3, image_size, image_size]
            
        Returns:
            Tensor: Output tensor of shape [batch_size, 2] with logits for each class
        """
        # Process through convolutional blocks
        # This loop applies each convolutional block in sequence
        for i in range(1, 5):
            # Get the layers for the current block
            conv = getattr(self, f'conv{i}')
            bn = getattr(self, f'bn{i}')
            pool = getattr(self, f'pool{i}')

            # Apply convolution
            x = conv(x)
            # Apply batch normalization
            x = bn(x)
            # Apply ReLU activation
            x = self.relu(x)
            # Apply max pooling
            x = pool(x)

        # Flatten for fully connected layers
        # Reshape from [batch, channels, height, width] to [batch, channels*height*width]
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
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


def load_data(data_dir, image_size, batch_size, val_split, device):
    """
    Load and prepare the dataset for training and validation with data augmentation.
    
    Data augmentation artificially increases the size of the training dataset by applying
    random transformations (flips, rotations, color changes) to the original images.
    This helps the model generalize better and reduces overfitting.
    
    Args:
        data_dir (str): Directory containing the dataset
        image_size (int): Size to resize images to
        batch_size (int): Number of images to process at once
        val_split (float): Fraction of data to use for validation
        device (torch.device): Device to use for computation
        
    Returns:
        tuple: (train_loader, val_loader) DataLoader objects for training and validation
    """
    # Suppress specific warnings from image processing
    warnings.filterwarnings("ignore", message="Truncated File Read",
                            category=UserWarning, module="PIL.TiffImagePlugin")

    # Determine if pin_memory should be enabled (not for MPS devices)
    # pin_memory speeds up data transfer to GPU but doesn't work well with MPS
    use_pin_memory = (device.type != 'mps')

    # Define transformations for training (with augmentation)
    # These transformations are applied randomly to each image during training
    train_transform = transforms.Compose([
        # Resize all images to the same dimensions
        transforms.Resize((image_size, image_size)),
        # Flip images horizontally with 50% probability
        transforms.RandomHorizontalFlip(p=0.5),
        # Randomly rotate by up to 15 degrees
        transforms.RandomRotation(15),
        # Randomly adjust brightness, contrast, saturation
        # This helps the model be robust to different lighting conditions
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        # Random affine transformations (translation, scaling)
        # This helps the model be robust to different object positions and sizes
        transforms.RandomAffine(
            degrees=0,                        # No additional rotation
            translate=(0.1, 0.1),             # Translate by up to 10% in x and y
            scale=(0.9, 1.1),                 # Scale by 0.9 to 1.1
        ),
        # Convert to PyTorch tensor
        transforms.ToTensor(),
        # Normalize pixel values using ImageNet means and stds
        # This standardizes the input data which helps training
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define transformations for validation (no augmentation)
    # For validation, we only resize, convert to tensor, and normalize
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    try:
        # First load the full dataset with a temporary transform
        dataset = datasets.ImageFolder(root=data_dir, transform=None)
        print(f"Found {len(dataset)} images in total.")
        print(f"Classes: {dataset.classes}")

        # Split dataset indices
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        indices = list(range(len(dataset)))

        # Use fixed random seed for reproducible splits
        generator = torch.Generator().manual_seed(42)
        train_indices, val_indices = torch.utils.data.random_split(
            indices, [train_size, val_size], generator=generator)

        # Create subset datasets with appropriate transforms
        train_dataset = torch.utils.data.Subset(
            datasets.ImageFolder(root=data_dir, transform=train_transform),
            train_indices.indices
        )

        val_dataset = torch.utils.data.Subset(
            datasets.ImageFolder(root=data_dir, transform=val_transform),
            val_indices.indices
        )

        print(f"Training set: {len(train_dataset)} images")
        print(f"Validation set: {len(val_dataset)} images")

    except Exception as e:
        print(f"Error loading dataset from {data_dir}: {e}")
        exit(1)

    # Create DataLoaders
    # DataLoader handles batching, shuffling, and loading data in parallel
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,              # Shuffle training data to prevent learning order biases
        num_workers=4,             # Use multiple workers for faster data loading
        pin_memory=use_pin_memory  # Only use pin_memory for CUDA devices
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,             # No need to shuffle validation data
        num_workers=4,
        pin_memory=use_pin_memory
    )

    return train_loader, val_loader


def load_data_no_augmentation(data_dir, image_size, batch_size, val_split, device):
    """
    Load and prepare the dataset for training and validation without augmentation.
    
    This version is used when augmentation is disabled. It only applies basic
    preprocessing (resize, normalize) without random transformations.
    
    Args:
        data_dir (str): Directory containing the dataset
        image_size (int): Size to resize images to
        batch_size (int): Number of images to process at once
        val_split (float): Fraction of data to use for validation
        device (torch.device): Device to use for computation
        
    Returns:
        tuple: (train_loader, val_loader) DataLoader objects for training and validation
    """
    # Suppress specific warnings
    warnings.filterwarnings("ignore", message="Truncated File Read",
                            category=UserWarning, module="PIL.TiffImagePlugin")

    # Determine if pin_memory should be enabled (not for MPS devices)
    use_pin_memory = (device.type != 'mps')

    # Define standard transformations (no augmentation)
    # Only basic preprocessing is applied to all images
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    try:
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        print(f"Found {len(dataset)} images in total.")
        print(f"Classes: {dataset.classes}")
    except Exception as e:
        print(f"Error loading dataset from {data_dir}: {e}")
        exit(1)

    # Split dataset
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
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
    # Track progress
    train_losses = []
    val_accuracies = []

    # Store best model
    best_val_accuracy = 0.0
    best_model_state = None

    # Initialize early stopping if enabled
    early_stopper = None
    if early_stopping_enabled:
        early_stopper = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta
        )
        print(
            f"Early stopping enabled with patience={early_stopping_patience}")

    # Training loop - iterate through the dataset multiple times
    for epoch in range(num_epochs):
        # ===== Training phase =====
        model.train()  # Set model to training mode (enables dropout, etc.)
        running_loss = 0.0
        # Progress bar for training
        train_loader_tqdm = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)")

        # Iterate through batches
        for inputs, labels in train_loader_tqdm:
            # Move data to the selected device (CPU/GPU)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            # This is necessary because gradients accumulate by default
            optimizer.zero_grad()
            
            # Forward pass: compute predictions
            outputs = model(inputs)
            
            # Compute loss between predictions and true labels
            loss = criterion(outputs, labels)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update model parameters based on gradients
            optimizer.step()
            
            # Track loss
            running_loss += loss.item()
            train_loader_tqdm.set_postfix(
                loss=running_loss/(train_loader_tqdm.n+1))

        # Calculate average training loss for this epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ===== Validation phase =====
        model.eval()  # Set model to evaluation mode (disables dropout, etc.)
        correct_predictions = 0
        total_samples = 0
        val_running_loss = 0.0

        # No gradient computation needed for validation
        with torch.no_grad():
            # Progress bar for validation
            val_loader_tqdm = tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)")
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass only (no backprop during validation)
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
                # Get predictions by taking the max value's index
                _, predicted = torch.max(outputs.data, 1)
                
                # Count correct predictions
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # Update progress bar
                val_loader_tqdm.set_postfix(
                    accuracy=f"{100 * correct_predictions / total_samples:.2f}%")

        # Calculate metrics
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * correct_predictions / total_samples
        val_accuracies.append(val_accuracy)

        # Learning rate adjustment
        # Reduce learning rate when validation loss plateaus
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

        # Save best model
        # Keep track of the model with the highest validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            print(f"New best model: {best_val_accuracy:.2f}% accuracy")

        # Print summary
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.2f}%")

        # Check early stopping
        # Stop training if validation loss hasn't improved for a while
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
    # Save PyTorch model
    torch.save(model.state_dict(), model_path)
    print(f"PyTorch model saved to {model_path}")

    # Export to ONNX format
    model.eval()  # Set to evaluation mode
    # Create a dummy input tensor with the correct shape
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,                              # Model being exported
                dummy_input,                        # Example input
                onnx_path,                          # Output file
                export_params=True,                 # Export model parameters
                opset_version=13,                   # ONNX version
                do_constant_folding=True,           # Optimization: fold constants
                input_names=['input'],              # Names for inputs
                output_names=['output'],            # Names for outputs
                dynamic_axes={'input': {0: 'batch_size'},  # Variable batch size
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

    # Validate files exist
    if not os.path.exists(image_path) or not os.path.exists(model_file):
        print("Error: Image or model file not found")
        return None, None

    # Define transformations - same as validation transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
        input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

    # Process based on model type
    if model_file.endswith('.pth'):
        # PyTorch model
        model = CatDogCNN(image_size).to(device)
        try:
            # Load model weights
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.eval()  # Set to evaluation mode
            
            # Run inference
            with torch.no_grad():  # No gradients needed for inference
                outputs = model(input_tensor)
                # Convert logits to probabilities with softmax
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                # Get the highest probability and its index
                confidence, predicted = torch.max(probabilities, 1)
                # Map index to class name
                pred_class = ['cat', 'dog'][predicted.item()]
                conf_score = confidence.item()
        except Exception as e:
            print(f"Error with PyTorch model: {e}")
            return None, None

    elif model_file.endswith('.onnx'):
        # ONNX model
        try:
            import onnxruntime as ort
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(model_file)
            # Prepare input for ONNX model
            ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.cpu().numpy()}
            # Run inference
            ort_outputs = ort_session.run(None, ort_inputs)
            outputs = ort_outputs[0]

            # Process results
            # Define softmax function (not directly available in numpy)
            def softmax(x):
                exp_x = np.exp(x - np.max(x))
                return exp_x / exp_x.sum()

            # Calculate probabilities and get prediction
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

    # Print results
    print(
        f"\nInference Results:\nImage: {image_path}\nPrediction: {pred_class}\nConfidence: {conf_score:.2%}")
    return pred_class, conf_score


def main():
    """
    Main function to train the model or run inference.
    
    This function:
    1. Parses command line arguments
    2. Sets up the device (CPU/GPU)
    3. Either runs inference on a single image or trains a new model
    4. Handles cleanup to prevent resource leaks
    """
    # Track if we're using DataLoader with workers
    using_workers = False
    args = parse_args()
    device = setup_device()

    # Check if we're in inference mode
    if args.inference:
        if not args.image_path:
            print("Error: --image_path is required for inference")
            exit(1)

        # Find model file
        if args.model_file:
            model_file = args.model_file
        elif os.path.exists(args.model_path):
            model_file = args.model_path
            print(f"Using default PyTorch model: {model_file}")
        elif os.path.exists(args.onnx_path):
            model_file = args.onnx_path
            print(f"Using default ONNX model: {model_file}")
        else:
            print("Error: No trained model found")
            exit(1)

        # Run inference
        run_inference(args.image_path, model_file, args.image_size, device)
        return

    # Training mode
    print("Running in training mode...")

    try:
        # Load data with or without augmentation
        if args.augmentation:
            print("Using data augmentation during training")
            train_loader, val_loader = load_data(
                args.data_dir, args.image_size, args.batch_size, args.val_split, device
            )
        else:
            print("No data augmentation")
            train_loader, val_loader = load_data_no_augmentation(
                args.data_dir, args.image_size, args.batch_size, args.val_split, device
            )

        # We are using workers if num_workers > 0
        using_workers = True

        # Initialize model
        model = CatDogCNN(args.image_size).to(device)
        print("\nModel Architecture:")
        print(model)

        # Define loss function
        # CrossEntropyLoss combines softmax and negative log likelihood loss
        # It's commonly used for classification problems
        criterion = nn.CrossEntropyLoss()
        
        # Define optimizer
        # SGD (Stochastic Gradient Descent) with momentum and weight decay
        # - Momentum helps accelerate gradients in the right direction
        # - Weight decay helps prevent overfitting
        optimizer = optim.SGD(
            model.parameters(), lr=args.learning_rate,
            momentum=args.momentum, weight_decay=args.weight_decay
        )

        # Learning rate scheduler
        # Reduces learning rate when validation loss plateaus
        # This helps fine-tune the model as training progresses
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=args.patience
        )

        # Train the model
        best_model_state, best_val_accuracy = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            device, args.num_epochs, args.early_stopping,
            args.early_stopping_patience, args.early_stopping_min_delta
        )

        # Load best model and save
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print("Loaded best model state")
        save_model(model, args.model_path, args.onnx_path,
                   args.image_size, device)

        print("All operations completed successfully!")

    finally:
        # Clean up resources to prevent hanging
        print("Cleaning up resources...")

        # Move model to CPU before deleting
        # This helps ensure GPU memory is properly freed
        if 'model' in locals():
            model.to('cpu')
            del model

        # Release memory for specific device types
        if device.type == 'cuda':
            # CUDA-specific cleanup
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            # MPS-specific cleanup (Apple Silicon)
            # No direct equivalent to cuda.empty_cache() for MPS yet
            import gc
            # Run garbage collection multiple times to ensure resources are freed
            gc.collect()
            gc.collect()

        # Explicitly close DataLoaders if they were created
        # This ensures background worker processes are terminated
        if 'train_loader' in locals():
            del train_loader
        if 'val_loader' in locals():
            del val_loader

        # Additional garbage collection for all devices
        import gc
        gc.collect()

        print("Exiting program")

        # If we used workers, we need to handle multiprocessing cleanup
        if using_workers:
            # Import multiprocessing to access cleanup methods
            import multiprocessing

            # Clean exit instead of forced exit when possible
            if hasattr(multiprocessing, '_exit_function'):
                # Call the multiprocessing cleanup function
                multiprocessing._exit_function()

            # For Apple Silicon, use a normal exit which should be sufficient
            # after proper cleanup
            import sys
            sys.exit(0)


if __name__ == "__main__":
    main()