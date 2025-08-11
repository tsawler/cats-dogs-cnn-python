"""
Cat vs Dog Image Classifier using PyTorch

This program trains a Convolutional Neural Network (CNN) model to classify images of cats and dogs.
Hyperparameters can be specified via command-line arguments.
"""

import argparse
import warnings
import gc
import sys
import multiprocessing
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
    Encapsulates the training process to ensure DataLoaders are properly cleaned up.
    """
    using_workers = False
    try:
        train_loader, val_loader = load_data(
            args.data_dir, args.image_size, args.batch_size, args.val_split, device, args.augmentation
        )
        using_workers = True

        model = CatDogCNN(args.image_size).to(device)
        print("\nModel Architecture:")
        print(model)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), lr=args.learning_rate,
            momentum=args.momentum, weight_decay=args.weight_decay
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=args.patience
        )

        best_model_state, best_val_accuracy = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            device, args.num_epochs, args.early_stopping,
            args.early_stopping_patience, args.early_stopping_min_delta
        )

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print("Loaded best model state")
        save_model(model, args.model_path, args.onnx_path, args.image_size, device)

        print("All operations completed successfully!")
        
        # Explicitly delete loaders and model to force cleanup
        del train_loader
        del val_loader
        del model
        
        return using_workers

    finally:
        print("Cleaning up resources...")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            gc.collect()
            gc.collect()
        gc.collect()
        
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
        
        # This block is now outside the function and only runs after training is done.
        # This is where the final multiprocessing cleanup can occur.
        if using_workers:
            if hasattr(multiprocessing, '_exit_function'):
                # Call the multiprocessing cleanup function
                multiprocessing._exit_function()
            # For Apple Silicon, use a normal exit which should be sufficient
            # after proper cleanup
            sys.exit(0)


if __name__ == "__main__":
    main()