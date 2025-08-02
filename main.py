#!/usr/bin/env python3
"""
Cat vs Dog Image Classifier using PyTorch

This program trains a Convolutional Neural Network (CNN) model to classify images of cats and dogs.
Hyperparameters can be specified via command-line arguments.

For someone new to machine learning:
- A CNN is a type of neural network designed specifically for processing image data
- Neural networks learn to recognize patterns through training on labeled examples
- This program will learn to distinguish between cats and dogs based on their visual features
"""

import warnings  # For suppressing unwanted warnings
import argparse  # For parsing command-line arguments 
import os        # For file and directory operations
import torch     # The main deep learning framework we're using
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms for training
from torch.utils.data import DataLoader  # For efficiently loading and batching data
from torchvision import datasets, transforms  # For working with image datasets and transformations
from tqdm import tqdm  # For displaying progress bars during training


def parse_args():
    """
    Parse command line arguments.
    
    This function sets up all the customizable parameters that can be specified when running
    the program. It defines reasonable default values for each parameter, but allows users
    to override them via command line.
    
    Returns:
        argparse.Namespace: An object containing all the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Train a cat vs dog classifier')

    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to the dataset directory')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size to resize images (square)')

    # Training hyperparameters
    # Hyperparameters are the settings that control the learning process
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (number of images processed at once)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (controls how quickly the model adapts to the problem)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs (complete passes through the dataset)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer (helps accelerate in consistent directions)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty) - helps prevent overfitting by penalizing large weights')

    # Output parameters
    parser.add_argument('--model_path', type=str, default='cat_dog_classifier.pth',
                        help='Path to save the PyTorch model')
    parser.add_argument('--onnx_path', type=str, default='cat_dog_classifier.onnx',
                        help='Path to save the ONNX model (a format for cross-platform model exchange)')

    # Other parameters
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation set split ratio (portion of data used for testing during training)')
    parser.add_argument('--patience', type=int, default=2,
                        help='Patience for learning rate scheduler (epochs to wait before reducing learning rate)')

    return parser.parse_args()


class CatDogCNN(nn.Module):
    """
    CNN architecture for cat vs dog classification.
    
    A Convolutional Neural Network (CNN) is specially designed for processing grid-like data
    such as images. Unlike traditional neural networks, CNNs use:
    
    1. Convolutional layers: These scan the image with small filters to detect features
       like edges, textures, and patterns
    2. Pooling layers: These downsample the image to focus on the most important features
       and reduce computational load
    3. Fully connected layers: These take the extracted features and make the final classification
    
    The network architecture dynamically adjusts to the input image size.
    """

    def __init__(self, image_size):
        """
        Initialize the CNN architecture.
        
        Args:
            image_size (int): The width/height of the input images (assuming square images)
        """
        super(CatDogCNN, self).__init__()
        
        # First convolutional block
        # Conv2d: Applies a 2D convolution over the input image
        # Parameters: (input_channels, output_channels, kernel_size, padding)
        # - input_channels=3: RGB images have 3 color channels
        # - output_channels=32: We create 32 different feature detectors
        # - kernel_size=3: Each filter is 3x3 pixels
        # - padding=1: Adds a 1-pixel border to preserve spatial dimensions
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        # BatchNorm2d: Normalizes the outputs to improve training stability
        # This helps training converge faster and be less sensitive to initialization
        self.bn1 = nn.BatchNorm2d(32)
        
        # MaxPool2d: Reduces spatial dimensions by taking maximum value in each region
        # This downsamples the image by a factor of 2 in each dimension
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block (similar structure but with more filters)
        # We increase the number of filters as we go deeper to detect more complex patterns
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the output feature dimensions dynamically based on image_size
        # After 4 pooling layers (each dividing dimensions by 2), the size is reduced by 2^4 = 16
        # For example, if image_size is 256, the feature size will be 256/16 = 16
        feature_size = image_size // 16
        
        # Calculate the total number of features after the last convolutional layer
        # This is: number of channels (256) × height of feature map × width of feature map
        fc_input_size = 256 * feature_size * feature_size

        # Fully connected layers for classification
        # First fully connected layer reduces the flattened features to 512 neurons
        self.fc1 = nn.Linear(fc_input_size, 512)
        
        # Dropout randomly "turns off" neurons during training
        # This helps prevent overfitting by forcing the network to not rely too much on any single neuron
        # The value 0.5 means each neuron has a 50% chance of being turned off during each training step
        self.dropout = nn.Dropout(0.5)
        
        # Final classification layer
        # Maps from 512 features to 2 output classes (cat or dog)
        self.fc2 = nn.Linear(512, 2)  # 2 output classes: cat, dog

        # ReLU activation function
        # This introduces non-linearity, allowing the network to learn complex patterns
        # It works by keeping positive values unchanged and setting negative values to zero
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Define the forward pass through the network.
        
        This method describes how data flows through the network layers during both training 
        and inference. The pattern for each block is:
        convolution → batch normalization → activation → pooling
        
        Args:
            x: Input tensor containing the image batch, shape [batch_size, 3, height, width]
               where 3 represents the RGB channels
        
        Returns:
            Output tensor with class scores, shape [batch_size, 2]
            where 2 is the number of classes (cat, dog)
        """
        # Block 1
        x = self.conv1(x)  # Apply convolution
        x = self.bn1(x)    # Normalize the outputs
        x = self.relu(x)   # Apply non-linearity
        x = self.pool1(x)  # Downsample

        # Block 2 - Same pattern repeats
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)

        # Flatten the output for the fully connected layers
        # Transforms from [batch_size, channels, height, width] to [batch_size, channels*height*width]
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = self.fc1(x)     # First fully connected layer
        x = self.relu(x)    # Apply non-linearity
        x = self.dropout(x) # Apply dropout (active only during training)
        # No activation here, as CrossEntropyLoss includes Softmax
        x = self.fc2(x)     # Final classification layer

        return x


def setup_device():
    """
    Set up and return the device for computation (GPU or CPU).
    
    Deep learning computations can be accelerated using specialized hardware:
    - CUDA: NVIDIA's platform for GPU computing
    - MPS: Apple's Metal Performance Shaders for Mac GPUs
    - CPU: Used as a fallback when no GPU is available
    
    Using a GPU can make training 10-50x faster than CPU alone.
    
    Returns:
        torch.device: The device to use for model training
    """
    if torch.cuda.is_available():
        # Use NVIDIA GPU if available (fastest option for most systems)
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA).")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Use Apple GPU if available (for newer Mac computers with M-series chips)
        device = torch.device("mps")
        print("Using Apple GPU (MPS).")
    else:
        # Fall back to CPU if no GPU is available
        device = torch.device("cpu")
        print("Using CPU.")
    return device


def load_data(data_dir, image_size, batch_size, val_split):
    """
    Load and prepare the dataset for training and validation.
    
    This function:
    1. Sets up image transformations to prepare raw images for the neural network
    2. Loads images from the specified directory
    3. Splits the dataset into training and validation sets
    4. Creates data loaders that efficiently feed batches of data to the model
    
    Args:
        data_dir (str): Directory containing the dataset
        image_size (int): Size to resize images to (square)
        batch_size (int): Number of images to process at once
        val_split (float): Fraction of data to use for validation (0.0 to 1.0)
        
    Returns:
        tuple: (train_loader, val_loader) for training and validation
    """
    # Suppress specific Pillow warnings about image format issues
    warnings.filterwarnings("ignore", message="Truncated File Read",
                            category=UserWarning, module="PIL.TiffImagePlugin")

    # Define transformations for the images
    # These operations are applied to each image before it's used for training
    transform = transforms.Compose([
        # Resize images to the specified dimensions
        # This ensures all images have the same size, regardless of their original dimensions
        transforms.Resize((image_size, image_size)),
        
        # Convert PIL images to PyTorch tensors
        # Tensors are multi-dimensional arrays optimized for neural network operations
        transforms.ToTensor(),
        
        # Normalize pixel values using ImageNet mean and std
        # This centers the data around zero and gives it unit variance, which helps training
        # The values [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225] are standard for 
        # models pretrained on ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset
    # ImageFolder expects a directory structure like:
    # data_dir/
    #   ├── cat/  (all cat images)
    #   └── dog/  (all dog images)
    try:
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        print(f"Found {len(dataset)} images in total.")
        print(f"Classes: {dataset.classes}")
        print(f"Class to index mapping: {dataset.class_to_idx}")
    except Exception as e:
        print(
            f"Error loading dataset from {data_dir}. Please ensure the directory exists and contains properly organized images.")
        print(f"Error details: {e}")
        exit(1)

    # Split the dataset into training and validation sets
    # Training set: Used to update the model weights
    # Validation set: Used to evaluate model performance during training
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    # Create DataLoaders
    # DataLoaders handle:
    # - Batching: Processing multiple images at once for efficiency
    # - Shuffling: Randomizing the order of images (important for training)
    # - Parallelization: Loading data in background threads
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs):
    """
    Train the model and validate it after each epoch.
    
    The training process involves:
    1. Forward pass: Running images through the network to get predictions
    2. Loss calculation: Measuring how wrong the predictions are
    3. Backward pass: Computing gradients to determine how to update weights
    4. Weight update: Adjusting the model parameters to reduce error
    5. Validation: Checking performance on unseen data
    
    Args:
        model: The neural network model
        train_loader: DataLoader for the training set
        val_loader: DataLoader for the validation set
        criterion: Loss function to measure prediction error
        optimizer: Algorithm to update model weights
        scheduler: Learning rate scheduler to adjust learning rate during training
        device: Device to run the training on (CPU or GPU)
        num_epochs: Number of complete passes through the training dataset
        
    Returns:
        tuple: (best_model_state, best_val_accuracy) - The weights of the best model and its accuracy
    """
    print("\nStarting training...")
    # Lists to track progress
    train_losses = []       # Track training loss over epochs
    val_accuracies = []     # Track validation accuracy over epochs

    # Variables to store the best model
    best_val_accuracy = 0.0
    best_model_state = None

    # Loop through each epoch (complete pass through the dataset)
    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()  # Set model to training mode (enables dropout, batch norm updates)
        running_loss = 0.0
        
        # Create progress bar for training
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)",
                                 unit="batch")

        # Process each batch of training data
        for inputs, labels in train_loader_tqdm:
            # Move data to the appropriate device (CPU or GPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            # This is important because PyTorch accumulates gradients by default
            optimizer.zero_grad()

            # Forward pass: compute predictions
            outputs = model(inputs)
            
            # Calculate loss (how far off predictions are from actual labels)
            loss = criterion(outputs, labels)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # Update model weights based on computed gradients
            optimizer.step()

            # Accumulate loss for reporting
            running_loss += loss.item()
            
            # Update progress bar with current loss
            train_loader_tqdm.set_postfix(
                loss=running_loss/(train_loader_tqdm.n+1))

        # Calculate average loss for this epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- VALIDATION PHASE ---
        # Evaluation mode disables dropout and uses fixed batch norm statistics
        model.eval()
        correct_predictions = 0
        total_samples = 0
        val_running_loss = 0.0

        # No need to track gradients during validation (saves memory and computation)
        with torch.no_grad():
            # Create progress bar for validation
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)",
                                   unit="batch")

            # Process each batch of validation data
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass only (no backward pass or optimization during validation)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                # Get predictions by taking the class with highest score
                _, predicted = torch.max(outputs.data, 1)
                
                # Count total samples and correct predictions
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # Update progress bar with current accuracy
                val_loader_tqdm.set_postfix(
                    accuracy=f"{100 * correct_predictions / total_samples:.2f}%")

        # Calculate validation metrics
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * correct_predictions / total_samples
        val_accuracies.append(val_accuracy)

        # --- LEARNING RATE ADJUSTMENT ---
        # Update the learning rate based on validation loss
        # This reduces the learning rate when the model stops improving
        scheduler.step(avg_val_loss)

        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

        # --- SAVE BEST MODEL ---
        # If this is the best model so far (based on validation accuracy), save it
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            print(
                f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%")

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Print final training summary
    print("\nTraining Finished!")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")

    return best_model_state, best_val_accuracy


def save_model(model, model_path, onnx_path, image_size, device):
    """
    Save the trained model in PyTorch and ONNX formats.
    
    Saving the model allows it to be reused later without retraining:
    - PyTorch format (.pth): Used within PyTorch applications
    - ONNX format (.onnx): Cross-platform format for deployment in various environments
    
    Args:
        model: The trained neural network model
        model_path (str): Path to save the PyTorch model
        onnx_path (str): Path to save the ONNX model
        image_size (int): Size of the input images
        device: Device where the model is loaded (CPU or GPU)
    """
    # Save PyTorch model
    # This saves the model's learned parameters (weights and biases)
    torch.save(model.state_dict(), model_path)
    print(f"PyTorch model saved to {model_path}")

    # Export to ONNX format
    # CRITICAL: Set model to evaluation mode for ONNX export
    # This ensures dropout and batch normalization behave correctly during inference
    model.eval()
    
    # Create a dummy input tensor with the same shape as real inputs
    # This is needed because ONNX traces the model by running inference on this input
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    try:
        # Disable gradient tracking for ONNX export
        # This saves memory and speeds up the export process
        with torch.no_grad():  # Disable gradient computation for export
            torch.onnx.export(
                model,                # The model to export
                dummy_input,          # A dummy input for tracing
                onnx_path,            # Output file path
                export_params=True,   # Export the trained parameter weights
                opset_version=11,     # ONNX version to target
                do_constant_folding=True,  # Optimize the model by folding constants
                input_names=['input'],     # Name for the model's input
                output_names=['output'],   # Name for the model's output
                dynamic_axes={
                    # Define which dimensions can change at runtime
                    # The batch size (dimension 0) can vary during inference
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
        print(f"Model exported to ONNX format at {onnx_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        print("Please ensure you have the 'onnx' package installed (`pip install onnx`).")


def main():
    """
    Main function to train and save the model.
    
    This orchestrates the entire training pipeline:
    1. Parse command line arguments
    2. Set up computing device
    3. Load and prepare the data
    4. Create and train the model
    5. Save the trained model
    """
    # Parse command line arguments
    # This allows users to customize the training process
    args = parse_args()

    # Set up device (CPU or GPU)
    # Using a GPU can make training 10-50x faster
    device = setup_device()

    # Load and prepare data
    # This loads images from disk and prepares them for training
    train_loader, val_loader = load_data(
        args.data_dir,     # Directory containing the dataset
        args.image_size,   # Size to resize images to
        args.batch_size,   # Number of images to process at once
        args.val_split     # Portion of data to use for validation
    )

    # Initialize model
    # This creates a new neural network with the specified architecture
    model = CatDogCNN(args.image_size).to(device)
    print("\nModel Architecture:")
    print(model)

    # Define loss function and optimizer
    # Loss function: Measures how wrong the model's predictions are
    # Optimizer: Updates the model's parameters to minimize the loss
    criterion = nn.CrossEntropyLoss()  # Standard loss function for classification
    optimizer = optim.SGD(
        model.parameters(),       # Parameters to optimize (model weights)
        lr=args.learning_rate,    # Learning rate controls step size
        momentum=args.momentum,   # Momentum helps accelerate in consistent directions
        weight_decay=args.weight_decay  # Weight decay helps prevent overfitting
    )

    # Learning rate scheduler
    # This reduces the learning rate when the model stops improving
    # It helps the model converge to a better solution
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,          # The optimizer to adjust
        mode='min',         # Monitor loss for improvement
        factor=0.1,         # Multiply learning rate by this factor when reducing
        patience=args.patience  # How many epochs to wait before reducing
    )

    # Train the model
    # This is the most time-consuming part where the model learns from the data
    best_model_state, best_val_accuracy = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        args.num_epochs
    )

    # Load the best model state
    # We keep track of the best model during training and load it back
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded the best model state based on validation accuracy")

    # Save the model
    # This saves the trained model so it can be used later for predictions
    save_model(model, args.model_path, args.onnx_path, args.image_size, device)


# Python's way of identifying the entry point of the program
# This code only executes when the script is run directly, not when imported
if __name__ == "__main__":
    main()