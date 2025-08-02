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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def parse_args():
    """Parse command line arguments with reasonable defaults."""
    parser = argparse.ArgumentParser(description='Train a cat vs dog classifier')

    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to the dataset directory')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size to resize images (square)')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')

    # Output parameters
    parser.add_argument('--model_path', type=str, default='cat_dog_classifier.pth',
                        help='Path to save the PyTorch model')
    parser.add_argument('--onnx_path', type=str, default='cat_dog_classifier.onnx',
                        help='Path to save the ONNX model')

    # Other parameters
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation set split ratio')
    parser.add_argument('--patience', type=int, default=2,
                        help='Patience for learning rate scheduler')
    # Early stopping parameters
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Number of epochs to wait before stopping training')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001,
                        help='Minimum change to qualify as improvement')

    # Inference parameters
    parser.add_argument('--inference', action='store_true',
                        help='Run inference on a single image')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to the image file for inference')
    parser.add_argument('--model_file', type=str, default=None,
                        help='Path to the model file (.pth or .onnx)')

    return parser.parse_args()


class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving."""
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        score = -val_loss  # Higher score is better
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered!")
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop


class CatDogCNN(nn.Module):
    """CNN architecture for cat vs dog classification."""
    def __init__(self, image_size):
        super(CatDogCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
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

        # Calculate dimensions for fully connected layer
        feature_size = image_size // 16
        fc_input_size = 256 * feature_size * feature_size

        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)  # 2 output classes: cat, dog
        self.relu = nn.ReLU()

    def forward(self, x):
        # Process through convolutional blocks
        for i in range(1, 5):
            conv = getattr(self, f'conv{i}')
            bn = getattr(self, f'bn{i}')
            pool = getattr(self, f'pool{i}')
            
            x = conv(x)
            x = bn(x)
            x = self.relu(x)
            x = pool(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def setup_device():
    """Set up and return the device for computation (GPU or CPU)."""
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


def load_data(data_dir, image_size, batch_size, val_split):
    """Load and prepare the dataset for training and validation."""
    # Suppress specific warnings
    warnings.filterwarnings("ignore", message="Truncated File Read",
                            category=UserWarning, module="PIL.TiffImagePlugin")

    # Define transformations
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs, early_stopping_enabled=False, early_stopping_patience=3, 
                early_stopping_min_delta=0.001):
    """Train the model and validate it after each epoch."""
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
        print(f"Early stopping enabled with patience={early_stopping_patience}")

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)")

        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=running_loss/(train_loader_tqdm.n+1))

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        correct_predictions = 0
        total_samples = 0
        val_running_loss = 0.0

        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)")
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

        # Calculate metrics
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * correct_predictions / total_samples
        val_accuracies.append(val_accuracy)

        # Learning rate adjustment
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            print(f"New best model: {best_val_accuracy:.2f}% accuracy")

        # Print summary
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.2f}%")
        
        # Check early stopping
        if early_stopping_enabled and early_stopper(avg_val_loss):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print(f"\nTraining finished! Best validation accuracy: {best_val_accuracy:.2f}%")
    return best_model_state, best_val_accuracy


def save_model(model, model_path, onnx_path, image_size, device):
    """Save the trained model in PyTorch and ONNX formats."""
    # Save PyTorch model
    torch.save(model.state_dict(), model_path)
    print(f"PyTorch model saved to {model_path}")

    # Export to ONNX format
    model.eval()
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    try:
        with torch.no_grad():
            torch.onnx.export(
                model, dummy_input, onnx_path,
                export_params=True, opset_version=13,
                do_constant_folding=True, input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
        print(f"Model exported to ONNX format at {onnx_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")


def run_inference(image_path, model_file, image_size, device):
    """Run inference on a single image using a trained model."""
    import numpy as np
    from PIL import Image
    
    # Validate files exist
    if not os.path.exists(image_path) or not os.path.exists(model_file):
        print("Error: Image or model file not found")
        return None, None
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None
    
    # Process based on model type
    if model_file.endswith('.pth'):
        # PyTorch model
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
            
    elif model_file.endswith('.onnx'):
        # ONNX model
        try:
            import onnxruntime as ort
            # Create session and run inference
            ort_session = ort.InferenceSession(model_file)
            ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.cpu().numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)
            outputs = ort_outputs[0]
            
            # Process results
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
    
    # Print results
    print(f"\nInference Results:\nImage: {image_path}\nPrediction: {pred_class}\nConfidence: {conf_score:.2%}")
    return pred_class, conf_score


def main():
    """Main function to train the model or run inference."""
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

    # Load data
    train_loader, val_loader = load_data(
        args.data_dir, args.image_size, args.batch_size, args.val_split
    )

    # Initialize model
    model = CatDogCNN(args.image_size).to(device)
    print("\nModel Architecture:")
    print(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.learning_rate,
        momentum=args.momentum, weight_decay=args.weight_decay
    )

    # Learning rate scheduler
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
    save_model(model, args.model_path, args.onnx_path, args.image_size, device)


if __name__ == "__main__":
    main()