import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch_geometric as pyg
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar
import torch.nn.functional as F

# Ensure reproducibility
torch.manual_seed(41)
np.random.seed(41)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ConvexHullDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_names = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')])
        self.samples = []
        self._prepare_dataset()
    
    def _prepare_dataset(self):
        for file_name in tqdm(self.file_names, desc="Loading data"):
            file_path = os.path.join(self.data_dir, file_name)
            with open(file_path, 'r') as f:
                lines = f.readlines()[1:]  # Skip header
                points = []
                # Only read 2 points since the dataset now contains two points
                for line in lines[:2]:  
                    x, y, z = map(float, line.strip().split())
                    points.append([x, y, z])
                points = np.array(points)
                # Compute Euclidean distance between the two points
                if len(points) == 2:
                    distance = np.linalg.norm(points[0] - points[1])
                else:
                    distance = 0.0  # Default value if not exactly two points
                self.samples.append({'points': points, 'distance': distance})
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        points = sample['points']
        distance = sample['distance']
        return {
            'points': torch.tensor(points, dtype=torch.float32),
            'distance': torch.tensor(distance, dtype=torch.float32)
        }

class GATrDistanceModel(nn.Module):
    def __init__(self, blocks=10, hidden_mv_channels=16, hidden_s_channels=32):
        super(GATrDistanceModel, self).__init__()
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
        )
        self.output_layer = nn.Linear(1, 1)
    
    def forward(self, inputs):
        """
        inputs: Tensor of shape [batch_size, num_points, 3]
        """
        # Embed point cloud in PGA
        embedded_inputs = embed_point(inputs)  # [batch_size, num_points, 16]
        embedded_inputs = embedded_inputs.unsqueeze(-2)  # [batch_size, num_points, 1, 16]
        
        # Pass data through GATr
        embedded_outputs, _ = self.gatr(embedded_inputs, scalars=None)  # [batch_size, num_points, 1, 16]
        
        # Extract scalar and aggregate
        nodewise_outputs = extract_scalar(embedded_outputs)  # [batch_size, num_points, 1]
        
        # **Updated Aggregation: Average only over num_points (dim=1)**
        outputs = torch.mean(nodewise_outputs, dim=1)  # [batch_size, 1]
        
        return outputs.squeeze(-2,-3)  # [batch_size]

def train_model(model, train_loader, test_loader, epochs=10, lr=1e-4, clip_value=1.0):
    """
    Training loop for the GATr model with gradient clipping and learning rate scheduling.

    Parameters
    ----------
    model : torch.nn.Module
        The GATr model.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training set.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the testing set.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    clip_value : float
        Maximum allowed value of gradients.

    Returns
    -------
    tuple
        Training and testing losses, predictions and targets.
    """
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler to reduce LR on plateau
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    train_losses = []
    test_losses = []
    all_predictions = []
    all_targets = []
    
    #weights_dir = './gatr_weights'
    #if not os.path.exists(weights_dir):
    #    os.makedirs(weights_dir)
    
    for epoch in range(epochs):
        model.train()
        running_loss = []
        pred_list = []
        target_list = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)):
            points = batch['points'].to(device)    # [batch_size, num_points, 3]
            targets = batch['distance'].to(device)  # [batch_size]
            
            optimizer.zero_grad()
            
            outputs = model(points)  # [batch_size]
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            # Check for NaNs in gradients
            nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN detected in gradients of parameter: {name}")
                    nan_grad = True
                    break
            if nan_grad:
                print("NaN detected in gradients. Skipping optimizer step.")
                continue  # Skip updating weights for this batch
            
            optimizer.step()
            
            running_loss.append(loss.item() * points.size(0))
            pred_list.extend(outputs.detach().cpu().numpy())
            target_list.extend(targets.detach().cpu().numpy())
        
        epoch_loss = sum(running_loss) / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        all_predictions.extend(pred_list)
        all_targets.extend(target_list)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        epoch_predictions = []
        epoch_targets = []
        with torch.no_grad():
            for batch in test_loader:
                points = batch['points'].to(device)
                targets = batch['distance'].to(device)
                outputs = model(points)
                epoch_predictions.extend(outputs.cpu().numpy())
                epoch_targets.extend(targets.cpu().numpy())
                loss = criterion(outputs, targets)
                test_loss += loss.item() * points.size(0)
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        # Plot predicted vs actual distances for this epoch
        plt.clf()
        plt.figure(figsize=(10, 5))
        plt.scatter(epoch_targets, epoch_predictions, alpha=0.6)
        plt.xlabel('Actual Distance')
        plt.ylabel('Predicted Distance')
        plt.title(f'Predicted vs Actual Distances (Epoch {epoch+1})')
        plt.plot([min(epoch_targets), max(epoch_targets)], [min(epoch_targets), max(epoch_targets)], 'r--')  # Diagonal line
        plt.grid(True)
        plt.savefig(f'./loss_curves/GATr_predicted_vs_actual_distance_epoch_{epoch+1}.png')
        plt.close()
        
        
        # Step the scheduler based on test loss
        #scheduler.step(test_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        # Save model weights for this epoch
        #weights_path = os.path.join(weights_dir, f'gatr_epoch_{epoch+1}.pt')
        #torch.save(model.state_dict(), weights_path)
        
        # Early stopping if loss is NaN
        #if np.isnan(epoch_loss) or np.isnan(test_loss):
        #    print("NaN detected in loss, stopping training.")
        #    break
        
        # Check for NaNs in model weights
        #nan_weights = False
        #for name, param in model.named_parameters():
        #    if torch.isnan(param).any():
        #        print(f"NaN detected in parameter: {name}")
        #        nan_weights = True
        #        break
        #if nan_weights:
        #    print("NaN detected in model weights, stopping training.")
        #    break
    
    return train_losses, test_losses, all_predictions, all_targets

if __name__ == "__main__":
    data_dir = '3d_two_points'  # Updated data directory
    dataset = ConvexHullDataset(data_dir)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    def collate_fn(batch):
        points = torch.stack([item['points'] for item in batch], dim=0)  # Shape: [batch_size, num_points, 3]
        distances = torch.stack([item['distance'] for item in batch], dim=0)  # Shape: [batch_size]
        return {'points': points, 'distance': distances}
    
    batch_size = 1
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        drop_last=True  # Ensure consistent batch sizes
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    # Initialize GATr model
    gatr_model = GATrDistanceModel(
        blocks=10,
        hidden_mv_channels=16,
        hidden_s_channels=32
    ).to(device)
    
    # Train the GATr model
    gatr_train_losses, gatr_test_losses, gatr_predictions, gatr_targets = train_model(
        gatr_model,
        train_loader,
        test_loader,
        epochs=3,
        lr=1e-4
    )
    
    # Create the loss_curves directory if it doesn't exist
    os.makedirs('./loss_curves', exist_ok=True)
    
    # Plot the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(gatr_train_losses, label='Train Loss', color='blue')
    plt.plot(gatr_test_losses, label='Test Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Losses over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Save the plot as a .png file
    plt.savefig('./loss_curves/GATr_loss_curves_task_distance.png')
    plt.close()
    
    print("Loss curves have been saved to ./loss_curves/GATr_loss_curves_task_distance.png")
    
    # Get predictions for the last epoch
    gatr_model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in test_loader:
            points = batch['points'].to(device)
            targets = batch['distance'].to(device)
            outputs = gatr_model(points)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(targets.cpu().numpy())
    
    # Plot predicted vs actual distances
    plt.figure(figsize=(10, 5))
    plt.scatter(targets, predictions, alpha=0.6)
    plt.xlabel('Actual Distance')
    plt.ylabel('Predicted Distance')
    plt.title('Predicted vs Actual Distances (Last Epoch)')
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')  # Diagonal line
    plt.grid(True)
    plt.savefig('./loss_curves/GATr_predicted_vs_actual_distance.png')
    plt.close()