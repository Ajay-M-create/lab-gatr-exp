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
                for line in lines:
                    x, y, z = map(float, line.strip().split())
                    points.append([x, y, z])
                points = np.array(points)
                if points.shape[0] < 4:
                    # Convex hull in 3D requires at least 4 non-coplanar points
                    volume = 0.0
                else:
                    try:
                        hull = ConvexHull(points)
                        volume = hull.volume
                    except:
                        # In case points are coplanar or singular
                        volume = 0.0
                self.samples.append({'points': points, 'volume': volume})
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        points = sample['points']
        volume = sample['volume']
        return {'points': torch.tensor(points, dtype=torch.float32), 'volume': torch.tensor(volume, dtype=torch.float32)}

class GATrVolumeModel(nn.Module):
    def __init__(self, blocks=10, hidden_mv_channels=16, hidden_s_channels=32):
        super(GATrVolumeModel, self).__init__()
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
        # Average over num_points and scalar dimensions only
        outputs = torch.mean(nodewise_outputs, dim=(1, 2))  # [batch_size]
        
        return outputs.squeeze(-1)  # [batch_size]

def train_model(model, train_loader, test_loader, epochs=50, lr=1e-4, clip_value=1.0):
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
    
    weights_dir = './gatr_weights'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    for epoch in range(epochs):
        model.train()
        running_loss = []
        pred_list = []
        target_list = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)):
            points = batch['points'].to(device)    # [batch_size, num_points, 3]
            targets = batch['volume'].to(device)  # [batch_size]
            
            optimizer.zero_grad()
            
            outputs = model(points)  # [batch_size]
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Apply gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            # Check for NaNs in gradients
            #nan_grad = False
            #for name, param in model.named_parameters():
            #     if param.grad is not None and torch.isnan(param.grad).any():
            #        print(f"NaN detected in gradients of parameter: {name}")
            #        nan_grad = True
            #        break
            #if nan_grad:
            #    print("NaN detected in gradients. Skipping optimizer step.")
            #    continue  # Skip updating weights for this batch
            
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
        pred_list = []
        target_list = []
        with torch.no_grad():
            for batch in test_loader:
                points = batch['points'].to(device)
                targets = batch['volume'].to(device)
                outputs = model(points)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * points.size(0)
                pred_list.extend(outputs.detach().cpu().numpy())
                target_list.extend(targets.detach().cpu().numpy())
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        
        plt.figure(figsize=(10, 5))
        plt.scatter(target_list, pred_list, alpha=0.6, edgecolors='w', linewidth=0.5)
        plt.xlabel('Actual Volume')
        plt.ylabel('Predicted Volume')
        plt.title('GATr Predicted vs Actual Volumes (Test) - Intermediate Epoch')
        min_val = min(min(target_list), min(pred_list))
        max_val = max(max(target_list), max(pred_list))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # Diagonal line
        plt.grid(True)
        plt.savefig(os.path.join("./loss_curves", 'GATR_predicted_vs_actual_volumes_intermediate.png'))
        plt.close()
        
        # Step the scheduler based on test loss
        #scheduler.step(test_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        # Save model weights for this epoch
        weights_path = os.path.join(weights_dir, f'gatr_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), weights_path)
        
        # Early stopping if loss is NaN
        if np.isnan(epoch_loss) or np.isnan(test_loss):
            print("NaN detected in loss, stopping training.")
            break
        
        # Check for NaNs in model weights
        nan_weights = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN detected in parameter: {name}")
                nan_weights = True
                break
        if nan_weights:
            print("NaN detected in model weights, stopping training.")
            break
    
    return train_losses, test_losses, all_predictions, all_targets

def visualize_results(train_losses, test_losses, predictions, targets, save_dir='./loss_curves/'):
    """
    Plots and saves the training/testing loss curves and the final predicted vs actual volumes.

    Parameters
    ----------
    train_losses : list
        List of training losses per epoch.
    test_losses : list
        List of testing losses per epoch.
    predictions : np.ndarray
        Array of all predicted volumes.
    targets : np.ndarray
        Array of all actual volumes.
    save_dir : str
        Directory to save the loss curves and final prediction plot.
    """
    # Create the loss_curves directory if it doesn't exist
    targets = np.array(targets)
    predictions = np.array(predictions)
    os.makedirs(save_dir, exist_ok=True)

    # Plot the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('GATr Training and Testing Losses over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'GATR_loss_curves_task_volume.png'))
    plt.close()

    # Plot predicted vs actual volumes with accuracy
    plt.figure(figsize=(10, 5))
    plt.scatter(targets, predictions, alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.xlabel('Actual Volume')
    plt.ylabel('Predicted Volume')
    plt.title('GATr Predicted vs Actual Volumes - Final Epoch')
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # Diagonal line
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'GATR_predicted_vs_actual_volumes_final.png'))
    plt.close()

    print(f"Loss curves and final prediction plot have been saved to {save_dir}")

if __name__ == "__main__":
    data_dir = '3d_point_cloud_dataset'  # Adjust the path if necessary
    dataset = ConvexHullDataset(data_dir)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    def collate_fn(batch):
        points = torch.stack([item['points'] for item in batch], dim=0)  # Shape: [batch_size, num_points, 3]
        volumes = torch.stack([item['volume'] for item in batch], dim=0)  # Shape: [batch_size]
        return {'points': points, 'volume': volumes}
    
    batch_size = 50
    
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
    gatr_model = GATrVolumeModel(
        blocks=10,
        hidden_mv_channels=16,
        hidden_s_channels=32
    ).to(device)
    
    # Train the GATr model
    train_losses, test_losses, predictions, targets = train_model(
        gatr_model,
        train_loader,
        test_loader,
        epochs=5,
        lr=1e-4
    )
    
    # Visualize the final results
    visualize_results(train_losses, test_losses, predictions, targets)