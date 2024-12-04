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
from lab_gatr import PointCloudPoolingScales, LaBGATr
import matplotlib.pyplot as plt
from gatr.interface.point import embed_point, extract_point

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU
device = 'cpu'
print(f"Using device: {device}")

#%%
class ConvexHullDataset(Dataset):
    def __init__(self, data_dir, target='inradius'):
        self.data_dir = data_dir
        self.file_names = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')])
        self.samples = []
        self.target = target
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
                    value = 0.0
                else:
                    try:
                        hull = ConvexHull(points)
                        if self.target == 'vertices':
                            value = len(hull.vertices)
                        elif self.target == 'facets':
                            value = len(hull.simplices)
                        elif self.target == 'volume':
                            value = hull.volume
                        elif self.target == 'area':
                            value = hull.area
                        elif self.target == 'inradius':
                            value = compute_inradius(hull)
                        else:
                            value = 0.0
                    except:
                        value = 0.0
                self.samples.append({'points': points, 'target': value})
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        points = sample['points']
        target = sample['target']
        return {'points': torch.tensor(points, dtype=torch.float32), 'target': torch.tensor(target, dtype=torch.float32)}

def compute_inradius(hull):
    """
    Computes the inradius of the convex hull.

    Parameters
    ----------
    hull : scipy.spatial.ConvexHull
        The convex hull object.

    Returns
    -------
    float
        The inradius of the convex hull.
    """
    # Compute the dual hull to find the inradius
    # In 3D, inradius is the radius of the largest sphere inscribed in the convex hull
    # This requires solving a linear programming problem
    # Here, we provide a simplified approximation using the radius of the inscribed sphere within the bounding box
    # For exact computation, consider using optimization libraries
    points = hull.points
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    bounding_box = max_coords - min_coords
    inradius = np.min(bounding_box) / 2.0
    return inradius

class GeometricAlgebraInterface:
    num_input_channels = 1
    num_output_channels = 1
    num_input_scalars = 1
    num_output_scalars = 1

    @staticmethod
    @torch.no_grad()
    def embed(data):
        """
        Embeds 3D points into multivectors.
        """
        points = data.pos
        multivectors = embed_point(points).unsqueeze(1)
        scalars = torch.zeros(multivectors.shape[0], multivectors.shape[1])
        return multivectors, scalars

    @staticmethod
    def dislodge(multivectors, scalars):
        """
        Extracts regression targets from multivectors.
        """
        # Remove channel dimension and flatten
        multivectors = multivectors.squeeze(1).view(-1, 16)  # [batch_size * num_points, 16]
        # Here, we ignore multivectors and return scalars as output
        output = scalars.squeeze(0)
        return output

#%%
def train_model_gatr(model, train_loader, test_loader, epochs=50, lr=1e-4, clip_value=1.0):
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
        Training and testing losses.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler to reduce LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=5, 
                                                     verbose=True)
    
    train_losses = []
    test_losses = []
    
    weights_dir = './gatr_weights'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    for epoch in range(epochs):
        model.train()
        running_loss = []
        
        # Save model weights for this epoch
        weights_path = os.path.join(weights_dir, f'gatr_epoch_{epoch}.pt')
        torch.save(model.state_dict(), weights_path)
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', leave=False)):
            points = batch['points'].to(device)    # [batch_size, num_points, 3]
            targets = batch['target'].to(device)  # [batch_size]
            
            optimizer.zero_grad()
            transform = PointCloudPoolingScales(rel_sampling_ratios=(1.0,), interp_simplex='triangle')
            data = pyg.data.Data(pos=points)  # Using 'points' directly
            
            # Reshape if necessary
            if data.pos.dim() == 3:
                batch_size, num_points, _ = data.pos.shape
                data.pos = data.pos.view(-1, 3)  # Flatten to [batch_size * num_points, 3]
            
            data = transform(data)
            
            # Ensure all required keys are present
            assert 'scale0_interp_source' in data, "scale0_interp_source key is missing!"
            assert 'scale0_interp_target' in data, "scale0_interp_target key is missing!"
            
            # Forward pass
            outputs = model(data) 
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
        
        epoch_loss = sum(running_loss) / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                points = batch['points'].to(device)
                targets = batch['target'].to(device)
                data = pyg.data.Data(pos=points)
                
                # Reshape if necessary
                if data.pos.dim() == 3:
                    batch_size, num_points, _ = data.pos.shape
                    data.pos = data.pos.view(-1, 3)  # Flatten to [batch_size * num_points, 3]
                
                data = PointCloudPoolingScales(rel_sampling_ratios=(0.2,), interp_simplex='triangle')(data)
                
                outputs = model(data)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * points.size(0)
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        
        # Step the scheduler based on test loss
        scheduler.step(test_loss)
        
        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        # Check for NaNs in loss
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
    
    return train_losses, test_losses

#%%
if __name__ == "__main__":
    # Initialize GATr model with the updated interface
    gatr_model = LaBGATr(
        GeometricAlgebraInterface,
        d_model=8,
        num_blocks=10,
        num_attn_heads=4,
        use_class_token=False
    ).to(device)
    
    # Initialize dataset with 'inradius' target
    dataset = ConvexHullDataset(data_dir, target='inradius')
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    def collate_fn(batch):
        points = torch.stack([item['points'] for item in batch], dim=0)  # Shape: [batch_size, num_points, 3]
        targets = torch.stack([item['target'] for item in batch], dim=0)  # Shape: [batch_size]
        return {'points': points, 'target': targets}
    
    batch_size = 1
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        drop_last=True  # Ensures consistent batch sizes
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        drop_last=True  # Ensures consistent batch sizes
    )
    
    # Train the GATr model
    gatr_train_losses, gatr_test_losses = train_model_gatr(
        gatr_model,
        train_loader,
        test_loader,
        epochs=50,
        lr=1e-3
    )
    
    # Import necessary modules for plotting
    import matplotlib.pyplot as plt

    # Create the loss_curves directory if it doesn't exist
    os.makedirs('./loss_curves', exist_ok=True)

    # Plot the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(gatr_train_losses, label='Train Loss', color='blue')
    plt.plot(gatr_test_losses, label='Test Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Losses over Epochs - Inradius')
    plt.legend()
    plt.grid(True)
    
    # Save the plot as a .png file
    plt.savefig('./loss_curves/GATR_loss_curves_inradius.png')
    plt.close()
    
    print("Loss curves have been saved to ./loss_curves/GATR_loss_curves_inradius.png") 