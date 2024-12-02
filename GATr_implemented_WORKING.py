#%%
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
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer
from gatr.interface.point import embed_point, extract_point

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

#%%
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

batch_size = 1

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=collate_fn, 
    drop_last=True  # Added to ensure consistent batch sizes
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    collate_fn=collate_fn,
    drop_last=True  # Added to ensure consistent batch sizes
)

#%%
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
        scalars = torch.zeros(multivectors.shape[0], multivectors.shape[1]).to(device)
        return multivectors, scalars

    @staticmethod
    def dislodge(multivectors, scalars):
        """
        Extracts 3D points from multivectors.
        """
        # Remove channel dimension and flatten
        multivectors = multivectors.squeeze(1).view(-1, 16)  # [batch_size * num_points, 16]
        # Extract points from multivectors
        points = extract_point(multivectors)  # [batch_size * num_points, 3]
        output = scalars[0]
        return output

#%%
from lab_gatr import LaBGATr
from lab_gatr import PointCloudPoolingScales

def train_model_gatr(model, train_loader, test_loader, epochs=50, lr=1e-3):
    """
    Training loop for the GATr model.

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

    Returns
    -------
    tuple
        Training and testing losses.
    """
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    #i = 0
    
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
            points = batch['points']   # [batch_size, num_points, 3]
            volumes = batch['volume'].to(device)  # [batch_size]
            
            # Debug: Print tensor shapes
            #print(f"Epoch {epoch}, Batch {batch_idx}")
            #print(f"Points shape: {points.shape}")
            #print(f"Volumes shape: {volumes.shape}")
            
            optimizer.zero_grad()
            transform = PointCloudPoolingScales(rel_sampling_ratios=(1.0,), interp_simplex='triangle')
            data = pyg.data.Data(pos=points)  # Move points to device first
            
            
            # Reshape if necessary
            if data.pos.dim() == 3:
                batch_size, num_points, _ = data.pos.shape
                data.pos = data.pos.view(-1, 3)  # Flatten to [batch_size * num_points, 3]
            
            data = transform(data)  # Transform before moving to device
            data = data.to(device)  # Move all data to device
            
            # Debug: Ensure all required keys are present
            assert 'scale0_interp_source' in data, "scale0_interp_source key is missing!"
            assert 'scale0_interp_target' in data, "scale0_interp_target key is missing!"
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, volumes)
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.item() * points.size(0))
            #i += 1
            #if i % 125 == 0:
            #    print(f"predicted: {outputs[0].item()}")
            #    print(f"target: {volumes[0].item()}")
            
        epoch_loss = sum(running_loss) / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                points = batch['points'].to(device)
                volumes = batch['volume'].to(device)
                data = pyg.data.Data(pos=points)
                
                # Reshape if necessary
                if data.pos.dim() == 3:
                    batch_size, num_points, _ = data.pos.shape
                    data.pos = data.pos.view(-1, 3)  # Flatten to [batch_size * num_points, 3]
                
                data = PointCloudPoolingScales(rel_sampling_ratios=(1,), interp_simplex='triangle')(data)
                
                outputs = model(data)
                loss = criterion(outputs, volumes)
                test_loss += loss.item() * points.size(0)
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        
        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
    
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
    
    # Train the GATr model
    gatr_train_losses, gatr_test_losses = train_model_gatr(
        gatr_model,
        train_loader,
        test_loader,
        epochs=50,
        lr=1e-3
    )
    
#%% 