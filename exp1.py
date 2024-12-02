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
from lab_gatr import PointCloudPoolingScales, LaBGATr
import matplotlib.pyplot as plt
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer



# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
#%%
import os
from scipy.spatial import ConvexHull

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
    
    #print(f"points.shape: {points.shape}")
    #print(f"volumes.shape: {volumes.shape}")
    return {'points': points, 'volume': volumes}

batch_size = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=3, embed_dim=64, num_heads=8, num_layers=3, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        """
        x: [batch_size, num_points, 3]
        """
        x = self.embedding(x)  # [batch_size, num_points, embed_dim]
        x = x.permute(1, 0, 2)  # [num_points, batch_size, embed_dim] for Transformer
        x = self.transformer(x)  # [num_points, batch_size, embed_dim]
        x = x.permute(1, 2, 0)  # [batch_size, embed_dim, num_points]
        x = self.pooling(x).squeeze(-1)  # [batch_size, embed_dim]
        x = self.regressor(x).squeeze(-1)  # [batch_size]
        return x

def train_model_transformer(model, train_loader, test_loader, epochs=50, lr=1e-3):
    """
    Training loop for the TransformerRegressor model.

    Parameters
    ----------
    model : torch.nn.Module
        The TransformerRegressor model.
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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    #i=0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            # Extract 'points' and 'volume' from the batch
            points = batch['points'].to(device)    # [batch_size, num_points, 3]
            volumes = batch['volume'].to(device)  # [batch_size]
            
            optimizer.zero_grad()
            outputs = model(points)                # [batch_size]
            loss = criterion(outputs, volumes)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * points.size(0)
            #i+=1
            #if i%100==0:
            #    print(f"predicted: {outputs[0]}")
            #    print(f"target: {volumes[0]}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                points = batch['points'].to(device)
                volumes = batch['volume'].to(device)
                outputs = model(points)
                loss = criterion(outputs, volumes)
                test_loss += loss.item() * points.size(0)
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        
        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    return train_losses, test_losses

# Initialize TransformerRegressor model
transformer_model = TransformerRegressor().to(device)

# Train the Transformer model

"""
transformer_train_losses, transformer_test_losses = train_model_transformer(
    transformer_model,
    train_loader,
    test_loader,
    epochs=50,
    lr=1e-3
)
"""

# %%
import torch
from lab_gatr import LaBGATr
from gatr.interface.point import embed_point, extract_point

class GeometricAlgebraInterface:
    num_input_channels = 1
    num_output_channels = 1
    num_input_scalars = 1
    num_output_scalars = 1

    @staticmethod
    @torch.no_grad()
    def embed2(data):
        """
        Embeds 3D points into multivectors.
        """
        points = data['points']  # Shape: [batch_size, num_points, 3]
        volumes = data['volume']  # Shape: [batch_size]

        print(f"[Embed] Points shape: {points.shape}")      # Debug
        print(f"[Embed] Volumes shape: {volumes.shape}")  # Debug

        # Flatten points for embedding
        batch_size, num_points, _ = points.shape
        points_flat = points.view(-1, 3)  # Shape: [batch_size * num_points, 3]
        print(f"[Embed] Points_flat shape: {points_flat.shape}")  # Debug

        multivectors = embed_point(points_flat)  # Shape: [batch_size * num_points, 16]
        print(f"[Embed] Multivectors shape after embedding: {multivectors.shape}")  # Debug

        # Reshape back to [batch_size, num_points, 16] and add channel dimension
        multivectors = multivectors.view(batch_size, num_points, 16).unsqueeze(1)  # [batch_size, 1, num_points, 16]
        print(f"[Embed] Multivectors reshaped: {multivectors.shape}")  # Debug

        # Replicate volume for each point
        scalars = volumes.view(batch_size, 1).repeat(1, num_points)  # [batch_size, num_points]
        scalars = scalars.unsqueeze(-1)  # [batch_size, num_points, 1]
        print(f"[Embed] Scalars shape: {scalars.shape}")  # Debug

        return multivectors, scalars
    
    @staticmethod
    @torch.no_grad()
    def embed(data):
        """
        Embeds 3D points into multivectors.
        """
        points = data.pos
        multivectors = embed_point(points).unsqueeze(1)
        scalars = torch.zeros(multivectors.shape[0],multivectors.shape[1])
        return multivectors, scalars

    @staticmethod
    def dislodge2(multivectors, scalars):
        """
        Extracts 3D points from multivectors.
        """
        # Remove channel dimension and flatten
        multivectors = multivectors.squeeze(1).view(-1, 16)  # [batch_size * num_points, 16]
        print(f"[Dislodge] Multivectors shape after squeezing and flattening: {multivectors.shape}")  # Debug

        # Extract points from multivectors
        points = extract_point(multivectors)  # [batch_size * num_points, 3]
        print(f"[Dislodge] Points shape after extraction: {points.shape}")  # Debug

        # Reshape back to [batch_size, num_points, 3]
        batch_size = scalars.shape[0]
        num_points = scalars.shape[1]
        points = points.view(batch_size, num_points, 3)
        print(f"[Dislodge] Points reshaped: {points.shape}")  # Debug

        return points
    @staticmethod
    def dislodge(multivectors,scalars):
        """
        Extracts 3D points from multivectors.
        """
        # Remove channel dimension and flatten
        multivectors = multivectors.squeeze(1).view(-1, 16)  # [batch_size * num_points, 16]
        # Extract points from multivectors
        points = extract_point(multivectors)  # [batch_size * num_points, 3]
        output=scalars[0]
        return output

# %%
from tqdm import tqdm
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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = []
        i=0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', leave=False, 
                         postfix={'output': 0.0, 'target': 0.0}):
            optimizer.zero_grad()
            transform = PointCloudPoolingScales(rel_sampling_ratios=(1.0,), interp_simplex='triangle')
            data = pyg.data.Data(pos=batch['points'])
            b,c,_=data.pos.shape
            data.pos = data.pos.view(-1, 3)
            
            data = transform(data)
            #data.pos = data.pos.view(b,c,3)
            #data.pos = data.pos.view(b*c,3)
            
            # Debug: Ensure all required keys are present
            assert 'scale0_interp_source' in data, "scale0_interp_source key is missing!"
            assert 'scale0_interp_target' in data, "scale0_interp_target key is missing!"
            
            # Forward pass
            outputs = model(data) 
            loss = criterion(outputs, batch['volume'])
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.item() * batch['points'].size(0))
            i+=1
            if i%100==0:
                print(f"predicted: {outputs[0]}")
                print(f"target: {batch['volume'][0]}")
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                transform = PointCloudPoolingScales(rel_sampling_ratios=(0.2,), interp_simplex='triangle')
                data = pyg.data.Data(pos=batch['points'])
                outputs = model(data)
                loss = criterion(outputs, batch['volume'])
                test_loss += loss.item() * batch['points'].size(0)
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        
        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    return train_losses, test_losses

# %%
# Initialize GATr model with the updated interface
device='cpu'
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



# %%



