#%%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

#%%
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
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        # Initialize tqdm progress bar for batches
        batch_bar = tqdm(enumerate(train_loader, 1), desc=f"Epoch {epoch}/{epochs}", total=len(train_loader), leave=False)
        for batch_idx, batch in batch_bar:
            # Extract 'points' and 'volume' from the batch
            points = batch['points'].to(device)    # [batch_size, num_points, 3]
            volumes = batch['volume'].to(device)  # [batch_size]

            optimizer.zero_grad()
            outputs = model(points)                # [batch_size]
            loss = criterion(outputs, volumes)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * points.size(0)

            # Update tqdm with current prediction and target every 200 iterations
            # if batch_idx % 200 == 0:
            #    batch_bar.set_postfix({
            #        "Predicted Volumes": f"{[f'{x:.4f}' for x in outputs.detach().cpu().numpy()]}",
            #        "Expected Volumes": f"{[f'{x:.4f}' for x in volumes.detach().cpu().numpy()]}"
            #      })

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

#%%
if __name__ == "__main__":
    # Initialize TransformerRegressor model
    transformer_model = TransformerRegressor().to(device)
    
    # Train the Transformer model
    transformer_train_losses, transformer_test_losses = train_model_transformer(
        transformer_model,
        train_loader,
        test_loader,
        epochs=50,
        lr=1e-3
    )