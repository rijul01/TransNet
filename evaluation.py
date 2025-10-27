#!/usr/bin/env python
# A minimal script to test an TransNet model using a single GPU

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from tqdm import tqdm
import datetime
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os
import torch
import numpy as np
from tqdm import tqdm
import time

# Import TransNet from your existing code
from TransNet_train import TransNet

def test_transnet():
    # Configuration
    checkpoint_path = "checkpoints/best_model.pt"  # Path to your saved model
    data_dir = "."  # Directory with your data files
    batch_size = 32
    device = torch.device("cuda:0")  # Use first GPU
    
    print("Loading data...")
    # Load data for testing
    scaled_data = np.load(os.path.join(data_dir, 'scaled_data.npy'))
    final_features = np.load(os.path.join(data_dir, 'final_features.npy'))
    data = np.concatenate([final_features, scaled_data[:-1, :, -1:]], axis=-1)
    
    # Load coordinates
    lats = np.load('lats.npy')
    lons = np.load('lons.npy')

    # Get test data (assuming last year of data)
    test_data = data[2*8760:3*8760]  # Last year
    
    print(f"Test data shape: {test_data.shape}")
    
    # Create sequences
    print("Creating test sequences...")
    X_test, y_test = create_sequences(test_data)
    
    print(f"Test sequences shape: {X_test.shape}")
    print(f"Test targets shape: {y_test.shape}")
    
    # Create dataset and loader
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Initialize model
    print("Initializing model...")
    model = TransNet(
        input_features=data.shape[-1],
        n_stations=len(lats),
        seq_length=24,
        pred_length=72,
        L=1,
        state_dim=64,
        hist_dim=64,
    )
    
    # Initialize graph processor
    model.initialize_graph_processor(lats, lons)
    
    # Load model weights
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Evaluate model
    print("Evaluating model...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader):
            # Move data to device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            
            # Store predictions and targets
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    # Concatenate predictions and targets
    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    # print("Calculating metrics...")
    # mae = np.mean(np.abs(predictions - targets))
    # mse = np.mean((predictions - targets) ** 2)
    # rmse = np.sqrt(mse)
    
    # # Calculate IOA (Index of Agreement)
    # target_mean = np.mean(targets)
    # numerator = np.sum((predictions - targets) ** 2)
    # denominator = np.sum((np.abs(predictions - target_mean) + np.abs(targets - target_mean)) ** 2)
    # ioa = 1 - (numerator / (denominator + 1e-6))
    
    # print(f"Test MAE: {mae:.4f}")
    # print(f"Test RMSE: {rmse:.4f}")
    # print(f"Test IOA: {ioa:.4f}")
    
    # Save predictions
    # os.makedirs('results', exist_ok=True)
    np.save('evaluation_results/predictions_2020.npy', predictions)
    np.save('evaluation_results/targets_2020.npy', targets)

    print("Evaluation completed!")

def create_sequences(data, seq_length = 24, target_horizons = 72):
    """
    Create sequences for training with specific forecast horizon
    
    Args:
        data: Shape (samples, nodes, features)
        original_pm25: Shape (samples, nodes) - contains original PM2.5 values
        seq_length: Input sequence length
        target_horizons: List of specific forecast horizons
        
    Returns:
        sequences: Shape (num_sequences, seq_length, nodes, features)
        targets: Shape (num_sequences, len(target_horizons), nodes)
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length - target_horizons):
        # Input sequence
        seq = data[i:(i + seq_length)]
        
        # Target sequence (PM2.5 values only) , original_pm25
        target = [
         data[(i + seq_length) : (i + seq_length + target_horizons), :, -1]]
                
        sequences.append(seq)
        targets.append(target)
        
    return np.array(sequences), np.array(np.squeeze(targets))

if __name__ == "__main__":
    test_transnet()