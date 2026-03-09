import torch
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.utils import load_graphdata_channel1

def verify_dataloader():
    print("Verifying DataLoader...")
    
    # Configuration (Simulated)
    graph_signal_matrix_filename = './data/processed/train_data.npz'
    num_of_hours = 1
    num_of_days = 0
    num_of_weeks = 0
    DEVICE = torch.device('cpu')
    batch_size = 32
    
    # Load Data
    try:
        train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
            graph_signal_matrix_filename, num_of_hours,
            num_of_days, num_of_weeks, DEVICE, batch_size
        )
        
        print("\nDataLoader Verification Success!")
        print("-" * 40)
        
        # Check Batch Dimension
        for i, (x, y) in enumerate(train_loader):
            print(f"Batch {i} Shape:")
            print(f"  Input X: {x.shape} (Expected: [32, 307, F, 12])")
            print(f"  Target Y: {y.shape} (Expected: [32, 307, 12])")
            
            # Verify Feature Channels
            # Current implementation in utils.py modified to take channel 2:3 (Speed)
            # But wait, preprocessor now adds time features (Index 3, 4)
            # Let's see what utils.py actually loads.
            # In previous turn, I modified utils.py to take `train_x[:, :, 2:3, :]` which is just Speed.
            # If we want Time Embeddings, we need to adjust utils.py to take more channels.
            
            print(f"  Feature Channels: {x.shape[2]}")
            if x.shape[2] == 1:
                print("  -> Only Speed channel loaded.")
            else:
                print(f"  -> {x.shape[2]} channels loaded.")
                
            break
            
        print("-" * 40)
        print("Mean/Std Shapes:", _mean.shape, _std.shape)
        
    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_dataloader()
