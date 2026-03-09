import time
import os
import numpy as np
import requests
import argparse
import pandas as pd
import sys

def main():
    p = argparse.ArgumentParser()
    # We should use the processed data, which contains Speed + Time Embedding
    p.add_argument("--data_path", default="./data/processed/train_data.npz")
    p.add_argument("--interval", type=float, default=1.0, help="Simulated interval in seconds (e.g. 1s = 5min)")
    p.add_argument("--endpoint", default="http://127.0.0.1:8000/predict")
    args = p.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data file {args.data_path} not found.")
        return

    print(f"Loading test data from {args.data_path}...")
    data = np.load(args.data_path)
    
    # We use test set for simulation
    # Shape: (B, T, N, F) or (B, N, F, T)?
    # DataPreprocessor saved: test_x: (3395, 12, 307, 5) -> (B, T, N, F)
    test_x = data["test_x"] 
    
    print(f"Test data shape: {test_x.shape}")
    
    # Our API expects a single frame: (N, F)
    # F should be 3: [Speed, Hour, Day]
    # In processed data, indices are: 0-Flow, 1-Occ, 2-Speed, 3-Hour, 4-Day
    # So we need to slice [:, :, :, [2, 3, 4]]
    
    # Let's iterate through the test set.
    # The test set is windowed: (Sample 1: t0..t11), (Sample 2: t1..t12)...
    # To simulate a continuous stream, we can just take the *last* time step of each sample, 
    # or better, just reconstruct the sequence.
    # Since window shift is 1, sample[i][-1] is the new point at time t+window_size.
    # But to be simple, let's just iterate samples and for each sample, take the last time step?
    # No, that would skip the first window_size-1 steps.
    
    # Let's use the first sample fully, then the last step of subsequent samples.
    # Or just iterate all samples' last step?
    # The samples are:
    # 0: [0, 1, ..., 11]
    # 1: [1, 2, ..., 12]
    # ...
    # So taking the last element of each sample gives us: 11, 12, 13...
    # We miss 0..10.
    
    # Let's collect the timeline.
    # First sample: all 12 steps.
    # Subsequent samples: last step.
    
    stream_data = []
    # Add first sample's sequence
    first_sample = test_x[0] # (12, 307, 5)
    for t in range(first_sample.shape[0]):
        stream_data.append(first_sample[t])
        
    # Add subsequent samples' last step
    for i in range(1, test_x.shape[0]):
        last_step = test_x[i, -1, :, :] # (307, 5)
        stream_data.append(last_step)
        
    print(f"Prepared stream of {len(stream_data)} time steps.")
    
    # Start simulation
    print(f"Starting simulation to {args.endpoint}...")
    print(f"Interval: {args.interval}s")
    
    for i, frame in enumerate(stream_data):
        # frame shape: (307, 5)
        # We need to extract Speed, Hour, Day -> indices [2, 3, 4]
        # API expects (307, 3)
        
        # Check if frame has 5 channels
        if frame.shape[1] >= 5:
            api_input = frame[:, [2, 3, 4]] # (307, 3)
        else:
            # Fallback if only 3 channels (Flow, Occ, Speed)
            # Just send Speed? Or handle gracefully
            print("Warning: Frame has less than 5 channels. Sending last channel as speed?")
            # Assuming channel 2 is speed
            api_input = frame[:, 2:3] # (307, 1) -> This might fail if API expects 3
            # If API expects 3, we might need to pad or fail.
            # Let's assume we have 5 channels as per previous steps.
        
        # Current timestamp (simulated)
        # We can construct it from Hour/Day info if needed, or just use current time
        ts = pd.Timestamp.now().isoformat()
        
        payload = {
            "values": api_input.tolist(),
            "timestamp": ts
        }
        
        try:
            resp = requests.post(args.endpoint, json=payload)
            if resp.status_code == 200:
                print(f"Step {i}/{len(stream_data)}: Sent successfully. Response: 200 OK")
            else:
                print(f"Step {i}/{len(stream_data)}: Failed. Status: {resp.status_code}, Msg: {resp.text}")
        except Exception as e:
            print(f"Step {i}/{len(stream_data)}: Connection Error: {e}")
            
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
