import time
import os
import numpy as np
import requests
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_graph_path", default="./data/PEMS04/PEMS04.npz")
    p.add_argument("--num_of_hours", type=int, default=1)
    p.add_argument("--num_of_days", type=int, default=0)
    p.add_argument("--num_of_weeks", type=int, default=0)
    p.add_argument("--interval", type=float, default=0.1)
    p.add_argument("--endpoint", default="http://127.0.0.1:8000/predict")
    args = p.parse_args()
    file = os.path.basename(args.base_graph_path).split('.')[0]
    dirp = os.path.dirname(args.base_graph_path)
    fn = os.path.join(dirp, f"{file}_r{args.num_of_hours}_d{args.num_of_days}_w{args.num_of_weeks}_astcgn.npz")
    z = np.load(fn)
    x = z["test_x"][:, :, 0:1, :]
    seq = x.reshape(x.shape[0], x.shape[1], x.shape[3])
    for s in range(seq.shape[0]):
        step = seq[s].mean(axis=1)
        for t in range(seq.shape[2]):
            frame = seq[s, :, t]
            requests.post(args.endpoint, json={"values": frame.tolist()})
            time.sleep(args.interval)

if __name__ == "__main__":
    main()
