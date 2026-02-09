import os
import argparse
import numpy as np
# 将 data/processed/train_data.npz 转换为 ASTGCN 训练脚本所需的统一结构文件
def split_data(X, Y, train_ratio=0.6, val_ratio=0.2):
    total = X.shape[0]
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_x, train_y = X[:train_end], Y[:train_end]
    val_x, val_y = X[train_end:val_end], Y[train_end:val_end]
    test_x, test_y = X[val_end:], Y[val_end:]
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)

def to_b_n_f_t(x):
    return np.transpose(x, (0, 2, 3, 1))

def compute_stats(train_b_n_f_t):
    mean = train_b_n_f_t.mean(axis=(0, 1, 3), keepdims=True)
    std = train_b_n_f_t.std(axis=(0, 1, 3), keepdims=True)
    return mean, std

def save_npz(out_path, train_x, train_y, val_x, val_y, test_x, test_y, mean, std):
    train_ts = np.arange(train_x.shape[0]).reshape(-1, 1)
    val_ts = np.arange(val_x.shape[0]).reshape(-1, 1)
    test_ts = np.arange(test_x.shape[0]).reshape(-1, 1)
    np.savez_compressed(
        out_path,
        train_x=train_x,
        train_target=train_y,
        train_timestamp=train_ts,
        val_x=val_x,
        val_target=val_y,
        val_timestamp=val_ts,
        test_x=test_x,
        test_target=test_y,
        test_timestamp=test_ts,
        mean=mean,
        std=std
    )

def build_output_path(base_graph_path, num_of_hours, num_of_days, num_of_weeks):
    base_dir = os.path.dirname(base_graph_path)
    base_name = os.path.basename(base_graph_path).split('.')[0]
    out_name = f"{base_name}_r{num_of_hours}_d{num_of_days}_w{num_of_weeks}_astcgn.npz"
    return os.path.join(base_dir, out_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="./data/processed", type=str)
    parser.add_argument("--base_graph_path", default="./data/PEMS04/PEMS04.npz", type=str)
    parser.add_argument("--num_of_hours", default=1, type=int)
    parser.add_argument("--num_of_days", default=0, type=int)
    parser.add_argument("--num_of_weeks", default=0, type=int)
    args = parser.parse_args()

    npz_path = os.path.join(args.processed_dir, "train_data.npz")
    data = np.load(npz_path)
    X = data["x"]
    Y = data["y"]

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = split_data(X, Y, train_ratio=0.6, val_ratio=0.2)

    train_x_b_n_f_t = to_b_n_f_t(train_x)
    val_x_b_n_f_t = to_b_n_f_t(val_x)
    test_x_b_n_f_t = to_b_n_f_t(test_x)

    mean, std = compute_stats(train_x_b_n_f_t)

    out_path = build_output_path(args.base_graph_path, args.num_of_hours, args.num_of_days, args.num_of_weeks)
    save_npz(out_path, train_x_b_n_f_t, train_y, val_x_b_n_f_t, val_y, test_x_b_n_f_t, test_y, mean, std)
    print("save file:", out_path)

if __name__ == "__main__":
    main()
