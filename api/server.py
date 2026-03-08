import os
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from lib.utils import re_normalization
from model.ASTGCN_r import make_model
from lib.utils import get_adjacency_matrix, load_graphdata_channel1

app = FastAPI()

class PredictRequest(BaseModel):
    values: list

state = {
    "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "net": None,
    "mean": None,
    "std": None,
    "buffer": None,
    "N": None,
    "T": None
}

def build_net(config_path, params_path):
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    data_cfg = cfg['Data']
    train_cfg = cfg['Training']
    adj_filename = data_cfg['adj_filename']
    num_of_vertices = int(data_cfg['num_of_vertices'])
    points_per_hour = int(data_cfg['points_per_hour'])
    num_for_predict = int(data_cfg['num_for_predict'])
    len_input = int(data_cfg['len_input'])
    in_channels = int(train_cfg['in_channels'])
    nb_block = int(train_cfg['nb_block'])
    K = int(train_cfg['K'])
    nb_chev_filter = int(train_cfg['nb_chev_filter'])
    nb_time_filter = int(train_cfg['nb_time_filter'])
    num_of_hours = int(train_cfg['num_of_hours'])
    time_strides = num_of_hours
    DEVICE = state["DEVICE"]
    adj_mx, _ = get_adjacency_matrix(adj_filename, num_of_vertices, None)
    net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx, num_for_predict, len_input, num_of_vertices)
    net.load_state_dict(torch.load(params_path, map_location=DEVICE))
    net.eval()
    return net, len_input, num_for_predict, num_of_vertices

@app.post("/warmup")
def warmup(config_path: str, params_path: str):
    net, T_in, T_out, N = build_net(config_path, params_path)
    state["net"] = net
    state["T"] = T_in
    state["N"] = N
    from configparser import ConfigParser
    cfg = ConfigParser()
    cfg.read(config_path)
    base = cfg['Data']['graph_signal_matrix_filename']
    r = cfg['Training']['num_of_hours']
    d = cfg['Training']['num_of_days']
    w = cfg['Training']['num_of_weeks']
    import os
    file = os.path.basename(base).split('.')[0]
    dirp = os.path.dirname(base)
    fn = os.path.join(dirp, f"{file}_r{r}_d{d}_w{w}_astcgn.npz")
    z = np.load(fn)
    mean = z['mean'][:, :, 0:1, :]
    std = z['std'][:, :, 0:1, :]
    state["mean"] = mean
    state["std"] = std
    state["buffer"] = np.zeros((N, 1, T_in), dtype=np.float32)
    return {"ok": True, "N": N, "T_in": T_in, "T_out": int(cfg['Data']['num_for_predict'])}

@app.post("/predict")
def predict(req: PredictRequest):
    assert state["net"] is not None
    x = np.array(req.values, dtype=np.float32).reshape(state["N"], 1)
    state["buffer"] = np.concatenate([state["buffer"][:, :, 1:], x.reshape(state["N"], 1, 1)], axis=2)
    DEVICE = state["DEVICE"]
    inp = torch.from_numpy(state["buffer"]).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y = state["net"](inp).cpu().numpy()
    pred = re_normalization(y, state["mean"], state["std"])
    return {"pred": pred[0].tolist()}

@app.get("/health")
def health():
    return {"ok": state["net"] is not None}
