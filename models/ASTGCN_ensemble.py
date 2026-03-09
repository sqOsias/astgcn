import torch
import torch.nn as nn
from lib.utils import scaled_Laplacian, cheb_polynomial
from model.ASTGCN_r import ASTGCN_submodule

"""支持为 closeness、period、trend 分别构建子网络并以可学习权重融合输出；便于做消融与论文创新
"""
class ASTGCNEnsemble(nn.Module):
    def __init__(self, device, adj_mx, num_of_vertices, in_channels, num_for_predict, cfg_c, cfg_p=None, cfg_t=None):
        super().__init__()
        L = scaled_Laplacian(adj_mx)
        def polys(K):
            return [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in cheb_polynomial(L, K)]
        self.branches = nn.ModuleDict()
        if cfg_c:
            self.branches["c"] = ASTGCN_submodule(device, cfg_c["nb_block"], in_channels, cfg_c["K"], cfg_c["nb_chev_filter"], cfg_c["nb_time_filter"], cfg_c["time_strides"], polys(cfg_c["K"]), num_for_predict, cfg_c["len_input"], num_of_vertices)
        if cfg_p:
            self.branches["p"] = ASTGCN_submodule(device, cfg_p["nb_block"], in_channels, cfg_p["K"], cfg_p["nb_chev_filter"], cfg_p["nb_time_filter"], cfg_p["time_strides"], polys(cfg_p["K"]), num_for_predict, cfg_p["len_input"], num_of_vertices)
        if cfg_t:
            self.branches["t"] = ASTGCN_submodule(device, cfg_t["nb_block"], in_channels, cfg_t["K"], cfg_t["nb_chev_filter"], cfg_t["nb_time_filter"], cfg_t["time_strides"], polys(cfg_t["K"]), num_for_predict, cfg_t["len_input"], num_of_vertices)
        n = len(self.branches)
        self.alpha = nn.Parameter(torch.zeros(n))
        self.keys = list(self.branches.keys())
        self.to(device)
    def forward(self, x_c=None, x_p=None, x_t=None):
        outs = []
        feed = {"c": x_c, "p": x_p, "t": x_t}
        for k in self.keys:
            outs.append(self.branches[k](feed[k]))
        w = torch.softmax(self.alpha, dim=0)
        y = None
        for i, o in enumerate(outs):
            y = o * w[i] if y is None else y + o * w[i]
        return y
