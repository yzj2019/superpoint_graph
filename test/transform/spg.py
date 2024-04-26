import torch
from superpoint_graph import SPG
from omegaconf import OmegaConf

data_dict = torch.load("scene0000_00.pth")
cfg = OmegaConf.load("spg.yaml")
spg = SPG(cfg)
data_list = spg.segment_points(torch.from_numpy(data_dict["coord"]), torch.from_numpy(data_dict["color"]))
print(data_list[0]["super_index"])