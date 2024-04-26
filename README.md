# Superpoint Graph python pack
Generate superpoint cluster using coords (and colors).
> Modified from [FRNN](https://github.com/lxxue/FRNN.git), [SPG](https://github.com/loicland/superpoint_graph), [SuperCluster](https://github.com/drprojects/superpoint_transformer)

## Environment

```bash
conda create -n spg python=3.8.16 -y
conda activate spg
# install pytorch
conda install -n spg pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
# install point_geometric_features
pip install git+https://github.com/drprojects/point_geometric_features.git
# install dependences
conda install -n spg omegaconf numpy pyyaml -y
# install superpoint_graph in main folder
pip install .
```

## Usage

See [spg.ipynb](test/transform/spg.ipynb) and [spg.py](test/transform/spg.py)