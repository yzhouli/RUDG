# RUDG: Robust Uncertainty-Aware Directed Graph Learning for Financial Fraud Detection

Official PyTorch and DGL implementation of the paper **"RUDG: Robust Uncertainty-Aware Directed Graph Learning for Financial Fraud Detection."**

## Requirements

```bash
conda create -n rudg python=3.10 -y
conda activate rudg
pip install -r requirements.txt
```

For GPU execution, install the PyTorch and DGL builds compatible with your CUDA version.

```bash
python -m unittest discover -s tests -v
```

## Dataset Preparation

```bash
mkdir -p data/raw data/processed data/splits
```

### Amazon

Source: <https://github.com/YingtongDou/CARE-GNN/tree/master/data>

```bash
curl -L https://github.com/YingtongDou/CARE-GNN/raw/master/data/Amazon.zip -o Amazon.zip
unzip Amazon.zip -d data/raw/
```

Expected file:

```text
data/raw/Amazon.mat
```

### YelpChi

Source: <https://github.com/YingtongDou/CARE-GNN/tree/master/data>

```bash
curl -L https://github.com/YingtongDou/CARE-GNN/raw/master/data/YelpChi.zip -o YelpChi.zip
unzip YelpChi.zip -d data/raw/
```

Expected file:

```text
data/raw/YelpChi.mat
```

### T-Finance

Sources:

- <https://github.com/squareRoot3/GADBench>
- <https://github.com/squareRoot3/Rethinking-Anomaly-Detection>
- <https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1/view?usp=sharing>

```bash
pip install gdown
gdown 1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1 -O datasets.zip
unzip datasets.zip -d downloaded_datasets/
```

Copy the extracted DGL graph file named `tfinance` to:

```text
data/raw/tfinance
```

Or specify its original location during preprocessing:

```bash
python preprocess.py --dataset tfinance \
  --raw-path /path/to/tfinance \
  --seed 42
```

### DGraph-Fin

Sources:

- <https://dgraph.xinye.com/>
- <https://github.com/DGraphXinye/DGraphFin_baseline>

Download `DGraphFin.zip` from the dataset website and run:

```bash
unzip DGraphFin.zip -d data/raw/dgraphfin/
cp data/raw/dgraphfin/dgraphfin.npz data/raw/dgraphfin.npz
```

Expected file:

```text
data/raw/dgraphfin.npz
```

## Preprocessing

```bash
python preprocess.py --dataset amazon --seed 42
python preprocess.py --dataset yelpchi --seed 42
python preprocess.py --dataset tfinance --seed 42
python preprocess.py --dataset dgraphfin --seed 42
```

Use a custom source path:

```bash
python preprocess.py \
  --dataset dgraphfin \
  --raw-path /path/to/dgraphfin.npz \
  --seed 42
```

Overwrite an existing processed graph:

```bash
python preprocess.py --dataset amazon --seed 42 --force
```

Generated files:

```text
data/processed/gpa_unified/*.dgl
data/processed/gaap_unified/*.dgl
data/splits/*_stratified_80_10_10_seed42.npz
data/splits/*_stratified_80_10_10_seed42.json
```

## Training

```bash
python train.py --dataset amazon --device cuda:0 --seed 42 --split-seed 42
python train.py --dataset yelpchi --device cuda:0 --seed 42 --split-seed 42
python train.py --dataset tfinance --device cuda:0 --seed 42 --split-seed 42
python train.py --dataset dgraphfin --device cuda:0 --seed 42 --split-seed 42
```

Use a specified configuration file:

```bash
python train.py \
  --dataset amazon \
  --config configs/amazon.yaml \
  --device cuda:0 \
  --seed 42 \
  --split-seed 42
```

Specify output paths:

```bash
python train.py \
  --dataset amazon \
  --device cuda:0 \
  --seed 42 \
  --split-seed 42 \
  --checkpoint outputs/checkpoints/rudg_amazon_seed42.state.pt \
  --output outputs/validation/rudg_amazon_seed42.json
```

## Testing

```bash
python test.py \
  --dataset amazon \
  --checkpoint outputs/checkpoints/rudg_amazon_seed42.state.pt \
  --device cuda:0 \
  --seed 42 \
  --split-seed 42
```

```bash
python test.py --dataset yelpchi \
  --checkpoint outputs/checkpoints/rudg_yelpchi_seed42.state.pt \
  --device cuda:0 --seed 42 --split-seed 42

python test.py --dataset tfinance \
  --checkpoint outputs/checkpoints/rudg_tfinance_seed42.state.pt \
  --device cuda:0 --seed 42 --split-seed 42

python test.py --dataset dgraphfin \
  --checkpoint outputs/checkpoints/rudg_dgraphfin_seed42.state.pt \
  --device cuda:0 --seed 42 --split-seed 42
```

Use the same configuration file for training and testing:

```bash
python test.py \
  --dataset amazon \
  --config configs/amazon.yaml \
  --checkpoint outputs/checkpoints/rudg_amazon_seed42.state.pt \
  --device cuda:0 \
  --seed 42 \
  --split-seed 42 \
  --output outputs/test/rudg_amazon_seed42.json
```

Generated files:

```text
outputs/test/rudg_<dataset>_seed<seed>.json
outputs/test/rudg_<dataset>_seed<seed>_predictions.npz
```

## Multiple Seeds

```bash
python train.py --dataset amazon --device cuda:0 --seed 42 --split-seed 42
python test.py --dataset amazon --device cuda:0 --seed 42 --split-seed 42 \
  --checkpoint outputs/checkpoints/rudg_amazon_seed42.state.pt

python train.py --dataset amazon --device cuda:0 --seed 3407 --split-seed 42
python test.py --dataset amazon --device cuda:0 --seed 3407 --split-seed 42 \
  --checkpoint outputs/checkpoints/rudg_amazon_seed3407.state.pt
```

## Configurations

```text
configs/amazon.yaml
configs/yelpchi.yaml
configs/tfinance.yaml
configs/dgraphfin.yaml
```

## License

The source code is released under the MIT License. Dataset licenses and terms are provided by the corresponding dataset owners.
