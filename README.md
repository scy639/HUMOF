[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://scy639.github.io/HoMoF.github.io/) 

Official implementation of the paper **"HUMOF: Human Motion Forecasting in Interactive Social Scenes"** (ICLR 2026). 


## TODO
- [ ] Dataset preprocess
    - [ ] GTA-IM
    - [x] Humanise
    - [ ] HOI-M3
    - [x] HIK
- [x] Train & Eval code
- [ ] Pre-trained weights


## Environment setup
```
conda create -n "humof" python=3.9.19 -y
conda activate humof
pip install -r requirements.txt
```

## Dataset preprocess

#### Humanise dataset

1. Download raw humanise dataset: follow https://github.com/Silverster98/HUMANISE to prepare data

2. 
```bash
cd datasets/dataset_preprocess/humanise
python preprocess.py \
--humanise_dir=<humanise_dir> \
--scene_dir=<scene_dir> \
--smplx_dir=<smplx_dir>
```

Where `<humanise_dir>`, `<scene_dir>`, and `<smplx_dir>` refer to the data from Step 1. The expected directory structure is:

```bash
<humanise_dir>
├── motions/
│   ├── 000000.pkl
│   ├── 000001.pkl
│   ├── 000002.pkl
│   ├── 000003.pkl
│   ├── 000004.pkl
│   ├── 000005.pkl
│   ├── 000006.pkl
│   ├── 000007.pkl
│   ├── 000008.pkl
│   ├── 000009.pkl
│   ├── 000010.pkl
│   └── ... (19637 more items)
├── annotation.csv


<scene_dir>
├── <scanId>
│   ├── <scanId>_vh_clean_2.ply
│   └── ... 
├── <scanId>
│   ├── <scanId>_vh_clean_2.ply
│   └── ... 
└── ...


<smplx_dir>
└── smplx/
    ├── SMPLX_FEMALE.npz
    ├── SMPLX_MALE.npz
    ├── SMPLX_NEUTRAL.npz
    └── md5sums.txt
```


#### HIK dataset

```bash
cd datasets/dataset_preprocess/hik
git clone https://github.com/felixbmuller/SAST.git
cd SAST
conda create -n "SAST" python=3.10
conda activate SAST
pip install -r requirements.txt
```

Download the [Humans in Kitchens](https://github.com/jutanke/hik/tree/main) and unpack its content to `data/`, such that `data/` contains `poses/`, `scenes/`, and `body_models/`.

```bash
cd .. # datasets/dataset_preprocess/hik/SAST -> datasets/dataset_preprocess/hik
python preprocess.py hik hik_shortterm.yaml
```

## Train & Eval

Modify the `DATASET_name` field in `conf0.py` and run `python main.py`

