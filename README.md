[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://scy639.github.io/HoMoF.github.io/) 

Official implementation of the paper **"HUMOF: Human Motion Forecasting in Interactive Social Scenes"** (ICLR 2026). 


## TODO
- [x] GTA-IM
- [x] Humanise
- [ ] HOI-M3
- [x] HIK


## Environment setup

```bash
conda create -n "humof" python=3.9.19 -y
conda activate humof
pip install -r requirements.txt
```

## Dataset preprocess


#### GTA-IM dataset

1. Download and unzip the raw GTA-IM dataset (see [STAG](https://github.com/L-Scofano/STAG)). Each recording folder (named `2020-...`) contains `info_frames.pickle`, `info_frames.npz`, `realtimeinfo.gz`, plus per-frame `{i:05d}.jpg` (rgb), `{i:05d}.png` (depth) and `{i:05d}_id.png` (instance-id map):

```
<gta_raw_dir>
├── 2020-05-20-21-13-13/
│   ├── info_frames.pickle
│   ├── info_frames.npz
│   ├── realtimeinfo.gz
│   ├── 00000.jpg
│   ├── 00000.png
│   ├── 00000_id.png
│   └── ...
├── 2020-05-21-13-54-43/
└── ...
```

2. Run the preprocessing:

```bash
python datasets/dataset_preprocess/gtaim/preprocess.py -pa=<gta_raw_dir>
```

For each recording `<rec>` in room `r<NNN>`, the preprocess.py produces in `./data/GTA-IM_Dataset/processed/data_v2_downsample0.02_fix/`:
- `<rec>_r<NNN>.npy`: scene point cloud — RGBD fusion of every 10th frame with the person masked out, voxel-downsampled at 0.01m per frame then 0.02m globally;
- `<rec>_r<NNN>_sf<k>.npy`: motion chunks, `joints_3d_world` split every 1000 frames, shape `[T<=1000, 21, 3]`.


#### Humanise dataset

1. Download raw humanise dataset (following the instructions at https://github.com/Silverster98/HUMANISE) to prepare data

2. 
```bash
cd datasets/dataset_preprocess/humanise
python preprocess.py \
--humanise_dir=<humanise_dir> \
--scene_dir=<scene_dir> \
--smplx_dir=<smplx_dir>
```

Where `<humanise_dir>`, `<scene_dir>`, and `<smplx_dir>` refer to the data from Step 1. `<smplx_dir>` is the folder containing the `smplx/` subfolder with the `.npz` files (`smplx.create()` appends `smplx/` automatically). This produces `./data/humanise/processed_/` (joints `{mid}.npy`, scenes `{scene_id}.xyz`, `annotation.csv`).

Expected input directory structure:

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

1. Setup conda env SAST:

```bash
cd datasets/dataset_preprocess/hik
git clone https://github.com/felixbmuller/SAST.git
cd SAST
conda create -n "SAST" python=3.10
conda activate SAST
pip install -r requirements.txt
pip install numpy==1.24.3  # must match the numpy version in env humof
```

2. Download the [Humans in Kitchens](https://github.com/jutanke/hik/tree/main) dataset and unpack its contents into `./data/` (note that `.` is `datasets/dataset_preprocess/hik/SAST` now), so that `./data/` contains `poses/`, `scenes/`, and `body_models/`.

3. Run preprocessing:

```bash
cd .. # datasets/dataset_preprocess/hik/SAST -> datasets/dataset_preprocess/hik
conda activate SAST
# Overwrite the installed official `hik/data/scene.py` with our modified one (`hik_patch/scene.py`)
python -c "import hik, os, shutil; shutil.copy('hik_patch/scene.py', os.path.join(os.path.dirname(hik.__file__), 'data', 'scene.py')); print('Modified:', hik.__file__)"
# preprocessing
python preprocess.py hik hik_shortterm.yaml
```

## Train & Eval

Test (modify `model_path` in `conf.py` to use a different checkpoint; by default it points to the pre-trained weights):
```bash
# Download Pre-trained weights from Hugging Face https://huggingface.co/scy639/HUMOF :
pip install "huggingface_hub[cli]"
huggingface-cli download scy639/HUMOF --include "*.pth" --local-dir checkpoints
# infer:
conda activate humof
python main.py <dataset>  # <dataset> = gta | humanise | hoi | hik
```

Train:
```bash
conda activate humof
python main.py <dataset> --train
```


## Citation

```bibtex
@inproceedings{sun2026humof,
  title={HUMOF: Human motion forecasting in interactive social scenes},
  author={Caiyi Sun and Yujing Sun and Xiao Han and Zemin Yang and Jiawei Liu and Xinge Zhu and Siu Ming Yiu and Yuexin Ma},
  booktitle={International Conference on Learning Representations},
  volume={2026},
  pages={6511--6534},
  year={2026}
}
```


