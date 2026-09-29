# Flea Beetle Damage Assessment

Lab Vision Systems (MA-INF 4308), University of Bonn

## Authors

- Emrullah Dagkusu (Matr.-Nr. 50268560)
- Federico Rosatelli (Matr.-Nr. 50357754)

## Project

The final project is in [`FINAL/`](FINAL/): code in `src/` and `scripts/`, configs in `configs/`,
results in `outputs/`, and the LaTeX report in `report/`. Build the report with
`cd FINAL/report && pdflatex main && bibtex main && pdflatex main && pdflatex main`.
A full command reference is in [`FINAL/analyses/COMMANDS.md`](FINAL/analyses/COMMANDS.md).

## Setup

```bash
cd FINAL
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Model weights

The weights are not tracked in git. Download them from our Google Drive folder:
https://drive.google.com/drive/folders/15my71BUyC346ksATji6mAB_NKftLYRYn?usp=share_link

| File | Place it at (inside `FINAL/`) |
| :--- | :--- |
| `dinov3-vits16-hf.zip` (DINOv3 ViT-S/16 backbone, 80 MB) | unzip into `weights/`, giving `weights/dinov3-vits16-hf/` |
| `checkpoint_best_total.pth` (RF-DETR hole/pitting detector, 116 MB) | `outputs/rfdetr_hole_pitting/checkpoint_best_total.pth` |

The DINOv3 folder must contain `config.json` and `model.safetensors`. It is the Hugging Face
checkpoint `facebook/dinov3-vits16-pretrain-lvd1689m` and can also be downloaded from there.
The trained MIL and whole-image checkpoints are already in `outputs/runs/`.

## Web demo

```bash
cd FINAL
streamlit run scripts/app.py
```

A recorded walkthrough of the demo (`Video_Lab_Audio.mpg`) is in the same Google Drive folder.

The live demo reads the field images from the dataset folder. Set `CSFB_DATASET_DIR` to the
`RSFB-Phenotyping_training_set` image folder if it is not at the default location.
