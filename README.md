# Intelligent Curriculum CNN Steganography — Final V9.3 Model

This repository contains the GitHub-ready reproducibility materials for the **final V9.3 detector-aware model** used in the paper:

**An Intelligent Curriculum CNN Framework for Secure and Enhanced Image-in-Image Steganography**

## Included files

This package includes only materials directly related to the final V9.3 model:

- `scripts/train_v93_soft_hard_sample_finetune.py` — V9.3 soft hard-sample fine-tuning code.
- `configs/config_v93.json` — configuration of the V9.3 run.
- `hard_samples/` — hard-sample indices/results used by V9.3 from the V9.2 mining stage.
- `results/v93_internal/` — final V9.3 internal detector-aware summaries and training log.
- `results/embedding_activity/` — V9.3 embedding-activity statistics.
- `figures/v93/` — V9.3 visual examples and embedding-activity figures.
- `logs/v93_training_console_log.txt` — console log for the V9.3 training run.
- `checkpoints_manifest/` — names of large checkpoint files that are not included.

## Not included

The earlier V9.1 notebook, V9.2 training code, and separate V7D+/V9.3 robustness notebook are not included here because this cleaned repository is intended to document the final V9.3 model only.

Large pretrained `.pt` checkpoint files are not included in this GitHub package due to file-size limitations. They can be uploaded separately using Git LFS or provided upon reasonable request.

## Dataset

The experiments use a COCO-based image dataset. Images are converted to RGB format and resized to 256 × 256 pixels. Cover-secret pairs are stored in train, validation, and test folders.

Expected dataset folder structure:

```text
coco_20k_stego_pairs/
  train/covers/
  train/secrets/
  val/covers/
  val/secrets/
  test/covers/
  test/secrets/
```

The dataset itself is not redistributed in this repository.

## V9.3 training summary

The V9.3 model uses soft hard-sample fine-tuning. It starts from the best V9.1 checkpoint and uses hard samples mined by V9.2 with soft weighting. The main goal is to improve detector-aware security while preserving cover imperceptibility and secret recovery quality.

Main settings:

- Image size: 256 × 256 × 3
- Batch size: 8
- Epochs: 30
- Patience: 10
- Alpha range: 0.054–0.060
- Hard sample weight: 2.0
- Normal sample weight: 1.0
- Detector-aware losses: XuNet-style and SRNet-style detector pressure

## Main V9.3 reported results

The final V9.3 internal detector-aware evaluation reports:

- Cover PSNR-Y: 67.34 dB
- Secret PSNR-Y: 31.03 dB
- BER: approximately 0.032
- XuNet-style detection: 4.10%
- SRNet-style detection: 0.20%
- Average internal detector-aware detection: 2.15%

## How to use

1. Prepare the COCO-based paired dataset using the expected folder structure.
2. Place the required starting checkpoint path according to the script configuration.
3. Run:

```bash
python scripts/train_v93_soft_hard_sample_finetune.py
```

The script was originally written for Google Colab with Google Drive paths. Update the path variables at the beginning of the script if running locally.

## Code availability

The V9.3 training code, configuration file, hard-sample indices, evaluation summaries, embedding-activity outputs, figures, and checkpoint manifest are provided to improve reproducibility and transparency. Full pretrained model weights are not included due to file-size limitations.
