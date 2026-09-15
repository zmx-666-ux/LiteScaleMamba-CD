# LiteScaleMamba-CD

Anonymous implementation for binary remote sensing change detection on LEVIR-CD, SYSU-CD and WHU-CD.

![Model overview](figures/overview.png)

The repository contains the final hybrid backbone, hybrid decoder and change detection model, together with one training entry point and one testing entry point. The testing script evaluates the main prediction from a specified checkpoint and prints the resulting metrics.

## Repository layout

```text
models/                    Hybrid_backbone.py, Hybrid_decoder.py, ChangeHybridBCD.py
datasets/                  augmentations.py, change_detection.py
configs/hybrid.yaml        Model settings used by the provided scripts
utils/                     Binary metrics, model construction and Lovasz-Softmax loss
vendor/                    VMamba VSS block implementation
third_party/selective_scan CUDA selective scan extension source
splits/SYSU-CD/            SYSU-CD train, validation and test filename lists
figures/                   PDF figures and PNG previews
train.py                   Training and validation
test.py                    Testing and optional change-map export
```

## Environment

Use Python 3.10 or later on Linux with an NVIDIA GPU and a CUDA toolkit compatible with the installed PyTorch build. The bundled selective scan extension requires CUDA 11.6 or later. Install a CUDA-enabled PyTorch build using the [official installation selector](https://pytorch.org/get-started/locally/), then run:

```bash
python -m pip install -r requirements.txt
python -m pip install ./third_party/selective_scan
```

The MobileNetV2 and VSS stages were initialized from ImageNet-pretrained checkpoints in the reported experiments. Pass those checkpoint paths to `train.py` with `--mobilenet-pretrained` and `--vssm-pretrained`. The repository does not contain those pretrained checkpoints or trained dataset checkpoints.

## Dataset preparation

Download [LEVIR-CD](https://justchenhao.github.io/LEVIR/), [SYSU-CD](https://github.com/liumency/SYSU-CD) and the [WHU building change detection data](https://gpcv.whu.edu.cn/data/building_dataset.html) from their dataset maintainers. Prepare non-overlapping 256 × 256 patches for the same split used in the paper. Each dataset has three directories:

```text
DATASET/
  train/
    A/         image_t1.png
    B/         image_t1.png
    label/     image_t1.png
  val/
    A/ B/ label/
  test/
    A/ B/ label/
```

Supply one UTF-8 text file per split with one filename per line. Images in `A/` and `B/` are loaded as RGB; labels may use 0/1 or 0/255 binary values. The paper reports 7,120/1,024/2,048 patches for LEVIR-CD, 12,000/4,000/4,000 for SYSU-CD and 5,947/743/744 for WHU-CD (train/validation/test). SYSU-CD filename lists with those counts are included under `splits/SYSU-CD/`; each list refers to its own split directory. The LEVIR-CD and WHU-CD split lists need to be supplied to reproduce the paper's split.

## Training

Train one model per dataset. This example uses LEVIR-CD:

```bash
python train.py \
  --dataset LEVIR-CD \
  --train-dir /path/to/LEVIR-CD/train --train-list /path/to/LEVIR-CD/train.txt \
  --val-dir /path/to/LEVIR-CD/val --val-list /path/to/LEVIR-CD/val.txt \
  --mobilenet-pretrained /path/to/mobilenet_v2.pth \
  --vssm-pretrained /path/to/vssm_tiny.pth \
  --seed 42
```

The defaults are 150 epochs, batch size 16, AdamW, learning rate 1e-4 and weight decay 5e-4. The script uses ImageNet normalization, synchronized spatial transformations, independent photometric augmentation, deep supervision, a difference-prior loss, cosine learning-rate decay and an optional exponential moving average. After every epoch, validation F1 selects `checkpoints/LEVIR-CD/seed_42/best.pth`. Use different seeds and separate model runs when computing a multi-run mean.

For SYSU-CD or WHU-CD, change `--dataset` and supply that dataset's train/validation paths and lists. Dataset-specific settings such as `--pos-weight-seg` should match the final experiment configuration.

## Testing

```bash
python test.py \
  --dataset LEVIR-CD \
  --test-dir /path/to/LEVIR-CD/test --test-list /path/to/LEVIR-CD/test.txt \
  --checkpoint checkpoints/LEVIR-CD/seed_42/best.pth
```

The script prints F1, IoU, precision, recall, overall accuracy and Kappa computed from all valid test pixels. It loads the checkpoint strictly, so weights from a different architecture cannot silently produce a result. Add `--save-dir /path/to/maps` to save binary change maps as PNG files. The validation-selected checkpoint is the only checkpoint used in a test run.

## Figures and reported results

The supplied figures are available as PDF and PNG: [overview](figures/overview.pdf), [difference-guided wavelet bridge](figures/wavelet_bridge.pdf), [refinement head](figures/refinement.pdf) and [LEVIR-CD error maps](figures/levir_error_maps.pdf).

The paper reports the following means ± standard deviations over three independent runs; these are manuscript values and require the final checkpoints and exact splits for independent reproduction:

| Dataset | F1 (%) | IoU (%) |
| --- | ---: | ---: |
| LEVIR-CD | 91.91 ± 0.08 | 85.03 ± 0.23 |
| WHU-CD | 94.13 ± 0.22 | 89.07 ± 0.19 |
| SYSU-CD | 83.57 ± 0.28 | 71.70 ± 0.42 |

## License and credits

The MambaCD-derived code uses Apache-2.0. VMamba-derived components and the Lovasz-Softmax loss carry MIT notices. See [THIRD_PARTY.md](THIRD_PARTY.md) for the origins and permission notices.

Run the lightweight checks with `python -m unittest discover -s tests -v`. Full training and model inference require the CUDA environment and dataset files described above.
