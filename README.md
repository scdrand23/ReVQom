# ReVQom: Residual Vector Quantization for Communication-Efficient Multi-Agent Perception

Official implementation of **ReVQom** (ICASSP 2026).

## Abstract

Multi-agent collaborative perception (CP) improves scene understanding by sharing information across connected agents such as autonomous vehicles, unmanned aerial vehicles, and robots. Communication bandwidth, however, constrains scalability. We present ReVQom, a learned feature codec that preserves spatial identity while compressing intermediate features. ReVQom is an end-to-end method that compresses feature dimensions via a simple bottleneck network followed by multi-stage residual vector quantization (RVQ). This allows only per-pixel code indices to be transmitted, reducing payloads from 8192 bits per pixel (bpp) of uncompressed 32-bit float features to 6-30 bpp per agent with minimal accuracy loss. On DAIR-V2X real-world CP dataset, ReVQom achieves 273x compression at 30 bpp to 1365x compression at 6 bpp. At 18 bpp (455x), ReVQom matches or outperforms raw-feature CP, and at 6-12 bpp it enables ultra-low-bandwidth operation with graceful degradation.

## Method

Each agent extracts BEV features with a sparse voxel encoder, applies a 1x1 bottleneck (channel reduction ratio `C_rr`) and `n_q`-stage residual vector quantization, then transmits only per-pixel code indices:

```
k_i = argmin_k || r_i - e_k ||^2        r_{i+1} = r_i - e_{k_i}
bitrate = n_q * log2(K) bits per spatial location
```

Codebooks are EMA-updated (decay 0.8) and pre-shared, so the receiver reconstructs features by codebook lookup and channel expansion before multi-agent fusion (CoBEVT backbone).

## Installation

```bash
conda env create -f environment.yaml
conda activate revqom

cd revqom/pcdet_utils && python setup.py build_ext --inplace
```

## Dataset Preparation

### DAIR-V2X

Download the cooperative-vehicle-infrastructure split from [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X) and set the paths in `revqom/hypes_yaml/dairv2x/*.yaml`:

```yaml
data_dir: "/path/to/dataset/DAIR-V2X/cooperative-vehicle-infrastructure"
root_dir: "/path/to/dataset/DAIR-V2X/cooperative-vehicle-infrastructure/train.json"
validate_dir: "/path/to/dataset/DAIR-V2X/cooperative-vehicle-infrastructure/val.json"
test_dir: "/path/to/dataset/DAIR-V2X/cooperative-vehicle-infrastructure/val.json"
```

### OPV2V

Download from [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/) and set the paths in `revqom/hypes_yaml/opv2v/*.yaml`:

```yaml
root_dir: "/path/to/dataset/OPV2V/train"
validate_dir: "/path/to/dataset/OPV2V/validate"
test_dir: "/path/to/dataset/OPV2V/test"
```

## Usage

Set `PYTHONPATH` to the repository root before running:

```bash
export PYTHONPATH=.
```

### Training

```bash
python revqom/tools/train.py --hypes_yaml revqom/hypes_yaml/dairv2x/revqom_k64.yaml

python revqom/tools/train.py --hypes_yaml revqom/hypes_yaml/opv2v/revqom_k64.yaml
```

Codebook-size variants: `revqom_k4.yaml`, `revqom_k16.yaml`, `revqom_k64.yaml`, `revqom_k256.yaml`, `revqom_k1024.yaml`. Ablation configs (EMA decay, `n_q`, `C_rr`) are under `revqom/hypes_yaml/dairv2x/ablations/`.

### Inference

```bash
python revqom/tools/inference.py --model_dir checkpoints/revqom_s_k64_dairv2x --fusion_method intermediate
```

### RVQ Codebook Visualization

```bash
python revqom/tools/inference_with_rvq_vis.py --model_dir checkpoints/revqom_s_k64_dairv2x --fusion_method intermediate --visualize_codebook
```

## Results

3D vehicle detection AP@0.3/AP@0.5 (from the paper):

### DAIR-V2X

| Method | K | bpp | AP@0.3 | AP@0.5 | Compression |
|--------|---|-----|--------|--------|-------------|
| No Collaboration | - | 0 | 0.589 | 0.544 | - |
| F-Cooper | - | 8192 | 0.704 | 0.648 | 1x |
| V2VNet | - | 4096 | 0.695 | 0.635 | 2x |
| AttFuse | - | 2048 | 0.697 | 0.638 | 4x |
| CoBEVT | - | 8192 | 0.728 | 0.657 | 1x |
| V2X-ViT | - | 6144 | 0.745 | 0.676 | 1.3x |
| Where2comm | - | 512 | 0.701 | 0.634 | 16x |
| ReVQom-u | 4 | 6 | 0.690 | 0.558 | 1365x |
| ReVQom-T | 16 | 12 | 0.699 | 0.609 | 683x |
| **ReVQom-S** | 64 | 18 | 0.747 | 0.651 | 455x |
| **ReVQom-M** | 256 | 24 | 0.753 | 0.666 | 341x |
| ReVQom-L | 1024 | 30 | 0.725 | 0.636 | 273x |

### OPV2V

| Method | K | bpp | AP@0.3 | AP@0.5 | Compression |
|--------|---|-----|--------|--------|-------------|
| CoBEVT | - | 8192 | 0.947 | 0.895 | 1x |
| ReVQom-S | 64 | 18 | 0.946 | 0.869 | 455x |

## Pretrained Checkpoints

Pretrained weights are hosted on Hugging Face at [scdrand23/ReVQom](https://huggingface.co/scdrand23/ReVQom). Download them into `checkpoints/`:

```bash
pip install -U huggingface_hub
hf download scdrand23/ReVQom --local-dir checkpoints
```

Each checkpoint ships with its exact training `config.yaml`:

| Checkpoint | Config | Dataset | Eval AP@0.3/AP@0.5/AP@0.7 |
|------------|--------|---------|---------------------------|
| `checkpoints/revqom_s_k64_dairv2x` | K=64, n_q=3, C_rr=16, EMA 0.8, epoch 30 | DAIR-V2X | 0.751 / 0.646 / 0.375 |

The ReVQom-S evaluation matches the experiment records for the paper's K=64 configuration. Additional checkpoints (ReVQom-M, K=256) will be added after re-evaluation.

Before running inference, update the dataset paths in each checkpoint's `config.yaml` to point to your local dataset.

## Citation

```bibtex
@inproceedings{shenkut2026revqom,
  title={Residual Vector Quantization for Communication-Efficient Multi-Agent Perception},
  author={Shenkut, Dereje and Kumar, B.V.K. Vijaya},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```

## Acknowledgments

This codebase builds on [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD), [CoBEVT](https://github.com/DerrickXuNu/CoBEVT), and [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch).

## License

MIT License
