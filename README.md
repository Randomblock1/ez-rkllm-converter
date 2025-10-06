# ez-rklllm-converter

Convert HuggingFace models to RKLLM format and upload to HuggingFace.

## Usage

This repo is set up for ROCM, modify pyproject.toml if you want to use CUDA. Just remove the ROCM related packages.

I recommend using `uv` to manage and run from a virtual environment.

### Interactive Mode (TUI)

Run the converter with an interactive Text User Interface:

```bash
uv run main.py
```

This will launch a TUI where you can configure:
- Model IDs to convert
- Target platform (rk3576, rv1126b, rk3588, rk3562)
- NPU cores (defaults to highest available for the platform)
- Quantization types (platform-specific)
- Hybrid rates (0.0-1.0)
- Optimization levels (0 or 1)
- Context lengths (up to 16k)

After configuration, a confirmation screen will show:
- Total number of model files to be generated
- Summary of all settings
- Options to confirm, edit, or cancel

### Non-Interactive Mode (CLI)

For scripting or environments without interactive shells, pass arguments directly:

```bash
uv run main.py \
  --model-ids Qwen/Qwen3-4B-Thinking-2507 \
  --platform rk3588 \
  --qtypes w8a8,w8a8_g128 \
  --hybrid-rates 0.0,0.2,0.4 \
  --optimizations 0,1 \
  --context-lengths 4k,16k \
  --npu-cores 1,2,3
```

Use `--help` to see all available options:

```bash
uv run main.py --help
```

### Platform-Specific Configuration

#### Quantization Types

- **RK3576 / RV1126B**: w4a16, w4a16_g32, w4a16_g64, w4a16_g128, w8a8
- **RK3588**: w8a8, w8a8_g128, w8a8_g256, w8a8_g512
- **RK3562**: w8a8, w4a16_g32, w4a16_g64, w4a16_g128, w4a8_g32

#### NPU Cores

The number of NPU cores determines inference performance. More cores generally means better performance.

- **RK3576**: 1 or 2 cores (default: 2)
- **RV1126B**: 1 core (default: 1)
- **RK3588**: 1, 2, or 3 cores (default: 3)
- **RK3562**: 1 core (default: 1)

The converter will validate that the selected quantization types and NPU core counts are compatible with the chosen platform.

### Upload Behavior

Models are now uploaded immediately after each export to save disk space. This means:
- No need to wait for all conversions to complete before uploading
- Local exported files are cleaned up after successful upload
- Better for large batches with many variants
