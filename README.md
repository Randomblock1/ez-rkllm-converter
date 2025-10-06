# ez-rkllm-converter

Convert HuggingFace models to RKLLM format and upload to HuggingFace.

## Usage

This repo is set up for ROCM, modify pyproject.toml if you want to use CUDA. Just remove the ROCM related packages.

I highly recommend using `uv` to manage and run a virtualenv.

### Interactive Mode (TUI)

Run the converter with an interactive Text User Interface:

```bash
uv run main.py
```

This will launch a TUI where you can configure:
- Huggingface model IDs to convert
- Target platform (rk3576, rv1126b, rk3588, rk3562)
- NPU cores (defaults to platform max)
- Quantization types
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

Use `--help` to see all available options.
