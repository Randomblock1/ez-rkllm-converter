# ez-rklllm-converter

Convert HuggingFace models to RKLLM format and upload to HuggingFace.

## Usage

This repo is set up for ROCM, modify pyproject.toml if you want to use CUDA. Just remove the ROCM related packages.

I recommend using `uv` to manage and run from a virtual environment.

Simply edit the bottom main.py (the congfiuration section) to specify the model you want to convert and run `uv run main.py`. You can also set quantization parameters there.
