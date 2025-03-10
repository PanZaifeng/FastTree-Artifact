# End-to-end SGLang benchmarks

This directory contains the end-to-end benchmark codes used in our paper. We implement FastTree as an SGLang plugin (compatible with v0.2.13). The plugin intercepts specific SGLang functions and replaces them to enable FastTree optimizations.

For benchmarking, we use the system prompt from [BigPromptLibrary](https://github.com/0xeb/TheBigPromptLibrary/blob/main/SystemPrompts/Meta.ai/metaai_llama3-04182024.md) and the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset. Additionally, we adapt code from SGLang’s existing benchmarks and leverage ChatGPT to assist in writing analysis scripts.

## Usage

To set up the experimental environment, we provide a `Dockerfile`, along with a single entry script, `run.sh`, to reproduce the final results.

```bash
./run.sh <your HF token>
```

Please replace `<your HF token>` with your own HuggingFace [token](https://huggingface.co/docs/hub/en/security-tokens) or set the environment variable `$HF_HOME` before running.
