# SCVLM: Smart Contract Vulnerability Detection via LLM with Expert Knowledge Integration

This repository contains the implementation of **SCVLM**, a dual-branch framework for detecting smart contract vulnerabilities, including **Reentrancy** and **Timestamp Dependency**.

## Framework Overview

![Framework](./Figure_1_The%20overall%20framework%20of%20SCVLM.pdf)

SCVLM consists of two branches:
- **Graph Branch**: UniXcoder/RoBERTa encoder with DFG+CFG-guided attention masks
- **LLM Branch**: CodeGemma-7b fine-tuned with LoRA, augmented by expert knowledge label embeddings
- **Fusion**: Weighted voting (`prob = 0.65 × LLM + 0.35 × Graph`)

The key innovation is the **Flexible Feature Fusion (FFC)** mechanism, which injects graph-guided code representations into the LLM's LoRA adaptation pathway via `UnixLoraLinear`.

---

## Requirements

```bash
pip install torch transformers peft bitsandbytes scikit-learn tree-sitter tqdm scipy
```

Python 3.8+, CUDA recommended.

---

## Model Downloads

Download the following pre-trained models from HuggingFace and place them under `./models/`:

| Model | HuggingFace ID | Local Path |
|-------|----------------|------------|
| CodeGemma-7b | `google/codegemma-7b` | `./models/code_gemma` |
| UniXcoder | `microsoft/unixcoder-base` | `./models/unixcoder` |

---

## Tree-sitter Parser Setup

The Solidity parser (`./tools/my-languages.so`) must be compiled before use:

```bash
cd tools
python build.py
```

This builds `my-languages.so` from the `tree-sitter-solidity` grammar. Ensure `tree-sitter-solidity` is present in `tools/`.

---

## Dataset

Datasets are located in `./data/`:

```
data/
├── reen/          # Reentrancy vulnerability dataset
│   ├── train.jsonl
│   ├── eval.jsonl
│   └── test.jsonl
└── time/          # Timestamp dependency dataset
    ├── train.jsonl
    ├── eval.jsonl
    └── test.jsonl
```

Each `.jsonl` file contains one sample per line:
```json
{"label": "1", "contract": "pragma solidity ...", "idx": 0, "file": "example.sol"}
```
- `label`: `"1"` = vulnerable, `"0"` = safe
- `contract`: full Solidity source code

---

## Configuration

Edit `config.py` to switch between vulnerability types:

```python
# For reentrancy:
train_data_file = './data/reen/train.jsonl'

# For timestamp dependency:
# train_data_file = './data/time/train.jsonl'
```

Set model paths:
```python
# LLM backbone
parser.add_argument('--model_path', default='./models/code_gemma')

# UniXcoder (graph branch)
parser.add_argument('--model_name_or_path', default='./models/unixcoder')

# Checkpoint save/load path
parser.add_argument('--save_model_path', default='./checkpoints/llm')
parser.add_argument('--output_dir', default='./checkpoints/graph')
```

---

## Training

### Step 1: Train the Graph Branch (UniXcoder)

```bash
python train_roberta.py
```

Trains the DFG+CFG-guided graph model. Saves checkpoint to `./checkpoints/graph/model.bin`.

### Step 2: Train the LLM Branch (CodeGemma + LoRA)

```bash
python train_llm.py
```

Trains CodeGemma with LoRA fine-tuning and expert knowledge label embeddings. Saves adapter to `./checkpoints/llm/`.

---

## Inference

### Single-branch evaluation

```bash
# Graph branch only
python train_roberta.py   # (test mode, set stage='test' in main())

# LLM branch only
python train_llm.py       # (test mode)
```

### Ensemble inference (both branches)

```bash
python Main.py
```

Loads both trained models and performs weighted voting fusion.

---

## File Structure

```
├── config.py             # Hyperparameters and path configuration
├── model.py              # Model definitions: UniXcoder, UnixLoraLinear, load_llm()
├── train_roberta.py      # Graph branch training and evaluation
├── train_llm.py          # LLM branch training and evaluation
├── all_dataload.py       # Dual-branch joint data loader
├── llm_dataload.py       # LLM-only data loader
├── Main.py               # Ensemble inference script
├── irlora_utils.py       # IRQLora (4-bit quantization + LoRA) utilities
├── pattern_extractor/
│   ├── Pattern_reen.py   # Reentrancy expert pattern extraction (AST + rules)
│   └── pattern_time.py   # Timestamp expert pattern extraction
├── tools/
│   ├── DFG.py            # Data-flow graph extraction for Solidity
│   ├── utils.py          # Tree-sitter utilities
│   └── build.py          # Build Solidity parser (.so)
└── data/
    ├── reen/             # Reentrancy dataset
    └── time/             # Timestamp dataset
```

---

## Expert Pattern Extraction

Pattern extraction is fully automated via AST parsing (`tree-sitter-solidity`) and rule-based matching. For each contract, 5 label slots are generated:

| Slot | Reentrancy | Timestamp Dependency |
|------|-----------|---------------------|
| 1 | `safe` (fixed context token) | `safe` |
| 2 | `reentrancy` | `timestamp dependency` |
| 3 | `keyword` / `None` | `keyword` / `None` |
| 4 | `deduction` / `None` | `assignment` / `None` |
| 5 | `checkpoint` / `None` | `return` / `None` |

`None` is inserted as a placeholder to maintain fixed-length prefix (C=5).

---

## Citation

If you use this code, please cite our paper:

```bibtex
@article{scvlm2024,
  title={SCVLM: Smart Contract Vulnerability Detection via Large Language Models with Expert Knowledge Integration},
  author={...},
  journal={...},
  year={2024}
}
```

---

## License

This project is released under the MIT License.
