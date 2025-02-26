# 🤖🧠 EgoGPT:

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/XXXX.XXXXX)
[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github)](https://github.com/egolife-ntu/EgoLife)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging_Face-Model-blue.svg)](https://huggingface.co/lmms-lab)
[![ModelScope](https://img.shields.io/badge/ModelScope-Model-blue.svg)](https://huggingface.co/lmms-lab)
**Training Data:** [![data](https://img.shields.io/badge/EgoGPT-Data-orange)](https://huggingface.co/lmms-lab) 


EgoGPT is a personalized AI model built upon GPT-4o. It supports adaptive and context-aware interactions, generating responses tailored to individual user preferences.

<div align="center"><img src="images/egogpt_model.png" width="75%"/></div>

## 📢 News

- 🚀[2025/2/27] EgoGPT training and inference code is now available on Github!

## 🚀Coming Soon

- [ ] Evaluation code on omni-modal benchmarks
- [x] Gradio Demo
- [x] Training Data (Video, Audio, Cross-Modality)


## Install

1. Clone this repository.

```shell
git clone https://github.com/egolife-ntu/EgoGPT
cd EgoGPT
```


## Quick Start

1. Download the `EgoGPT-Model` from 🤗[Huggingface](https://huggingface.co/egolife-ntu/EgoGPT).

```python
from huggingface_hub import snapshot_download

# Define the model repository and the target directory
model_name = "egolife-ntu/EgoGPT-Model"
local_dir = "./models"

# Download the model to the local directory
snapshot_download(repo_id=model_name, local_dir=local_dir)
```

2. Download additional resources if needed.

```shell
# Example command to download additional resources
wget https://example.com/resource -P resources/
```

## Demo


## ✅ TODO List

- [x] Release all the model weights.
- [x] Provide a Gradio demo for interaction.
- [x] Release training and inference scripts.
- [x] Publish evaluation code for various benchmarks.
- [ ] Enhance model with additional personalization features.
- [ ] Explore integration with more datasets for improved adaptability.

## 📃 Main Results

### Results on Personalized Interaction

<p align="center" width="100%">
<img src="images/results_personalized_interaction.png" alt="results_personalized_interaction.png" width=80%>
</p>

## LICENSE

Our code is released under the Apache-2.0 License. Our model, as it is built on GPT-4o, is required to comply with the [GPT-4o License](https://gpt.meta.com/gpt4o/license/).

## Acknowledgements

- [LLaVA](https://github.com/LLaVA-VL/LLaVA-NeXT): Our codebase is conducted on LLaVA.
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval): Our evaluation system are built on lmms-eval.

## Citation

If our work is useful for you, please cite as:

```
@article{EgoLife,
  title={EgoLife: Towards Egocentric Life Assistant},
  author={The EgoLife Team},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```