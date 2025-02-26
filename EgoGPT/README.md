# 🤖🧠 EgoGPT:

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/XXXX.XXXXX)
[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github)](https://github.com/egolife-ntu/EgoLife)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging_Face-Model-blue.svg)](https://huggingface.co/lmms-lab)
[![data](https://img.shields.io/badge/EgoGPT-Data-orange)](https://huggingface.co/lmms-lab) 

EgoGPT is a vision-audio-language model trained on egocentric datasets, achieving state-of-the-art performance on egocentric video understanding. 

<div align="center"><img src="images/egogpt_model.png" width="75%"/></div>

## 📢 News

- 🚀[2025/2/27] EgoGPT training and inference code is now available on Github!

## 🚀Coming Soon

- [ ] Evaluation code on omni-modal benchmarks
- [x] Gradio Demo
- [x] Training Data (Video, Audio, Cross-Modality)


## Installation

1. Clone this repository.

```shell
git clone https://github.com/egolife-ntu/EgoLife
cd EgoLife/EgoGPT
```

2. Install the dependencies.

```shell
conda create -n egogpt python=3.10
conda activate egogpt
pip install --upgrade pip
pip install -e .

3. Install the dependencies for training and inference.

```shell
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```


## Quick Start

### Setup

1. Download EgoGPT-7b-EgoIT-EgoLife from 🤗[Huggingface](https://huggingface.co/lmms-lab/EgoGPT-7b-EgoIT-EgoLife).

```shell
wget https://huggingface.co/lmms-lab/EgoGPT-7b-EgoIT-EgoLife/resolve/main/EgoGPT-7b-EgoIT-EgoLife.tar.gz
tar -xzvf EgoGPT-7b-EgoIT-EgoLife.tar.gz
```

### Training

```shell
bash scripts/train_egogpt.sh
```

### Inference

```shell
python inference.py --pretrained_path checkpoints/EgoGPT-7b-EgoIT-EgoLife --video_path data/train/A1_JAKE/DAY1/DAY1_A1_JAKE_11223000.mp4 --audio_path audio/DAY1_A1_JAKE_11223000.mp3 --query "Please describe the video in detail."
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
Our code is released under the Apache-2.0 License.
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