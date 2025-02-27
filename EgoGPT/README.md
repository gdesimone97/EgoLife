# 🤖🧠 EgoGPT:

**Project Page:** [![EgoLife](https://img.shields.io/badge/EgoLife-project_page-white)](https://egolife-ai.github.io/) 

**Blog:** [![demo](https://img.shields.io/badge/EgoGPT-Blog-lightblue)](https://egolife-ai.github.io/blog/) 

**Demo:** [![demo](https://img.shields.io/badge/EgoGPT-Demo-teal)](https://egolife.lmms-lab.com/) 

**Weights in Huggingface:** [![hf_checkpoint](https://img.shields.io/badge/🤗-EgoGPT_7b-yellow)](https://huggingface.co/collections/lmms-lab/egolife-67c04574c2a9b64ab312c342)

**arXiv Paper:** [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg?logo=arXiv)](https://arxiv.org/)

**Training Data:** [![data](https://img.shields.io/badge/EgoGPT-Data-purple)](https://huggingface.co/collections/lmms-lab/egolife-67c04574c2a9b64ab312c342) 


## 📢 News

- 🚀[2025/2/28] EgoGPT codebase is released!

## Introduction

EgoGPT is an omni-modal model trained on
egocentric datasets, achieving state-of-the-art performance
on egocentric video understanding. 


### Architecture
<div align="center"><img src="assets/method.png" width="100%"/></div>
The system comprises (a) a Captioning Stage powered by EgoGPT for dense visual-audio
understanding of egocentric clips, and (b) a Question Answering Stage utilizing EgoRAG for memory retrieval and response generation. The
example demonstrates temporal reasoning across multiple days, with keyword extraction, evidence retrieval, and context-aware answer
generation for a breakfast-related query

### Performance

<div align="center"><img src="assets/main_results.png" alt="results_personalized_interaction.png" width=100%></div>
EgoGPT achieves state-of-the-art performance among existing egocentric benchmarks.

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

### Download & Setup

1. Download EgoGPT-7b from 🤗[EgoGPT](https://huggingface.co/collections/lmms-lab/egolife-67c04574c2a9b64ab312c342) and audio encoder from [Audio Encoder](https://huggingface.co/EgoGPT/speech_encoder).

2. Download EgoIT dataset from 🤗[Huggingface](https://huggingface.co/datasets/EgoGPT/EgoIT_Video) and construct the directory as follows:
```python
from huggingface_hub import snapshot_download
local_path = snapshot_download(
    repo_id="EgoGPT/EgoIT_Video", 
    repo_type="dataset", 
    local_dir="data"
)
```
```bash
data/ # The directory for videos and audio (keep the same as the huggingface dataset)
├── ADL/
│   ├── images/
│   ├── audio/
├── ChardesEgo/
│   ├── *.mp4/
│   ...

datasets/ # The directory for json
├── ADL/
│   ├── ADL.json
├── ChardesEgo/
│   ├── ChardesEgo.json
├── ...
├── EgoIT.json # The concatenated json for training
```

3. If you want to train EgoGPT from scratch(e.g from LLaVA-Onevision), please download the audio project from [here](https://github.com/egolife-ntu/EgoLife-Audio).

### Inference

```shell
python inference.py --pretrained_path checkpoints/EgoGPT-7b-EgoIT-EgoLife --video_path data/train/A1_JAKE/DAY1/DAY1_A1_JAKE_11223000.mp4 --audio_path audio/DAY1_A1_JAKE_11223000.mp3 --query "Please describe the video in detail."
```

### Training
Please replace the `DATA_PATH`, `MODEL_PATH`, `SPEECH_PROJECTOR_PATH` and `SPEECH_ENCODER_PATH` in the following command with your own paths.
```shell
bash scripts/train_egogpt.sh
```

### Evaluation
#### Setup
Our evaluation are conducted on lmms-eval. Please refers to the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) repository for the evaluation setup.

#### Run
```shell
python3 -m accelerate.commands.launch \
    --main_process_port 10043 \
    --num_processes=8 \
    -m lmms_eval \
    --model egogpt \
    --model_args pretrained=YOUR_EGOGPT_MODEL_PATH, conv_template="qwen_1_5"\
    --tasks egoplan, egothink \
    --batch_size 1 \
    --log_samples \
    --output_path YOUR_OUTPUT_PATH
```

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