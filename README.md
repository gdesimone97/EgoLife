# The EgoLife Project
<p align="center">
  <a href="" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-CVPR2025-b31b1b?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://egolife-ai.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/Page-egolife--ai.github.io-228c22?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/collections/lmms-lab/egolife-67c04574c2a9b64ab312c342" target='_blank'>
    <img src="https://img.shields.io/badge/Data-HuggingFace-FFD21E?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://github.com/egolife-ai/EgoLife" target='_blank'>
    <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fegolife-ai%2FEgoLife&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=true">
  </a>
</p>

| ![teaser.png](assets/egolife_teaser.png) |
|:---|
| <p align="justify"><b>Figure 1. The Overview of EgoLife Project.</b> EgoLife is an ambitious egocentric AI project capturing multimodal daily activities of six participants over a week. Using Meta Aria glasses, synchronized third-person cameras, and mmWave sensors, it provides a rich dataset for long-term video understanding. Leveraging this dataset, the project enables AI assistants—powered by EgoGPT and EgoRAG—to support memory, habit tracking, event recall, and task management, advancing real-world egocentric AI applications.
</p>


## 🚀 News
🌟 2025-02-28: We release the EgoIT dataset at [HuggingFace](https://huggingface.co/collections/lmms-lab/egolife-67c04574c2a9b64ab312c342). The EgoLife video is also released, and is uploading to [Youtube](https://www.youtube.com/playlist?list=PLlweuFnfdo6F9Fu2Kyhc-kXu3qnaVsYOu). The EgoLife Captioning and EgoLifeQA are in final preparation.

🌟 2025-02-28: We release the first version of [EgoGPT](./EgoGPT/) and [EgoRAG](./EgoRAG/).

🎉 2025-02-27: The paper is accepted by CVPR 2025.


## What is in this repo?
---
### 🧠 EgoGPT: Clip-Level Multimodal Understanding
EgoGPT is an **omni-modal vision-language model** fine-tuned on egocentric datasets. It performs **continuous video captioning**, extracting key events, actions, and context from first-person video and audio streams. 

**Key Features:**
- **Dense captioning** for visual and auditory events.
- **Fine-tuned for egocentric scenarios** (optimized for EgoLife data).

### 📖 EgoRAG: Long-Context Question Answering
EgoRAG is a **retrieval-augmented generation (RAG) module** that enables long-term reasoning and memory reconstruction. It retrieves **relevant past events** and synthesizes contextualized answers to user queries.

**Key Features:**
- **Hierarchical memory bank** (hourly, daily summaries).
- **Time-stamped retrieval** for context-aware Q&A.

---

## 📂 Code Structure
```bash
EgoLife/
│── assets/                # General assets used across the project
│── EgoGPT/                # Core module for egocentric omni-modal model
│── EgoRAG/                # Retrieval-augmented generation (RAG) module
│── README.md              # Main documentation for the overall project
```
Please dive in to the README of [EgoGPT](./EgoGPT/README.md) and [EgoRAG](./EgoRAG/README.md) for more details.

## 📢 Citation

If you use EgoLife in your research, please cite our work:

```bibtex
@inproceedings{yang2025egolife,
  title={EgoLife: Towards Egocentric Life Assistant},
  author={Yang, Jingkang and Liu, Shuai and Guo, Hongming and Dong, Yuhao and Zhang, Xiamengwei and Zhang, Sicheng and Wang, Pengyun and Zhou, Zitang and Xie, Binzhu and Wang, Ziyue and Ouyang, Bei and Lin, Zhengyu and Cominelli, Marco and Cai, Zhongang and Zhang, Yuanhan and Zhang, Peiyuan and Hong, Fangzhou and Widmer, Joerg and Gringoli, Francesco and Yang, Lei and Li, Bo and Liu, Ziwei},
  booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025},
}
```