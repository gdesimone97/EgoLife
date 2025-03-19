# The EgoLife Project
<p align="center">
  <a href="https://arxiv.org/abs/2503.03803" target='_blank'>
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
🤹 2025-02: We provide [HuggingFace gradio demo]() and [self-deployed demo]() for EgoGPT.

🌟 2025-02: The EgoLife video is released at [HuggingFace](https://huggingface.co/datasets/lmms-lab/EgoLife) and uploaded to [Youtube](https://www.youtube.com/playlist?list=PLlweuFnfdo6F9Fu2Kyhc-kXu3qnaVsYOu) as video collection.

🌟 2025-02: We release the EgoIT-99K dataset at [HuggingFace](https://huggingface.co/collections/lmms-lab/egolife-67c04574c2a9b64ab312c342). 

🌟 2025-02: We release the first version of [EgoGPT](./EgoGPT/) and [EgoRAG](./EgoRAG/) codebase.

📖 2025-02: Our arXiv submission is currently on hold. For an overview, please visit our [academic page](https://egolife-ai.github.io/blog/).

🎉 2025-02: The paper is accepted to CVPR 2025. Please be invited to our [online EgoHouse](https://egolife-ai.github.io/).


## What is in this repo?
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


## 📂 Code Structure
```bash
EgoLife/
│── assets/                # General assets used across the project
│── EgoGPT/                # Core module for egocentric omni-modal model
│── EgoRAG/                # Retrieval-augmented generation (RAG) module
│── README.md              # Main documentation for the overall project
```
Please dive in to the project of [EgoGPT](./EgoGPT/) and [EgoRAG](./EgoRAG/) for more details.

## 📢 Citation

If you use EgoLife in your research, please cite our work:

```bibtex
@misc{yang2025egolifeegocentriclifeassistant,
      title={EgoLife: Towards Egocentric Life Assistant}, 
      author={Jingkang Yang and Shuai Liu and Hongming Guo and Yuhao Dong and Xiamengwei Zhang and Sicheng Zhang and Pengyun Wang and Zitang Zhou and Binzhu Xie and Ziyue Wang and Bei Ouyang and Zhengyu Lin and Marco Cominelli and Zhongang Cai and Yuanhan Zhang and Peiyuan Zhang and Fangzhou Hong and Joerg Widmer and Francesco Gringoli and Lei Yang and Bo Li and Ziwei Liu},
      year={2025},
      eprint={2503.03803},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.03803}, 
}
```

## 📝 License
This project is licensed under the S-Lab license. See the [LICENSE](LICENSE) file for details.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=EvolvingLMMs-Lab/EgoLife&type=Date)](https://star-history.com/#EvolvingLMMs-Lab/EgoLife&Date)
