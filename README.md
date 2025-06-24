# TradeVision: Optimizing Amazon Berkeley VQA


TradeVision is a computer Vision-Language Model (VLM) that leverages Low-Rank Adaptation (LoRA) fine-tuning to optimize Visual Question Answering performance on the Amazon Berkeley Objects dataset. The project focuses on creating a curated multiple-choice VQA dataset from product catalog images, establishing baseline performance using pre-trained multimodal models like BLIP, and then applying parameter-efficient LoRA fine-tuning to achieve superior accuracy and F1 scores.

By combining automated data curation through multimodal APIs, strategic model optimization, and comprehensive evaluation metrics, TradeVision optimizes visual understanding of e-commerce product catalogs while maintaining computational efficiency through constrained resource training on cloud GPUs.

## Key Results
* **70.3% accuracy** on fine-tuned BLIP model (vs 31.05% baseline)
* **99.7% parameter reduction** - only 1.2M trainable params out of 385M total
* Efficient training on free cloud GPUs (Google Colab/Kaggle)

## Setup & Installation

```bash
git clone https://github.com/standing-on-giants/TradeVision.git
cd TradeVision
pip install -r requirements.txt
```

```bash
python inference.py --image_dir /path/to/your/images
```

Example  :

```bash
python inference.py --image_dir ./images/small
```

Quick Start
1. **Download ABO Dataset**: Get the small variant (3GB) from Amazon Berkeley Objects
2. **Generate VQA Dataset**: By running the "Data Curation.ipynb" notebook


### Key Features
* **Automated Dataset Curation**: Uses Gemini API to generate 54K image-question-answer triplets
* **Parameter-Efficient Training**: LoRA reduces trainable parameters by 99.7%
* **Multi-Model Support**: BLIP and ViLT baseline comparisons
* **Comprehensive Evaluation**: Accuracy, F1-Score, and BERTScore metrics
* **Resource Optimized**: Designed for free cloud GPU platforms

## Configuration used for training

| Parameter | Value |
|-----------|-------|
| Base Model | Salesforce/blip-vqa-base |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Learning Rate | 5e-5 |
| Batch Size | 4 |
| Epochs | 3 |

## Summary of results

| Model | Accuracy | F1 Score | Trainable Params |
|-------|----------|----------|------------------|
| BLIP Baseline | 31.05% | 0.45 | 385M |
| BLIP + LoRA | **70.30%** | **0.48** | **1.2M** |

## Contributors
* Shashank Devarmani - LoRA implementation, model analysis and selection, debugging
* Soham Pawar - Baseline evaluation, and debugging
* Aaryan Dev - Dataset cleaning and dataset curation


