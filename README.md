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

Quick Start
1. **Download ABO Dataset**: Get the small variant (3GB) from Amazon Berkeley Objects
2. **Generate VQA Dataset**:

```
bash
```


```bash
python data_curation/gemini_qa_generation.py --data_path /path/to/abo
```

1. **Run Baseline Evaluation**:

```
bash
```


```bash
python models/baseline_evaluation.py --dataset_path ./datasets/vqa_dataset.csv
```

1. **Fine-tune with LoRA**:

```
bash
```


```bash
python models/blip_lora_training.py --epochs 3 --batch_size 4 --rank 16
```

### Key Features
* **Automated Dataset Curation**: Uses Gemini API to generate 54K image-question-answer triplets
* **Parameter-Efficient Training**: LoRA reduces trainable parameters by 99.7%
* **Multi-Model Support**: BLIP and ViLT baseline comparisons
* **Comprehensive Evaluation**: Accuracy, F1-Score, and BERTScore metrics
* **Resource Optimized**: Designed for free cloud GPU platforms
Training Configuration

```
ParameterValueBase ModelSalesforce/blip-vqa-baseLoRA Rank16LoRA Alpha32Learning Rate5e-5Batch Size4Epochs3
```

## Results Summary

```
ModelAccuracyF1 ScoreTrainable ParamsBLIP Baseline31.05%0.45385MBLIP + LoRA70.30%0.481.2M
```
## Requirements
* Python 3.8+
* PyTorch 1.11+
* Transformers 4.20+
* PEFT library
* Google Colab or Kaggle (recommended)

## Contributors
* [Your Name] - Dataset curation, LoRA implementation
* [Teammate 1] - Baseline evaluation, metrics
* [Teammate 2] - Model optimization, analysis  


