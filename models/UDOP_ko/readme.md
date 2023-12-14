# Design Layout Generation with Vision-Text-Layout Multimodel Transformer

## Model Architecture
The Vision-Text-Layout Transformer is based on UDOP, a model proposed in [Unifying Vision, Text, and Layout for Universal Document Processing (CVPR 2023 Highlight)](https://arxiv.org/abs/2212.02623). UDOP is a foundation Document AI model which unifies text, image, and layout modalities together with varied task formats, including document understanding and generation.
We have modified and fine-tuned the UDOP model for design layout generation. The model architecture is as follows:

<img src="https://github.com/miridi-sanhak/UDOP/assets/96368116/337c8acc-ab63-48d5-9aa6-ae6901dd93cf">

<img src="https://github.com/miridi-sanhak/UDOP/assets/96368116/5ec3149a-83bc-467f-9a62-b997ece85696">

The difference between UDOP_en is about tokenizer. In UDOP_en, the model uses UDOP's tokenizer as is. But in UDOP_ko, the model uses finetuned Ke-T5 Tokenizer based on MIRIDIH's dataset.

About Ke-T5 : <https://github.com/AIRC-KETI/ke-t5>

## Scripts:

### Training Model
```
python main.py config/train.yaml
```

### Inference
```
python main.py config/inference.yaml
```
