# Bidirectional Layout Transformer (BLT)

**[BLT: Bidirectional Layout Transformer for Controllable Layout Generation](https://arxiv.org/abs/2112.05112)** In ECCV'22.

원본 코드 출처: [layout-blt](https://github.com/google-research/google-research/tree/master/layout-blt)

# Introduction

Automatic generation of such layouts is important as we seek scale-able and diverse visual designs. We introduce BLT, a bidirectional layout transformer. BLT differs from autoregressive decoding as it first generates a draft layout that satisfies the user inputs and then refines the layout iteratively.

## Set up environment

```
conda env create -f environment.yml
conda activate layout_blt
```
```
pip install jaxlib==0.1.69+cuda110 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 출처로 남긴 원본 코드에서 환경 구축 시 유의 사항

원본 코드에서 environment.yml이 update되지 않았다면 아래 링크를 참고해 **jax.Array**가 **jax.numpy.DeviceArray** 로 변경되어 있는지 확인해주세요.

- [[commit error] layout-blt/utils/layout_bert_fast_decode.py](https://github.com/google-research/google-research/commit/89bd283df95962480163778d32ca62baec06392e#diff-d50bc9b308611a6985e4b5a22be2550862a65a951ab4c76909e6318076e9d07e)
- [[commit error] layout-blt/utils/layout_fast_decode.py](https://github.com/google-research/google-research/commit/89bd283df95962480163778d32ca62baec06392e#diff-54e5487c1e5f718c0155f009478d5f506d842f21865583c1a8dd5fdd252314a8)



## Running

```
# model_dir 이름 형식: 실험 번호와 실험 이름을 '_'로 구분
# 예시) exp1_expName
# Training a model
python  main.py --config configs/${config} --workdir ${model_dir}

# Testing a model
python  main.py --config configs/${config} --workdir ${model_dir} --mode test

# Evaluating a model with IOU
python main.py --config configs/${config} --workdir ${model_dir} --mode eval
```
