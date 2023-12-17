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

## Preprocessing data
### 데이터 경로 설정
**input**
``` python
make_sample_json('/data/path/sample.csv')       # line 298
```
**output**
``` python
filename = f"/data/path/processed_{idx}.pickle" # line 284
```
**유의 사항**
output 경로는 아래 directroy 구조를 따라야 합니다.
``` bash
data/
  └── json_data/
      └── processed_{idx}.pickle
```
``` bash
python utils/xml_to_json.py
```
### 데이터셋 분리
RendorPos만 따로 분리하여 train, eval, test 데이터셋을 생성합니다.

**경로 설정**
``` python
source_path = "path/all_data"         # .pickle 파일 경로
renderPos_data_path = "path/posData"  # RendorPos 데이터셋 저장 경로
splitted_data_path = "path/data"      # train, eval, test 데이터셋 저장 경로
```
**실행**
``` bash
python utils/split_folders.py
```
**분리 결과**
``` bash
splitted_data_path/
        ├── train/
            ├── json_data/
        ├── eval/
            ├── json_data/
        └── test/
            ├── json_data/
```
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

## Testing model options
위 Test 명령어를 실행하면 아래와 같은 옵션을 선택할 수 있습니다.
| option | 설명 |
| --- | --- |
| `${iteration}` | 반복할 횟수 (int) |
| `'none'` | Unconditional |
| `'a'` | Conditional on Category |
| `'a+s'` | Conditional on Category and Size |

| Custom option | 설명 |
| --- | --- |
| `'custom'` | custom한 방식으로 test함 아래 option들은 `'custom'` 을 선택했을 때만 입력할 수 있음 |
| `'txt'` | 이미지와 텍스트 중 텍스트만 Masking하지 않음 |
| `'im'` | 이미지와 텍스트 중 이미지만 Masking하지 않음 |
| `''` | `'custom'` option 선택 시 이미지 또는 텍스트로만 이루어진 템플릿은 입력 sequence의 첫 번째 요소만 Masking함 |
