# Design Template Recommendation With SBERT

## data prepare
- prepare xml2json.py output ('.pickle' file or '.zip' file)


## how to run:
```
!python models/SBERT/main.py  --is_save=True --is_evaluate=True --save_path=None --folder_path=None --data_path=None --extract_path=None --num_epoch=7 --batch_size=64
```

## params: 
- --is_save: 저장 여부 (deafult: True)
- --is_evaluate: 평가 여부 (deafult: True)
- --save_path: 저장 경로 (deafult: None)
- --folder_path: 폴더 경로 (deafult: 'make_dataset/sample/sbert_data') -> .zip(.pickle) 경로
- --data_path: 데이터 경로 (deafult: 'models/SBERT/data') -> (.pickle) 경로
- --extract_path: 저장 경로 (deafult: 'models/SBERT/data') 
- --num_epoch: 에폭 수 (deafult: 3) -> best = 7
- --batch_size: 배치 크기 (deafult: 1) -> best = 100


## folder structure
```python
.
├── data
│   └── readme.md        # save .zip file in here 
├── main.py              # main train/inference code
├── readme.md
├── requirements.txt
└── src
    ├── test_dataset.py  # torch.dataset of test dataset. include positive pair only
    ├── train_dataset.py # torch.dataset of train dataset. include positive/negative pair
    └── utils.py         # utils including process data, eval metric, etc...
```