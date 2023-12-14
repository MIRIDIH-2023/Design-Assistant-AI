# Design Template Recommendation With SBERT

## data prepare
- prepare xml2json.py output '.pickle' file
- pickle 파일을 zip으로 압축한 경우 folder_path로 설정
- pickle 파일 원본을 가지고 있는 경우 data_path로 설정 필요

## how to run:
```
!python main.py --is_save=True --is_evaluate=True --save_path=None --folder_path=None --data_path=None --extract_path=None --num_epoch=7 --batch_size=64
```

## params: 
- --is_save: 저장 여부 (default: True)
- --is_evaluate: 평가 여부 (default: True)
- --save_path: 저장 경로 (default: None)
- --folder_path: 폴더 경로 (default: None) -> .zip(.pickle) path
- --data_path: 데이터 경로 (default: None) -> (.pickle) path
- --extract_path: 저장 경로 (default: None) 
- --num_epoch: 에폭 수 (default: 7)
- --batch_size: 배치 크기 (default: 64)