# Design-Assistant-AI
## 2023 MIRIDIH Corporate Collaboration Project
--- 
<div align="center">

<img src="img/stop2023.png" width="100%" >

</div>

## Contributors 🙌 
- 김상현 (성균관대학교 소프트웨어학과 22학번)
- 김종효 (성균관대학교 소프트웨어학과 22학번)
- 신상윤 (성균관대학교 소프트웨어학과 21학번)
- 이찬영 (성균관대학교 소프트웨어학과 19학번)
- 주현진 (성균관대학교 소프트웨어학과 22학번)

# How to use (in local host)
## 1. run api
```bash
python models/SBERT/api.py #run SBERT api
python models/UDOP_en/udop.py #run UDOP api
```
## 2. run local host
```bash
npm install
npm run dev
```

# Repository Structure
```
.
├── data_folder
│
├── main.py
├── readme.md
├── requirements.txt
└── src
    ├── custom_dataset.py
    ├── test_dataset.py
    └── utils.py

```

```
.
|── make_dataset            # convert xml file to json
└── models                  # models including (BLT, SBERT, UDOP-en/kr)
    ├── BLT                 # BLT
    │   ├── configs     
    │   ├── dataset     
    │   ├── nets            # main model archicture
    │   ├── trainers        # model training codes
    │   └── utils       
    ├── SBERT
    │   └── src             # codes including train/inference
    ├── UDOP_en             # UDOP english
    │   ├── config      
    │   ├── core            # main source code
    │   │   ├── common
    │   │   ├── datasets
    │   │   └── trainers    # main training codes
    │   ├── data            # data for train/inference
    │   │   ├── images
    │   │   └── json_data
    │   └── utils
    ├── UDOP_ko             # UDOP korean
    │   ├── config
    │   ├── core            # main source code
    │   │   ├── common
    │   │   ├── datasets    # data for train/inference
    │   │   ├── models      # main source code
    │   │   │   ├── embedding
    │   │   │   │   └── relative
    │   │   │   └── mae     
    │   │   └── trainers    # main training code
    │   └── ket5-finetuned  # tokenizer with korean
    └── VSE                 # VSE
        ├── data            # data for train/inference/pre-training
        │   ├── coco
        │   └── f30k
        ├── docs            # folder just for readme
        │   ├── _layouts
        │   └── assets
        │       ├── css
        │       └── img
        └── lib
            ├── datasets    
            └── modules     # main source code
                └── aggr
```

```
.
└── web
    ├── backend                         # backend code 
    │   ├── core                        # main backend source code
    │   │   ├── common
    │   │   └── models                  # model apis
    │   │       ├── sbert               # dealing sbert api
    │   │       └── udop                # dealing udop api
    │   │           └── transformers    # for inference 
    │   │               ├── generation 
    │   │               └── utils
    │   └── utils
    └── frontend                        # frontend code
        ├── components
        ├── pages
        │   └── api
        │       ├── chat                # dealing gpt api
        │       │   └── agents          # dealing react agent
        │       └── utils
        ├── public                  
        │   ├── fonts
        │   └── images
        ├── styles 
        ├── types
        └── utils
```
