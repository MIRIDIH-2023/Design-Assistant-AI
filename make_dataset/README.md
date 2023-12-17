# 학습 데이터 셋 생성 코드

## 소개
- xml_to_json.py : UDOP, SBERT 학습 데이터 생성용
- blt_xml_to_json.py : BLT 학습 데이터 생성용

## 사용 방법
`xml_to_json.py` 사용 방법만을 다룹니다. BLT 모델의 학습 데이터 생성에 대한 구체적인 설명은 다음 링크를 참고해주세요.
[데이터 생성](https://github.com/MIRIDIH-2023/Design-Assistant-AI/blob/main/models/BLT/README.md)

### 데이터 경로 설정
**input**
``` python
# load_data()
# 템플릿 데이터
df_1 = pd.read_csv('data/sample1.csv')
df_2 = pd.read_csv('data/sample2.csv')

# 래스터 이미지 링크 데이터
image_links_1 = pd.read_csv('data/link1.csv')
image_links_2 = pd.read_csv('data/link2.csv')

# 벡터 이미지 링크 데이터
svg_links_1 = pd.read_csv('data/link3.csv')
svg_links_2 = pd.read_csv('data/link4.csv')
```
**output**
``` python
filename = f"./data/json_data/{idx}.pickle"        # line 449 
image.save('data/images/image_{}.png'.format(idx)) # line 477 
```
**유의 사항**
output 경로는 아래 directroy 구조를 따라야 합니다.
``` bash
data/
  ├── images/
  │   └── image_{idx}.png
  └── json_data/
      └── processed_{idx}.pickle
```

``` bash
python xml_to_json.py
```

## 데이터 구성
| key | 데이터 형식 |
| --- | --- |
| `'form'` | ```List[Dict[str, Union]]``` |
| `'template_idx'` | `int` |
| `'sheet_url'` | `str` |
| `'thumbnail_url'` | `str` |
| `'page_no'` | `int` |
| `'title'` | `str` |
| `'primary_colors'` | `List[List[int]]` |
| `'primary_color_weights'` | `List[int]` |
| `'language'` | `str` |
| `'category'` | `str` |
| `'keyword'` | `List[str]` |

|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; form's key &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| 데이터 형식 |
| --- | --- |
| `'text'` | `str` |
| `'words'` | `List[Dict[str, Union]]` |
| `'box'` | `List[float]` |
| `'font_id'` | `int` |
| `'font_size'` | `float` |
| `'style'` | `Dict[str, bool]` |
| `'linespace'` | `float` |
| `'opacity'` | `str` |
| `'rotate'` | `str` |
| `'id'` | `int` |
| `'sheet_size'` | `Tuple[int]` |

### BLT 학습 데이터 구성
**위 학습 데이터에서 추가되거나 수정된 부분**

BLT는 이미지 데이터를 사용하지 않기 때문에 XML 데이터만 저장합니다. 
| key | 데이터 형식 |
| --- | --- |
| `'tags_info'` | ```List[Dict[str, Union]]``` |
| `'no_texts'` | `int` |
| `'no_RendorPos'` | `str` |
| `'thumbnail_size'` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| `str` |
| `'idx'` | `int` |

|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; form's key &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| 데이터 형식 |
| --- | --- |
| (추가) `'raw_bbox'` | `List[float]` |
| (수정) `'font_family'` | `List[str]` |
| (수정) `'font_family_id'` | `List[int]` |
| (수정) `'font_size'`| `List[float]` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
| (추가) `'priority'` | `str` |
| (추가) `'linespace'` | `List[float]` |
| (추가) `'outline_size'` | `List[float]` |

|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tags_info's key &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| 데이터 형식 |
| --- | --- |
| `'tag'` | `str` |
| `'box'` | `List[float]` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
| `'rotate'` | `str` |
| `'priority'` | `str` |
