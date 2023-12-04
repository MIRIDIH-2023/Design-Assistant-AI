import pandas as pd
import requests
import xmltodict
import xml_to_dict
from PIL import Image, ImageOps, ImageEnhance
import io
from wand.image import Image as WandImage
from wand.color import Color
from tqdm import tqdm
import re
import PIL
import os
import pickle
import math
import json
from io import BytesIO

# SVG의 색상을 color_map에 따라 변경
def change_svg_color(svg_content, color_map):  
  svg_content = svg_content.decode('utf-8')
  
  color_mapping = {}
  # fillColor가 리스트가 아닌 경우 리스트로 변환
  if not isinstance(color_map['fillColor'], list):
    color_map['fillColor'] = [color_map['fillColor']]
    # 색상 변경을 위한 color_map 생성
  for fill_color in color_map['fillColor']:
    color_mapping[fill_color['@originColor']] = fill_color['@color']
  for origin_color, color in color_mapping.items():
    # origin_color와 color가 같은 경우 무시
    if origin_color == color:
      continue
    origin_color = origin_color.upper()
    color = color.upper()
    svg_content = svg_content.replace(origin_color, color) # 색상 변경

  return svg_content.encode('utf-8')

# 필요한 데이터를 로드하고 합친 후 반환
def load_data():
  # 템플릿 데이터
  df_1 = pd.read_csv('data/xml샘플데이터_20230703-1.csv')
  df_2 = pd.read_csv('data/xml샘플데이터_20230703-2.csv')
  df = pd.concat([df_1, df_2])
  df = df.reset_index(drop=True)
  # 래스터 이미지 링크 데이터
  image_links_1 = pd.read_csv('data/이미지링크데이터_20230926-1.csv')
  image_links_2 = pd.read_csv('data/이미지링크데이터_20230926-2.csv')
  image_links = pd.concat([image_links_1, image_links_2])
  image_links = image_links.reset_index(drop=True)
  # 벡터 이미지 링크 데이터
  svg_links_1 = pd.read_csv('data/이미지링크데이터_20231012-1.csv')
  svg_links_2 = pd.read_csv('data/이미지링크데이터_20231012-2.csv')
  svg_links = pd.concat([svg_links_1, svg_links_2])
  svg_links = svg_links.reset_index(drop=True)
  return df, image_links, svg_links

# 주어진 key로부터 이미지의 url을 반환
def get_image_link(key, image_links, svg_links, is_svg): # is_svg: svg인지 아닌지 여부
  if is_svg:
    return svg_links[svg_links['key'] == key]['image_url'].values[0]
  return image_links[image_links['key'] == key]['image_url'].values[0]


# URL로부터 XML을 다운로드하고 파싱하여 반환
def parse_xml_from_url(url):
  response = requests.get(url)
  content = response.content
  xml = xmltodict.parse(content)
  return xml

# XML로부터 전체 sheet의 크기를 추출하여 반환
def extract_size_from_xml(xml):
  size = (int(xml['SHEET']['SHEETSIZE']['@cx']), int(xml['SHEET']['SHEETSIZE']['@cy']))
  return size

# XML로부터 이미지를 렌더링하는데 필요한 데이터를 추출하여 반환
def extract_image_data_from_xml(xml, image_links, svg_links):
  images = []

  # 이미지 관련 XML태그와 해당 태그의 key값이 적혀 있는 속성의 이름을 매핑
  xml_tags_and_keys = {
    'STICKER': '@StickerId',
    'PHOTO': '@PhotoKey',
    'SVG': '@SvgKey',
    'SHAPESVG': '@SvgKey'
  }

  for tag in xml['SHEET']:
    if tag in xml_tags_and_keys:
      items = xml['SHEET'][tag]
      if not isinstance(items, list):
        items = [items]
      for item in items:
        images.append({
          'left': item['Position']['@Left'],
          'top': item['Position']['@Top'],
          'right': item['Position']['@Right'],
          'bottom': item['Position']['@Bottom'],
          'image': get_image_link(item[xml_tags_and_keys[tag]], image_links, svg_links, is_svg=tag == 'SVG' or tag == 'SHAPESVG'),
          'priority': item['@Priority'],
          'flipH': item['@FlipH'],
          'flipV': item['@FlipV'],
          'fillColorMap': item.get('fillColorMap', None),
          'isSvg': tag == 'SVG' or tag == 'SHAPESVG',
          'opacity': item.get('@Opacity', '255'),
          'rotate': item.get('@Rotate', '0')
        })

  # priority가 낮은 이미지가 먼저 그려지도록 정렬
  images = sorted(images, key=lambda x: x['priority'])
  return images

# XMl로부터 배경색을 추출하여 반환
def extract_background_color_from_xml(xml):
  background_color = xml['SHEET']['BACKGROUND']['@Color']
  return background_color

# XML로부터 배경 이미지를 렌더링하여 반환
def create_image_from_xml(xml, image_links, svg_links, size_):
  size = extract_size_from_xml(xml)
  images_data = extract_image_data_from_xml(xml, image_links, svg_links)
  background_color = extract_background_color_from_xml(xml)

  canvas = Image.new('RGBA', size, background_color)

  # 각 이미지를 렌더링하여 canvas에 추가
  for image_data in images_data:
    left, top, right, bottom = float(image_data['left']), float(image_data['top']), float(image_data['right']), float(image_data['bottom'])
    left, top, right, bottom = round(left), round(top), round(right), round(bottom)
    img_url = image_data['image']

    response = requests.get(img_url)
    content = response.content
    if image_data['fillColorMap'] is not None:
      content = change_svg_color(content, image_data['fillColorMap'])
    if image_data['isSvg']:
      with WandImage(blob=content, format='svg', background=Color('transparent'), resolution=72) as svg_image:
        img_data = svg_image.make_blob(format='png')
      img = Image.open(io.BytesIO(img_data))
    else:
      img = Image.open(io.BytesIO(response.content))

    if img.mode != 'RGBA':
      img = img.convert('RGBA')

    opacity = round(float(image_data['opacity'])) / 255.0
    enhancer = ImageEnhance.Brightness(img.split()[3])
    alpha = enhancer.enhance(opacity)
    img.putalpha(alpha)

    angle = float(image_data['rotate'])
    if angle != 0:
      img = img.rotate(angle, resample=Image.BICUBIC, expand=True)

    if image_data['flipH'] != "0":
      img = ImageOps.mirror(img)
    if image_data['flipV'] != "0":
      img = ImageOps.flip(img)
    
    img = img.resize((right-left, bottom-top))
    
    canvas.paste(img, (left, top), img)

  canvas = canvas.resize(size_, resample=Image.BICUBIC)
  return canvas

# 샘플 데이터의 링크에 몇몇 잘못된 형식의 링크가 있음
# 이를 처리하기 위한 함수
def clean_url(url):
    return re.sub(r'(https://file\.miricanvas\.com/).*(template_sheet)', r'\1\2', url)

# 바운딩 박스를 이미지 크기에 맞게 리사이징하고, 주어진 각도만큼 회전
def process_bbox(XML_BBOX, IM_SIZE, SHEET_SIZE, angle, center):
    RATIO = IM_SIZE[0] / SHEET_SIZE[0]
    x1, y1, x2, y2 = map(float, XML_BBOX)
    x1, y1, x2, y2 = (x1 * RATIO, y1 * RATIO, x2 * RATIO, y2 * RATIO)
    center = (center[0] * RATIO, center[1] * RATIO)

    if angle != 0:
        angle = 360 - angle
        angle = math.radians(angle)
        
        center_x, center_y = center

        distance_x = (x1 - center_x)
        distance_y = (y1 - center_y)

        new_distance_x = distance_x * math.cos(angle) - distance_y * math.sin(angle)
        new_distance_y = distance_x * math.sin(angle) + distance_y * math.cos(angle)

        x1 = center_x + new_distance_x
        y1 = center_y + new_distance_y
        x2 = center_x - new_distance_x
        y2 = center_y - new_distance_y

    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    return x1, y1, x2, y2

# 주어진 텍스트 데이터로부터 바운딩 박스를 추출하여 반환
def get_render_bbox(text):
    global total_no_renderpos
    # RenderPos가 없는 경우 무시
    if "RenderPos" not in text or text['RenderPos'] == None:
        total_no_renderpos += 1
        print("No RenderPos", end=" ")
        return None
    render_pos = json.loads(text['RenderPos'])
    render_bbox = []

    left, top, right, bottom = map(float, text['Position'].values())

    for render in render_pos['c']:
        try:
            x, a, w, y = map(float, [render['x'], render['a'], render['w'], render['y']])
        except:
            print("Wrong render pos", end=" ")
            return None
        left_ = left + x
        right_ = left_ + w
        bottom_ = top + y
        top_ = bottom_ - a

        render_bbox.append((left_, top_, right_, bottom_))

    return render_bbox

# 텍스트 데이터로부터 추출한 바운딩 박스 리스트로부터 전체 바운딩 박스를 계산하여 반환
def get_bbox(render_bbox):
    min_x = min([x[0] for x in render_bbox])
    min_y = min([x[1] for x in render_bbox])
    max_x = max([x[2] for x in render_bbox])
    max_y = max([x[3] for x in render_bbox])

    return min_x, min_y, max_x, max_y

# 오류 처리 함수 
def dictIntoList(TEXT_TagData) :
  if type(TEXT_TagData) is dict :
    temp = []
    temp.append(TEXT_TagData)
    TEXT_TagData = temp
  
  return TEXT_TagData

# XML데이터를 UDOP 학습을 위한 json 형태로 변환
def process_xml_dict(xml_dict, thumbnail):
    global total_no_text
    processed_json = {}
    processed_json['form'] = []

    SHEET_SIZE = tuple(map(int, xml_dict['SHEET']['SHEETSIZE'].values()))
    IM_SIZE = thumbnail.size

    # 오류: TEXT, RendorPos, TEXT안 text 데이터가 list가 아닌 dict인 경우
    if 'TEXT' not in xml_dict['SHEET']:
        total_no_text += 1
        print("No TEXT", end=" ")
        return None

    TEXT_TagData = xml_dict['SHEET']['TEXT']
    TEXT_TagData = dictIntoList(TEXT_TagData) # TEXT안 text 데이터 형태가 list가 아닌 dict인 오류 처리 

    # Process XML to json
    for idx, text in enumerate(TEXT_TagData):
        left, top, right, bottom = map(float, text['Position'].values())
        center = ((left + right) / 2, (top + bottom) / 2)

        render_bbox = get_render_bbox(text)
        if render_bbox is None or len(render_bbox) == 0: return None

        XML_BBOX = get_bbox(render_bbox)

        t = text['Text']
        x1, y1, x2, y2 = process_bbox(XML_BBOX, IM_SIZE, SHEET_SIZE, int(float(text['@Rotate'])), center)

        processed_json['form'].append({
            "text": t,
            "box": [x1, y1, x2, y2],
            "font_id": text['Font']['@FamilyIdx'],
            "font_size": text['Font']['@Size'],
            "style": {
                "bold": text['Font']['Style']['@Bold'] == 'true',
                "italic": text['Font']['Style']['@Italic'] == 'true',
                "strikeout": text['Font']['Style']['@Strikeout'] == 'true',
                "underline": text['Font']['Style']['@Underline'] == 'true'
            },
            "linespace": text['Font']['@LineSpace'],
            "opacity": text['@Opacity'],
            "rotate": text['@Rotate'],
            "id": idx,
            "sheet_size" : SHEET_SIZE
        })

        processed_json['form'][-1]['words'] = []

        render_pos = json.loads(text['RenderPos'])

        for j, bbox in enumerate(render_bbox):
            x1_, y1_, x2_, y2_ = process_bbox(bbox, IM_SIZE, SHEET_SIZE, int(float(text['@Rotate'])), center)
            color = render_pos['c'][j]['f']
            if color.startswith('rgba'):
                color = color[5:-1]
            else: color = color[4:-1]
            color = list(map(int, color.split(",")))
            processed_json['form'][-1]['words'].append({
                "text": render_pos['c'][j]['t'],
                "box": [x1_, y1_, x2_, y2_],
                "font_size": float(render_pos['c'][j]['s']),
                "letter_spacing": float(render_pos['c'][j]['ds']),
                #"font_id": int(render_pos['c'][j]['yd']),
                "color": color
            })

    return processed_json

# XML의 링크와 썸네일의 링크로부터 json 형태로 변환한 데이터와 원본 썸네일 이미지의 크기를 반환
def process_xml(sheet_url, thumbnail_url):
    global total_wrong_image
    global total_wrong_xml
    try:
        sample_thumbnail = Image.open(BytesIO(requests.get(thumbnail_url).content))
    except:
        print(f"Error at loading image, {thumbnail_url}", end = " ")
        total_wrong_image += 1
        return None
    try:
        sample_xml = requests.get(sheet_url).content.decode("utf-8")
    except:
        print(f"Error at loading xml, {sheet_url}", end = " ")
        total_wrong_xml += 1
        return None
    sample_json = xml_to_dict.XMLtoDict().parse(sample_xml)

    processed_json = process_xml_dict(sample_json, sample_thumbnail)

    return processed_json, sample_thumbnail.size

# 16진수 색상을 RGB로 변환
def hex_to_rgb(hex_string):
    return [int(hex_string[idx:idx+2], 16) for idx in (1, 3, 5)]

# 샘플 csv의 각 행을 json 형태로 변환하여 파일로 저장
def make_sample_json(row, idx):
    # 영어 데이터만 사용
    if row['language'] == 'en':
        processed_json, size = process_xml(row['sheet_url'], row['thumbnail_url'])
        if processed_json is None : 
            return None
        processed_json['template_idx'] = int(row['tempate_idx'])
        processed_json['sheet_url'] = row['sheet_url']
        processed_json['thumbnail_url'] = row['thumbnail_url']
        processed_json['page_no'] = int(row['page_no'])
        processed_json['title'] = row['title']
        processed_json['primary_colors'] = [hex_to_rgb(color) for color in row['primary_colors'][1:-1].split(",")]
        processed_json['primary_color_weights'] = [float(weight) for weight in row['primary_color_weights'][1:-1].split(",")]
        processed_json['language'] = row['language']
        processed_json['category'] = row['category']
        processed_json['keyword'] = list(set(row['keyword'].split("|")))
        filename = f"./data/json_data/{idx}.pickle"
        with open(filename, "wb") as file_:
            pickle.dump(processed_json, file_)
    else:
       # 영어가 아닌 경우 무시
       return None
    return processed_json, size

def main():
  df, image_links, svg_links = load_data()

  start_idx = int(input('start index: '))
  end_idx = int(input('end index (-1 for end): '))
  if end_idx == -1:
    end_idx = len(df) + 1

  # 샘플 csv의 각 행을 json 형태로 변환하여 파일로 저장
  # 또한 글자를 제거한 배경 이미지를 생성하여 파일로 저장
  for idx in tqdm(range(start_idx, end_idx - 1)):
    try: 
      processed_json, size = make_sample_json(df.iloc[idx], idx)
      if processed_json is None:
        continue
      url = df['sheet_url'][idx]
      print("thumbnail_url: ", df['thumbnail_url'][idx])
      url = clean_url(url)
      xml = parse_xml_from_url(url)
      image = create_image_from_xml(xml, image_links, svg_links, size)
      image.save('data/images/image_{}.png'.format(idx))
    except Exception as e:
      with open('data/error.txt', 'a') as f:
        f.write('[{}]\n'.format(idx))
        f.write('{}\n'.format(e))
      # 오류가 발생한 경우 생성된 파일을 모두 삭제
      if os.path.exists('data/images/image_{}.png'.format(idx)):
        os.remove('data/images/{}.png'.format(idx))
      if os.path.exists('data/json_data/{}.pickle'.format(idx)):
        os.remove('data/json_data/{}.pickle'.format(idx))
      continue

if __name__ == "__main__":
    main()
