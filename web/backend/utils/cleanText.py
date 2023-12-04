import os
import openai
import requests
import xml_to_dict
import math
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from io import BytesIO
from PIL import Image
import pickle, json
import numpy as np

# def get_bbox_list(data):
#   bbox_list = []
#   for text in data['form']:
#     if text["text"] == " " or text["text"] == None or text["text"] == '' : continue
    
#     bbox_list.append(text["box"])
    
#   return bbox_list

def process_bbox(XML_BBOX, IM_SIZE, SHEET_SIZE, angle, center):
    RATIO = IM_SIZE[0] / SHEET_SIZE[0]
    x1, y1, x2, y2 = map(float, XML_BBOX)
    x1, y1, x2, y2 = (x1 * RATIO, y1 * RATIO, x2 * RATIO, y2 * RATIO)
    center = (center[0] * RATIO, center[1] * RATIO)

    if angle != 0:
        angle = 360 - angle
        angle = math.radians(angle)
        # Calculate the center point of the bbox
        center_x, center_y = center
        # Calculate the distance from the center to each corner of the bbox
        distance_x = (x1 - center_x)
        distance_y = (y1 - center_y)
        # Apply rotation to the distances
        new_distance_x = distance_x * math.cos(angle) - distance_y * math.sin(angle)
        new_distance_y = distance_x * math.sin(angle) + distance_y * math.cos(angle)
        # Calculate the new corners after rotation
        x1 = center_x + new_distance_x
        y1 = center_y + new_distance_y
        x2 = center_x - new_distance_x
        y2 = center_y - new_distance_y

    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    return x1, y1, x2, y2

def get_render_bbox(text):
    if "RenderPos" not in text or text['RenderPos'] == None:
        return None
    render_pos = json.loads(text['RenderPos'])
    render_bbox = []

    left, top, right, bottom = map(float, text['Position'].values())

    for render in render_pos['c']:
        try:
            x, a, w, y = map(float, [render['x'], render['a'], render['w'], render['y']])
        except:
            return None
        left_ = left + x
        right_ = left_ + w
        bottom_ = top + y
        top_ = bottom_ - a

        render_bbox.append((left_, top_, right_, bottom_))

    return render_bbox

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


def get_bbox_list(xml_dict, thumbnail):
  bbox_list = []
  
  SHEET_SIZE = tuple(map(int, xml_dict['SHEET']['SHEETSIZE'].values()))
  IM_SIZE = thumbnail.size
  
  if 'TEXT' not in xml_dict['SHEET']:
    total_no_text += 1
    print("No TEXT", end=" ")
    return None

  TEXT_TagData = xml_dict['SHEET']['TEXT']
  TEXT_TagData = dictIntoList(TEXT_TagData)       # TEXT안 text 데이터 형태가 list가 아닌 dict인 오류 처리 

  # Process XML to json
  for idx, text in enumerate(TEXT_TagData):
    t = text['Text']
    if t == " " or t == None or t == '' : continue 
    
    left, top, right, bottom = map(float, text['Position'].values())
    center = ((left + right) / 2, (top + bottom) / 2)

    render_bbox = get_render_bbox(text)
    if render_bbox is None or len(render_bbox) == 0: return None

    XML_BBOX = get_bbox(render_bbox)

    x1, y1, x2, y2 = process_bbox(XML_BBOX, IM_SIZE, SHEET_SIZE, int(float(text['@Rotate'])), center)
    
    bbox_list.append([x1, y1, x2, y2])
    
  return bbox_list

def masking(img, bbox_list, offset=10):
  array_img = np.array(img)
  
  for ltrb in bbox_list:
    left, top, right, bottom = ltrb
    
    left -= offset
    top -= offset
    right += offset
    bottom += offset
    
    array_img[top:bottom, left:right, 3] = 0
    
  mask_img = Image.fromarray(array_img)
  
  return mask_img

# =====================================
# 이미지를 저장하지 않는 함수
# =====================================
def get_original_and_mask_image(data):
  response = requests.get(data['thumbnail_url'])
  xml_data = requests.get(data['sheet_url']).content.decode("utf-8")
  json_file = xml_to_dict.XMLtoDict().parse(xml_data)

  if response.status_code == 200:
      img = Image.open(BytesIO(response.content))
      _bbox_list = get_bbox_list(json_file, img)
      
      if img.mode == 'RGB':
          img = img.convert('RGBA')

      # 이미지를 PNG로 변환 (메모리 상에서)
      original_img = BytesIO()
      img.save(original_img, format='PNG')

      # BytesIO를 다시 바이트 데이터로 되돌리기
      final_original_img = original_img.getvalue()
      
      mask_img = masking(
        img=img,
        bbox_list=_bbox_list if _bbox_list != None else [],
        offset=10
      )
      
      bytes_mask_img = BytesIO()
      mask_img.save(bytes_mask_img, format="PNG")
      
      final_mask_img = bytes_mask_img.getvalue()
  else:
      print('There is no response!')
      
  return final_original_img, final_mask_img

# =====================================
# 이미지를 저장하는 함수
# =====================================
def save_original_and_mask_image(data, original_path, mask_path):
  response = requests.get(data['thumbnail_url'])

  if response.status_code == 200:
      img = Image.open(BytesIO(response.content))
      
      if img.mode == 'RGB':
          original_img = img.convert('RGBA')

      # 원본 이미지를 .png로 저장
      original_img.save(original_path)
      
      mask_img = masking(
        img=original_img,
        bbox_list=get_bbox_list(data),
        offset=10
      )
      
      # mask 이미지 .png로 저장
      mask_img.save(mask_path)
      
  else:
      print('There is no response!')

def get_text_removed_imageURL(original_img, mask_img, topk=1, s=512):
  edited_img = openai.Image.create_edit(
    image=original_img,
    mask=mask_img,
    prompt="Remove all the texts.",
    n=topk,
    size=f"{s}x{s}"
  )
  
  '''
  edited_img 형식 
  {
    data : [
      {
        url: "string"
      },
      {
        url: "string"
      },
      {
        url: ... 
      } ==> url은 topk 개수만큼 존재함 
    ]
  }
  '''
  return edited_img["data"]

if __name__ == "__main__" :
  pickle_folder_path = "../data"
  pickle_file = "processed_15545.pickle"

  path = os.path.join(pickle_folder_path, pickle_file)
  with open(path, 'rb') as f:
    obj = pickle.load(f)
    json_data = json.loads(json.dumps(obj, default=str))

  # 이미지를 저장하지 않는 시나리오
  original_img, mask_img = get_original_and_mask_image(json_data)
  
  url = get_text_removed_imageURL(
    original_img=original_img,
    mask_img=mask_img,
    topk=1,
    s=512
  )
  
  print(url)
  
  # # 이미지를 저장하고 불러오는 시나리오
  # data_path = "../data"
  # original_path = os.path.join(data_path, "original")
  # mask_path = os.path.join(data_path, "mask")

  # # 폴더가 존재하지 않으면 폴더를 생성
  # if not os.path.exists(original_path):
  #     os.makedirs(original_path)
  # if not os.path.exists(mask_path):
  #     os.makedirs(mask_path)
  
  # img_name = "{0}_{1}.png".format(json_data["template_idx"], json_data["page_no"])
  # original_img_path = os.path.join(original_path, img_name)
  # mask_img_path = os.path.join(mask_path, img_name)
  # if not os.path.exists(original_img_path) or not os.path.exists(mask_img_path):
  #   save_original_and_mask_image(
  #     data=json_data,
  #     original_path=original_img_path,
  #     mask_path=mask_img_path
  #   )
    
  # url = get_text_removed_imageURL(
  #   original_img=open(original_img_path, "rb"),
  #   mask_img=open(mask_img_path, "rb"),
  #   topk=1,
  #   s=512
  # )
  
  # print(url)