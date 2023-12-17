# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plot layout."""
from typing import Sequence, Tuple, Union
import matplotlib.pyplot as plt
# import cv2

from dataset import coco_info
from dataset import magazine_info
from dataset import publaynet_info
from dataset import rico_info
from dataset import miri_info
from dataset import categorized_info

import numpy as np
import tensorflow as tf
import matplotlib.patches as patches
from PIL import Image, ImageFilter
from PIL import ImageFont
from PIL import ImageDraw

from io import BytesIO
import requests
import os


def parse_entry(
    dataset, entry
):
  """Parses a dataset entry according to its dataset.

  Args:
    dataset: Name of the dataset type.
    entry: [asset_dim] (class_id, width, height, center_x, center_y). Entry in
      the layoutvae network output format.

  Returns:
    A tuple with the class id, the class name, an associated color, and the
      bounding box.
  """
  if dataset == "RICO":
    info = rico_info
  elif dataset == "PubLayNet":
    info = publaynet_info
  elif dataset == "MAGAZINE":
    info = magazine_info
  elif dataset == "COCO":
    info = coco_info
  elif dataset == "MIRI" :
    info = miri_info
  elif dataset == "CATEGORIZED" :
    info = categorized_info
  else:
    raise ValueError(f"Dataset '{dataset}' not found")
  class_id = entry[0]
  class_name = info.ID_TO_LABEL[class_id]
  color = info.COLORS[class_name]
  bounding_box = entry[1:]

  return class_id, class_name, color, bounding_box


def parse_layout_sample(data, dataset_type):
  """Decode to a sequence of bounding boxes."""
  result = {}
  for idx in range(0, data.shape[-1], 5):
    entry = data[idx:idx+5]
    _, class_name, _, bounding_box = parse_entry(dataset_type, entry)

    width, height, center_x, center_y = bounding_box
    # Adds a small number to make sure .5 can be rounded to 1.
    x_min = np.round(center_x - width / 2. + 1e-4)
    x_max = np.round(center_x + width / 2. + 1e-4)
    y_min = np.round(center_y - height / 2. + 1e-4)
    y_max = np.round(center_y + height / 2. + 1e-4)

    x_min = np.clip(x_min / 31., 0., 1.)
    y_min = np.clip(y_min / 31., 0., 1.)
    x_max = np.clip(x_max / 31., 0., 1.)
    y_max = np.clip(y_max / 31., 0., 1.)
    result[class_name] = [np.clip(bounding_box / 31., 0., 1.),
                          [y_min, x_min, y_max, x_max]]
  return result

# image위에 bbox layout을 그릴 때 색깔을 넣기 위해 만든 함수
# PIL로 대체하면서 사용하지 않음
def rgb_to_hex(rgb):
    r, g, b = map(int, rgb)
    if 15 < r and r < 256 : r_hex = hex(r)[2:]
    elif r < 16 : r_hex = "0{}".format(hex(r)[2:])
    else : r_hex = "00"
    if 15 < g and g < 256 : g_hex = hex(g)[2:]
    elif g < 16 : g_hex = "0{}".format(hex(g)[2:])
    else : g_hex = "00"
    if 15 < b and b < 256 : b_hex = hex(b)[2:]
    elif b < 16 : b_hex = "0{}".format(hex(b)[2:])
    else : b_hex = "00"

    return "#{0}{1}{2}".format(r_hex, g_hex, b_hex)

# 시각화한 결과 이미지를 workdir 이름에 따라서 저장하기 위해 폴더를 만드는 함수 
def create_folder(conditional, exp, base_path):
    # 폴더 경로 생성
    # /base_path/실험 번호(exp6)/a/
    # /base_path/실험 번호(exp6)/a_s/
    # exp = exp.split('/')[1]
    if conditional == "a":
        folder_path = os.path.join(base_path, exp.split('_')[0], conditional)
    elif conditional == "a+s":
        folder_path = os.path.join(base_path, exp.split('_')[0], conditional.replace('+', '_'))
    else :
        folder_path = os.path.join(base_path, exp.split('_')[0], conditional)
    
    # 폴더 생성
    os.makedirs(folder_path, exist_ok=True)

    return folder_path

def plot_sample_with_PIL(data,
                workdir,
                base_path,
                dataset_type="CATEGORIZED",
                border_size=1,
                thickness=4,
                im_type="no_input",
                idx=None,
                image_link=None,
                conditional="a",
                composition="default"):
    """Draws an image from a sequence of bounding boxes.

    Args:
        data: A sequence of bounding boxes. They must be in the 'networks output'
        workdir: The name of the experiment.
        base_path: The path to the folder where the results will be saved.
        dataset_type: Dataset type keyword. Necessary to assign labels. "CATEGORIZED", "MIRI"
        im_type: The type of the image. "no_input", "input", "infer"
        idx: The index of the image in the dataset.
        image_link: The link to the image.
        conditional: The type of the conditional. "a", "a+s", "custom"
        composition: The type of the composition. "default", "ltwh", "ltrb"
        target_width: Result image width.
        target_height: Result image height.
        dataset_type: Dataset type keyword. Necessary to assign labels.
        thickness: It is the thickness of the rectangle border line in px.
        Thickness of -1 px will display each box with a colored box without text.
    """
    image = None
    if image_link is not None and idx is not None :
      try:
        image = Image.open(BytesIO(requests.get(image_link).content))
        target_width, target_height = image.size
        # inference 결과를 시각화한 배경은 원본 이미지 대신 흰 배경으로 대체 
        if im_type.endswith("_infer") : image = Image.new("RGB", (target_width, target_height), "white")
      except:
          print(f"Error at loading image, {image_link}")
          image = None
    else :
      image = Image.new("RGB", (500, 500), "white")

    if image is None : return

    # pad 부분을 없애고 decode한 sequence부분만 추출 
    data = data[data >= 0]

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_color = (0, 0, 0)

    for idx in range(0, len(data), 5):
        entry = data[idx:idx + 5]
        _, class_name, color, bounding_box = parse_entry(dataset_type, entry)

        width, height, center_x, center_y = bounding_box
        # Adds a small number to make sure .5 can be rounded to 1.
        # 실험을 할 때 sequence 구성 방식에 따라서 시각화를 위한 데이터 복원도 다르게 함 
        if composition == "ltwh" :
          print("ltwh visualization")
          x_min = np.round(center_x + 1e-4)
          x_max = np.round(center_x + width + 1e-4)
          y_min = np.round(center_y + 1e-4)
          y_max = np.round(center_y + height + 1e-4)
        elif composition == "ltrb" :
          print("ltrb visualization")
          x_min = np.round(center_x + 1e-4)
          x_max = np.round(width + 1e-4)
          y_min = np.round(center_y + 1e-4)
          y_max = np.round(height + 1e-4)
        else :
          print("default visualization")
          x_min = np.round(center_x - width / 2. + 1e-4)
          x_max = np.round(center_x + width / 2. + 1e-4)
          y_min = np.round(center_y - height / 2. + 1e-4)
          y_max = np.round(center_y + height / 2. + 1e-4)

        # 0과 32사이로 discretization된 것에서 원본 이미지에 맞춰서 layout 값 복구
        x_min = round(np.clip(x_min / 31., 0., 1.) * target_width)
        y_min = round(np.clip(y_min / 31., 0., 1.) * target_height)
        x_max = round(np.clip(x_max / 31., 0., 1.) * target_width)
        y_max = round(np.clip(y_max / 31., 0., 1.) * target_height)

        print(class_name, (x_min, y_min), x_max - x_min, y_max - y_min)

        draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=thickness)
        draw.text((x_min, y_min), class_name, fill=text_color, font=font)

    # Save the image
    '''
      이미지가 저장되는 경로 예시
        base_path
            |____ result
                    |_____ exp1
                            |____ a
                                  |____ image.png
                            |____ a+s
                            |____ custom
                    |_____ exp2
                            |____ a
    '''
    basePath = os.path.join(base_path, "result")
    folder_path = create_folder(conditional, workdir, basePath)

    if folder_path is None : return

    image.save(os.path.join(folder_path, f"{im_type}.png"))