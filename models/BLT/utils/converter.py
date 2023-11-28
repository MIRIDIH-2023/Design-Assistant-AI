import pickle
import json
import math
import os
from tqdm import tqdm

def load_text_infos(json_data, im_width) :
    text_area = 0
    text_coord_lr = []
    text_coord_tb = []
    text_info_pair = []
    text_size_db = set()
    for text_info in json_data["form"] :
        # rotate가 0이 아니고, text의 내용이 None이거나 비어있으면 데이터로 사용하지 않음
        if(text_info["rotate"] != "0" or text_info["text"] == None or text_info["text"] == "" or text_info["text"] == " ") : continue
        
        # key가 "box"인 경우 RendorPos값에 해당, key가 "raw_bbox"인 경우 Position값
        # 값들은 thumnail image 크기에 맞춰서 조정된 값임
        left, top, right, bottom = text_info["box"]

        center_x, center_y = float((left + right) / 2), float((top + bottom) / 2)
        width_ = right - left
        height_ = bottom - top

        # data가 이상할 수도 있는 혹시 모를 방지 
        if (width_ * height_) == 0 : continue

        doc_width , doc_height = text_info["sheet_size"]
        RATIO = im_width / doc_width

        # text size와 line space 추출
        ts = min(text_info["font_size"])
        ls = min(text_info["linespace"])
        # bounding box를 나타내는 좌표가 thumnail image 크기에 맞춰서 조정되었기 때문에
        # text size와 line space도 bbox와 동일하게 image크기에 맞춰줌
        ratio_text_size = ts * RATIO
        line_space = ls * RATIO

        if ratio_text_size <= 0 : continue

        # text의 줄 수를 알아내기 위한 알고리즘 
        # (개행 문자로 나눠서 구하는 방식 + text size와 line space로 bbox 높이를 나눠서 계산하는 방식)
        # 단순 계산하는 방식을 보완하기 위해서 개행 문자로 나누는 방식을 추가한 것
        # 개행 문자('\n') 을 기준으로 나눔
        split_cnt1 = len(text_info["text"].split('\n'))
        split_cnt2 = len(text_info["text"].split('\\n'))
        split_cnt = split_cnt1 if split_cnt1 > split_cnt2 else split_cnt2
        # line space 값이 음수인 경우가 있음.
        # 나누는 수가 음수이거나 0이면 에러가 남으로 방지하기 위함
        if (ratio_text_size + line_space) <= 0 : line_space = 0
        line_nums = int(math.floor(float(height_ / (ratio_text_size + line_space))))
        line_nums = line_nums if line_nums > 0 else 1
        
        # 2가지 방식으로 구한 값을 비교해서 더 큰 값을 최종 줄 수로 정함
        line_nums = split_cnt if split_cnt >= line_nums else line_nums

        # text_size는 XML 데이터의 원본 값임
        text_size = int(math.floor(ts))
        if text_size <= 0 : continue

        # 각 text 정보 쌍을 만듬
        # 쌍{text 크기(XML 원본), text 줄 수, text bounding box정보(thumbnail image 크기에 맞게 조정된 값)}
        text_info_pair.append((text_size, line_nums, center_x, center_y, width_, height_))

        # text bbox 좌표 저장하기
        text_coord_lr.append([left, right])
        text_coord_tb.append([top, bottom])

        # text 영역 넓이 구하기
        text_area += width_ * height_

        # text size db 저장
        text_size_db.add(text_size)

    text_infos = {
        "text_area" : text_area,
        "text_coord_lr" : text_coord_lr,
        "text_coord_tb" : text_coord_tb,
        "text_info_pair" : text_info_pair,
        "text_size_db" : text_size_db
    }

    return text_infos

# 이미지와 텍스트가 겹치는 정도에 따라 레이블을 반환
def get_image_label(image_info, text_infos):
    max_overlap_ratio = 0 # 이미지와 가장 많이 겹치는 텍스트의 겹치는 정도
    image_box = image_info["box"]

    for i in range(len(text_infos['text_coord_lr'])):
        text_box = [text_infos['text_coord_lr'][i][0],
                    text_infos['text_coord_tb'][i][0],
                    text_infos['text_coord_lr'][i][1],
                    text_infos['text_coord_tb'][i][1]]
        text_area = (text_box[2] - text_box[0]) * (text_box[3] - text_box[1])

        overlap_x = max(0, min(image_box[2], text_box[2]) - max(image_box[0], text_box[0]))
        overlap_y = max(0, min(image_box[3], text_box[3]) - max(image_box[1], text_box[1]))
        overlap_area = overlap_x * overlap_y

        #가능한 최대 겹침 비율(나중에 쓸 수도 있음)
        image_area = abs(image_box[2]-image_box[0]) * abs(image_box[3]-image_box[1])
        possible_max_overlap_area_per_text =  image_area / text_area

        if text_area == 0:
          continue
        overlap_ratio = overlap_area / text_area

        if overlap_ratio > max_overlap_ratio:
            max_overlap_ratio = overlap_ratio

    if max_overlap_ratio > 0.8:
        return "image_with_text"
    elif max_overlap_ratio > 0.2:
        return "image_partially_with_text"
    else:
        return "image_without_text"

# path: data가 있는 폴더 경로
'''
label_names: 사용하려는 XML tag이름
label_to_id: label에 할당된 id값이 저장된 dictionary
dataset/categorized_info.py 참고
'''
# idx(int): test시 사용되는 변수, 데이터 리스트에서 어떤 데이터를 뽑아 시각화할지 명시하는 값
# with_background_test: inference 결과를 시각화할 때 thumbnail이미지를 넣을 것인지 명시
def load_categorized(path, label_names, label_to_id, idx, with_background_test=False, composition="defualt") :
    data = []
    image_link=None

    file_name_lst = os.listdir(path)
    # 시각화할 경우 file 1개만 가져옴 
    file_name_lst = [file_name_lst[idx]] if with_background_test else tqdm(file_name_lst) 
    for file_name in file_name_lst :
        # pickle file open
        file_path = os.path.join(path, file_name)
        with open(file_path, 'rb') as f:
            try:
                obj = pickle.load(f)
                json_data = json.loads(json.dumps(obj, default=str))
            except:
                raise AssertionError(f"Wrong file: {file_path}")

        im_width, im_height = json_data["thumbnail_size"]
        template = {
            "children" : [],
            "width" : im_width,
            "height" : im_height,
        }

        # text 정보를 가져옴
        text_infos =  load_text_infos(json_data=json_data, im_width=im_width)

        text_area = text_infos["text_area"]
        text_coord_lr = text_infos["text_coord_lr"]
        text_coord_tb = text_infos["text_coord_tb"]
        text_info_pair = text_infos["text_info_pair"]
        text_size_db = text_infos["text_size_db"]

        # text size db 내림차순 정렬하기 
        text_size_db = list(text_size_db)
        text_size_db.sort(reverse=True)
        # text size list가 4개보다 많을 경우, 4개만 사용함
        # 예시: [60, 40, 30, 20, 10, 8] => [60, 40, 30, 20]
        text_size_db = text_size_db if len(text_size_db) <= 4 else text_size_db[:4]

        # text data category화 하여 저장
        # text_info_pair의 원소: (text 크기, text 줄 수, bounding box정보(x, y, w, h))
        for pair in text_info_pair :
            # label 이름 참고(label_names)
            label = "text_"
            # text 크기 별로 category를 부여함 가장 큰 text는 "text_0_"
            # text 줄 수별로 추가로 category를 부여함 text_0_2
            # text_0_2의 의미: template내에서 가장 크기가 큰 text이면서 줄 수가 2
            for idx, size in enumerate(text_size_db) :
                if pair[0] >= size :
                    label += f"{idx}_{pair[1]}" if pair[1] < 4 else f"{idx}_3_over"
                    break
                
                if idx == 3 and label == "text_" :
                    label += f"3_{pair[1]}" if pair[1] < 4 else "3_3_over"

            
            # data sequence 구성 - XML 요소 별 bbox 정보 활용 
            # default : <category id> <width> <height> <center x> <center y>
            # ltwh : <category id> <width> <height> <left top x> <left top y>
            # ltrb : <category id> <right bottom x> <right bottom y> <center x> <center y>
            template["children"].append({
                "category_id" : label_to_id[label],
                "center" : [(pair[2]-pair[4]/2.), (pair[3]-pair[5]/2.)] if composition == "ltwh" or composition == "ltrb" else [pair[2], pair[3]],
                "width" : (pair[2]+pair[4]/2.) if composition == "ltrb" else pair[4],
                "height" : (pair[3]+pair[5]/2.) if composition == "ltrb" else pair[5]
            })

        # text이외의 tag들을 category화하는 알고리즘
        for info in json_data["tags_info"] :
            # rotate가 0이고 필요한 XML tag만 category화에 사용함
            if info["tag"] not in label_names or info["rotate"] != "0": continue
            
            # image크기에 맞게 조정된 값
            left, top, right, bottom = info["box"]
            center_x, center_y = float((left + right) / 2), float((top + bottom) / 2)
            width_ = right - left
            height_ = bottom - top

            # data가 이상할 수도 있는 거 혹시 모를 방지 
            if (width_ * height_) == 0 : continue
            
            # text의 bounding box넓이의 반보다 큰 것만 사용함
            # 작은 것들은 꾸미는 용도(sticker 등)이고 크게 template 내에서 비중을 차지하지 않는다는 가정
            if (width_ * height_) < (text_area / 2) : continue

            # text의 bounding box와 겹치는 것으로 image와 background로 SVG, SHAPESVG 등의 tag를 분류함
            # image: text의 bounding box와 겹치지 않는 것
            # background: text의 bounding box와 겹치지 않으면서 template내의 모든 text를 포함하는 것
            lr_dup = False
            tb_dup = False
            is_back_lr = True
            is_back_tb = True
            for lr, tb in zip(text_coord_lr, text_coord_tb) :
                if (lr[0] < left and left < lr[1]) or (lr[0] < right and right < lr[1]) :
                    lr_dup = True
                if (tb[0] < top and top < tb[1]) or (tb[0] < bottom and bottom < tb[1]) :
                    tb_dup = True
                if (lr[0] < left or right < lr[1]) :
                    is_back_lr = False
                if (tb[0] < top or bottom < tb[1]) :
                    is_back_tb = False

            # 아래 주석 처리한 코드는 이미지 category 종류를 세분화한 실험부터 사용하지 않음
            # image가 text의 bounding box의 좌표가 모두 겹치는 경우 사용하지 않는다는 코드 
            # if lr_dup and tb_dup : continue

            # XML tag중 chart와 grid는 그대로 category로 사용
            if info["tag"] == "Chart" or info["tag"] == "GRID":
                label = info["tag"]
            else :
                if is_back_lr and is_back_tb :
                    label = "background"
                else :
                    label = get_image_label(info, text_infos)

            # 아래 코드는 이미지 category 종류를 세분화한 실험부터 사용함
            # category가 background로 분류될 경우 학습에 사용하지 않음 
            if label == "background" : continue

            # data sequence 구성 - XML 요소 별 bbox 정보 활용 
            # default : <category id> <width> <height> <center x> <center y>
            # ltwh : <category id> <width> <height> <left top x> <left top y>
            # ltrb : <category id> <right bottom x> <right bottom y> <center x> <center y>
            template["children"].append({
                "category_id" : label_to_id[label],
                "center" : [(center_x-width_/2.), (center_y-height_/2.)] if composition == "ltwh" or composition == "ltrb" else [center_x, center_y],
                "width" : (center_x+width_/2.) if composition == "ltrb" else width_,
                "height" : (center_y+height_/2.) if composition == "ltrb" else height_
            })

        if with_background_test : image_link = json_data["thumbnail_url"]

        # XML tag가 분류되지 않아서 template 데이터를 만들 수 없는 경우 학습에 사용하지 않음 
        if len(template["children"]) == 0 : continue
        
        data.append(template)
    
    return data, image_link


# 기업에서 제공한 XML tag를 분류하지 않고 실험할 때 사용하는 함수
# path: data가 있는 폴더 경로
'''
label_names: 사용하려는 XML tag이름
label_to_id: label에 할당된 id값이 저장된 dictionary
dataset/categorized_info.py 참고
'''
# idx(int): test시 사용되는 변수, 데이터 리스트에서 어떤 데이터를 뽑아 시각화할지 명시하는 값
# with_background_test: inference 결과를 시각화할 때 thumbnail이미지를 넣을 것인지 명시
def miri_load(path, label_names, label_to_id, idx, with_background_test=False, composition="defualt") :
    data = []
    image_link=None
    file_name_lst = tqdm(os.listdir(path))
    file_name_lst = [file_name_lst[idx]] if with_background_test else tqdm(file_name_lst) 
    for file_name in file_name_lst :
        # pickle file open
        file_path = os.path.join(path, file_name)
        with open(file_path, 'rb') as f:
            try:
                obj = pickle.load(f)
                json_data = json.loads(json.dumps(obj, default=str))
            except:
                raise AssertionError(f"Wrong file: {file_path}")

        template = {
            "children" : [],
            "width" : json_data["thumbnail_size"][0],
            "height" : json_data["thumbnail_size"][1],
        }

        # load tags
        for info in json_data["tags_info"] :
            if info["tag"] not in label_names: continue
            
            left, top, right, bottom = info["box"]
            center_x, center_y = float((left + right) / 2), float((top + bottom) / 2)
            width_ = right - left
            height_ = bottom - top
            template["children"].append({
                "category_id" : label_to_id[info["tag"]],
                "center" : [center_x, center_y],
                "width" : width_,
                "height" : height_
            })

        # load texts
        for text_info in json_data["form"] :
            left, top, right, bottom = text_info["raw_bbox"]
            center_x, center_y = float((left + right) / 2), float((top + bottom) / 2)
            width_ = right - left
            height_ = bottom - top
            template["children"].append({
                "category_id" : label_to_id["TEXT"],
                "center" : [center_x, center_y],
                "width" : width_,
                "height" : height_
            })

        if with_background_test : image_link = json_data["thumbnail_url"]

        if len(template["children"]) == 0 : continue

        data.append(template)

    return data, image_link
