import splitfolders
from tqdm import tqdm
import shutil
import os
import pickle
import json

source_path = "/home/work/increased_en_data/BLT/all_data"
renderPos_data_path = "/home/work/increased_en_data/BLT/posData"
splitted_data_path = "/home/work/increased_en_data/BLT/data2"
no_renderPos_data_list = []

assert os.path.exists(source_path)
assert os.path.exists(renderPos_data_path) and len(os.listdir(renderPos_data_path)) == 0

# /renderPos_data_path/json_data/ 경로에 데이터를 복사 붙여넣기 함
target_path = os.path.join(renderPos_data_path, "json_data")
os.makedirs(target_path, exist_ok=True)

assert os.path.exists(splitted_data_path) and len(os.listdir(splitted_data_path)) == 0

# renderPos data만 따로 저장하는 로직
files = os.listdir(source_path)
assert len(files) != 0
for file_ in tqdm(files) :
    source_file = os.path.join(source_path, file_)
    with open(source_file, 'rb') as f:
        try:
            obj = pickle.load(f)
            json_data = json.loads(json.dumps(obj, default=str))
        except:
            raise AssertionError(f"Wrong file: {source_file}")
    if(not json_data["no_RendorPos"]) :
        shutil.copy(source_file, target_path)
    else :
        no_renderPos_data_list.append(file_)

# train, eval, test 폴더를 생성해 8:1:1로 나눠서 분리하는 함수
# input 폴더 조건
'''
/renderPos_data_path
        |___ json_data
'''
# output 폴더 결과
'''
/splitted_data_path
        |___train
                |___ json_data
        |___eval
                |___ json_data
        |___test
                |___ json_data
'''
tqdm(splitfolders.ratio(renderPos_data_path, output=splitted_data_path, seed=77, ratio=(0.8, 0.1, 0.1)))

data = { "no_render_pos" : no_renderPos_data_list }

file_path = os.path.join(splitted_data_path, "no_renderPos_data_file_list.json")
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, indent="\t")

# count dir files 
print("test: ", len(os.listdir(os.path.join(splitted_data_path, "test", "json_data"))))
print("val: ",len(os.listdir(os.path.join(splitted_data_path, "val", "json_data"))))
print("train: ",len(os.listdir(os.path.join(splitted_data_path, "train", "json_data"))))
print("total data: ", len(os.listdir(os.path.join(renderPos_data_path, "json_data"))))