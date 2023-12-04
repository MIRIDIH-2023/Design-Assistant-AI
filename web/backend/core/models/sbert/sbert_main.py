from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import os

from .dalle import *
from typing import Any, Dict

from PIL import Image
import requests
from io import BytesIO

MODEL_PATH = "core/models/sbert/weights"
MODEL_NAME = "sbert_keyword_extractor_2023_07_18"
DATA_PATH = "core/models/sbert/weights/data"
DATA_LIST_PATH = os.path.join( DATA_PATH , "data_list.pickle")
EMBEDDING_LIST_PATH = os.path.join( DATA_PATH , "embedding_list.pth")
KEYWORD_LIST_PATH = os.path.join( DATA_PATH , "keyword_embedding_list.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class SBERT():
    def __init__(self):
        self.model = SentenceTransformer(os.path.join(MODEL_PATH , MODEL_NAME) )
        with open(DATA_LIST_PATH, 'rb') as file:
            self.data_list = pickle.load(file)
        self.embedding_list = torch.load(EMBEDDING_LIST_PATH)
        self.keyword_embedding_list = torch.load(KEYWORD_LIST_PATH)
        
        self.text_list = None
        self.make_text_list()
        
    def make_text_list(self):
        text_list = []
        for i in range(len(self.data_list)):
            text = ""
            for j in range(len(self.data_list[i]['form'])):
                if type(self.data_list[i]['form'][j]['text']) == str:
                    text += " " + self.data_list[i]['form'][j]['text']
            text_list.append(text)
        self.text_list = text_list
    
    # Return [(index, text, score), ...]
    def get_most_similiar_template(self, user_input, top_k=3, compare_with='max'):
        user_embedding = self.model.encode(user_input, convert_to_tensor=True)
        if compare_with == 'text':
            cos_scores = util.pytorch_cos_sim(user_embedding, self.embedding_list)[0]
        elif compare_with == 'keyword':
            cos_scores = util.pytorch_cos_sim(user_embedding, self.keyword_embedding_list)[0]
        elif compare_with == 'sum':
            cos_scores = util.pytorch_cos_sim(user_embedding, self.embedding_list + self.keyword_embedding_list)[0]
        elif compare_with == 'max':
            text_score = util.pytorch_cos_sim(user_embedding, self.embedding_list)[0]
            keyword_score = util.pytorch_cos_sim(user_embedding, self.keyword_embedding_list)[0]
            cos_scores = torch.max(text_score, keyword_score)
        else:
            raise Exception('compare_with must be one of text, keyword, sum, max')
        cos_scores = cos_scores.cpu()

        top_results = torch.topk(cos_scores, k=1000)

        result = []
        for score, idx in zip(top_results[0], top_results[1]):
            if len(result)==top_k:
                break
            if self.check_valid(idx.item()):
                result.append((idx.item(), self.text_list[idx], score.item()))
        
        return result
    
    def check_valid(self, idx):
        image_url = self.data_list[idx]['thumbnail_url']
        image = Image.open(BytesIO(requests.get(image_url)))
        # 이미지의 가로와 세로 크기를 가져옵니다.
        width, height = image.size

        # 가로 세로 비율을 계산합니다.
        aspect_ratio = width / height
        
        if aspect_ratio>=0.74:
            return True
        else:
            return False
        
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method recommend image from input text. 

        Args:
            data (`Dict[str, Any]`): A dictionary containing the following keys:
                - text (`List[str]`): 
                    A list of strings, each string is a sentence.
                    ex) ['This is a sentence.', 'This is another sentence.']
                - num_recommend ('int'):
                    how many urls to return
                    ex) 20
                - use_dalle ('boolean'):
                    whether use dalle or not.
                    use dalle when making background
                    do not use dalle when just thumnail recommend
                    ex) True, False

        Returns:
            prediction (`Dict[str, Any]`): A dictionary containing the following keys:
                - image_url (`List[str]`):
                    list of string that contains image url
                    ex) [ 'this-is-url-1' , 'this-is-url-2' ]
        """
    
        text = data['text']
        num_recommend = data['num_recommend']
        use_dalle = data['use_dalle']
        
        sbert_input_text = ""
        #add \n in every line
        for line in text:
            sbert_input_text = sbert_input_text + line + "\n"
        
        results = self.get_most_similiar_template(user_input=sbert_input_text, top_k=num_recommend, compare_with='max')
        
        recommend_idx = []
        thumbnail_urls = []
        for cur_idx, cur_text, cur_score in results:
            thumbnail_urls.append(self.data_list[cur_idx]['thumbnail_url'])
            recommend_idx.append(cur_idx)
        
        
        output = {}
        #when use dalle (background recommendation)
        if use_dalle:
            output['image_url'] = make_dalle_url(self.data_list , recommend_idx)
        #when do not use dalle (thumnail search engine)
        else:
            output['image_url'] = thumbnail_urls
        
        return output