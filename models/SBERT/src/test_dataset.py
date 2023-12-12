from sentence_transformers import SentenceTransformer
from typing import Tuple,List
from torch.utils.data import DataLoader, Dataset

class test_dataset(Dataset):
    """
    test dataset doesn't have negative pair. just positive pair
    """
    def __init__(self, keys, texts, datas=None) -> None:
        super().__init__()
        self.keys = keys # [# of data, # of keyword]
        self.texts = texts # [# of data , # of sentences]
        self.datas = datas # [# of data]
    
    def __len__(self) -> int:
        return len(self.keys)
    
    def __getitem__(self, index:int) -> Tuple[str,str]:
        
        sentence_1, sentence_2 = self.get_positive_pair(index)
        
        return sentence_1, sentence_2
    
    def get_positive_pair(self,index:int) -> Tuple[str, str]:
        """
        convert list into string and return
        """
        cur_text = self.texts[index]
        cur_key = self.keys[index]
        
        cur_text = ' '.join(cur_text)
        cur_key = ' '.join(cur_key)
        
        return cur_text, cur_key
        
