from .utils import processing_data
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
from transformers import AutoTokenizer
from typing import Tuple,List
from sentence_transformers import SentenceTransformer, InputExample

class custom_dataset(Dataset):
    """
    our custom dataset with augmentation. \n
    there are two augmentations. \n
    1. masking ratio (self.drop_prob) \n
    2. shuffle texts/keywords
    """
    def __init__(self, keys, texts, datas) -> None:
        super().__init__()
        self.keys = keys # [# of data, # of keyword]
        self.texts = texts # [# of data , # of sentences]
        self.datas = datas # [# of data]
        self.tokenizer = SentenceTransformer("distiluse-base-multilingual-cased-v1").tokenizer
        self.drop_prob = 0.2

    def __len__(self) -> int:
        #do not use front 1000 datas
        return 2 * len(self.keys) - 2000

    def __getitem__(self, index:int) -> InputExample:
        index = index + 2000
        
        #when positive
        if index%2==0:

            sentence_1, sentence_2 = self.get_positive_pair(index//2)
            sim = 1

        else:

            sentence_1, sentence_2 = self.get_negative_pair(index//2)
            sim = 0

        output = {
            "texts" : [sentence_1, sentence_2],
            "label" : sim
        }

        return InputExample(texts=[output['texts'][0], output['texts'][1]], label=float(output['label']))

    def get_positive_pair(self,index):
        """
        text는 순서 유지하는게 잘 뽑히도록 weighted shuffle
        keyword는 무작위 shuffle

        (cur_text, cur_key) 모두 리스트
        Args:
            index (_type_): _description_
        """
        #cur_text = self.weighted_shuffle(self.texts[index])
        cur_text = self.texts[index]
        cur_key = self.keys[index]
        np.random.shuffle(cur_key)

        masked_text, masked_key = self.tokenize_and_masking(cur_text, cur_key)

        return masked_text, masked_key

    def get_negative_pair(self,index: int) -> Tuple[str, str]:
        
        #cur_text = self.weighted_shuffle(self.texts[index])
        cur_text = self.texts[index] # -> List[str]
        cur_key = self.keys[index] # -> List[str]

        #keyword가 안겹치도록 negative sampling
        negative_keyword = random.choice(self.keys)
        while set(negative_keyword).intersection(set(cur_key)):
            negative_keyword = random.choice(self.keys)
        cur_key = negative_keyword

        masked_text, masked_key = self.tokenize_and_masking(cur_text, cur_key)

        return masked_text, masked_key

    def tokenize_and_masking(self, cur_text: List[str], cur_key: List[str]) -> Tuple[str, str]:
        cur_text = '\n'.join(cur_text)
        cur_key = ' '.join(cur_key)

        token_text = self.tokenizer.tokenize(cur_text)
        token_key = self.tokenizer.tokenize(cur_key)

        id_text = self.tokenizer.convert_tokens_to_ids(token_text)
        id_key = self.tokenizer.convert_tokens_to_ids(token_key)

        masked_text = self.caption_augmentation(id_text)
        masked_key = self.caption_augmentation(id_key)

        text = self.tokenizer.decode(masked_text, skip_special_tokens=False)
        key = self.tokenizer.decode(masked_key, skip_special_tokens=False)

        return (text,key)

    def caption_augmentation(self,tokens: List[str]) -> List[str]:
        """
        mask some tokens in the sentence

        Args:
            tokens (_type_): list of tokens

        Returns:
            _type_: list of tokens (some of tokens are masked)
        """
        idxs = []
        mask_idx = self.tokenizer.convert_tokens_to_ids('[MASK]')
        vocab_len = self.tokenizer.vocab_size

        for t in tokens:
            prob = random.random()
            if prob < self.drop_prob:
                prob /= self.drop_prob
                if prob < 0.5:
                    idxs += [mask_idx]
                elif prob < 0.6:
                    idxs += [random.randrange(vocab_len)]
            else:
                idxs += [t]
        return idxs

    def weighted_shuffle(self,arr: List[str]) -> List[str]:
        """
        문장(text) shuffle시 순서 유지하는게 잘 뽑히도록 가중치 부여

        Args:
            arr (_type_): original list

        Returns:
            _type_: shuffled list
        """
        arr = arr.copy()
        shuffled = []
        while arr:
            weights = np.arange(len(arr), 0, -1)
            weights = weights / np.sum(weights)
            choice = np.random.choice(len(arr), p=weights)
            shuffled.append(arr.pop(choice))
        return shuffled