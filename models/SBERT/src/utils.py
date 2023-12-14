from zipfile import ZipFile
import os
import pickle
import torch
from typing import List, Tuple
import numpy as np

def has_five_digits(string:str) -> bool:
    """
    check whehter string has more than 5 digits (for check dummy keywords)
    """
    count = 0
    for char in string:
        if char.isdigit():
            count += 1
    return count >= 5


def is_valid_keyword(keyword:str) -> bool:
  """
  check whether keyword is valid (not dummy keyword)
  """
  if keyword.isdigit():
    return False
  if has_five_digits(keyword):
    return False
  return True


def unzip_data(data_path:str, extract_path:str)->None:
    """
    unzip data to predefined path
    """
    with ZipFile(data_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

"""process data. if data doesn't exist, unzip folder"""
def processing_data(folder_path, data_path=None, extract_path=None) -> Tuple[List[List[str]],List[List[str]],List[dict]]:
    """
    keyword_list example: [ [key1, key2, key3], [key4, key5, key6], ... ]
    sentence_list example: [ [set1, set2, set3], [set4, set5, set6], ... ]
    data_list: list of xml dict
    """
    
    #unzip zip file into pickle if needed
    if (not os.path.exists(folder_path)):
        unzip_data(data_path=data_path, extract_path=extract_path)

    #get data list
    file_names = os.listdir(folder_path)

    #read pickle file and save as list
    data_list = []
    for file_name in file_names:
        if file_name.startswith("processed_") and file_name.endswith(".pickle"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                data_list.append(data)
    
    keyword_list = []
    text_list = []
    new_data_list = []

    #change unprocessed json list to keyword/text/data list
    for i in range(len(data_list)):
        #check keyword valide
        keyword = data_list[i]['keyword']
        keyword = [word for word in keyword if is_valid_keyword(word)]
        
        #error case
        if len(keyword) == 0:
            continue
        
        keyword_list.append(keyword)
        text_list.append([])
        
        #only save layout text info in text_list
        for j in range(len(data_list[i]['form'])):
            if type(data_list[i]['form'][j]['text']) == str:
                text = data_list[i]['form'][j]['text']
                text_list[i].append(text)
        
        #in new_data_list, save all information
        new_data_list.append(data_list[i])

    data_list = new_data_list
    
    return keyword_list, text_list, data_list

def i2t(npts:int, sims:np.array, return_ranks=False) -> Tuple(float,float,float,float,float):
    """
    our metric for evaluate SBERT performance. \n
    int our case, image==xml text, caption==keyword
    """
    
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (N, max_n_word, d) matrix of captions
    CapLens: (N) array of caption lengths
    sims: (N, 1) matrix of similarity im-cap
    """
    
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        rank = np.where(inds == index)[0][0]
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)