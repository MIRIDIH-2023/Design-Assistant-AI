"""
This module contains the main function for UDOP.
"""

from io import BytesIO
from typing import Any, Dict

import requests
import torch
from PIL import Image

from core.common.utils import str_to_img
from core.models.udop import (UdopConfig, UdopForConditionalGeneration,
                              UdopImageProcessor, UdopProcessor, UdopTokenizer)

TASK = "Layout Modeling."
MODEL_PATH = "core/models/udop/weights/noCurriculum"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UDOP():
    r"""
    This is the main class of UDOP to generate layout predictions.
    It combines the image processor, tokenizer, and a model.
    """
    def __init__(self):
        image_processor = UdopImageProcessor(apply_ocr=False)
        tokenizer = UdopTokenizer.from_pretrained("ArthurZ/udop", legacy=True)
        config = UdopConfig.from_pretrained("nielsr/udop-large")

        self.model = UdopForConditionalGeneration.from_pretrained(MODEL_PATH, config=config).to(DEVICE)
        self.processor = UdopProcessor(image_processor=image_processor, tokenizer=tokenizer)


    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method generates layout predictions for a given text and an optional image. 

        Args:
            data (`Dict[str, Any]`): A dictionary containing the following keys:
                - text (`List[str]`): 
                    A list of strings, each string is a sentence.
                    ex) ['This is a sentence.', 'This is another sentence.']
                - image (`str`, *optional*): 
                    A base64 encoded image.
                - url (`str`, *optional*): 
                    A url to an image.

        Returns:
            prediction (`Dict[str, Any]`): A dictionary containing the following keys:
                - layout (`str`):
                    A string containing the layout predictions.
                    ex) '<extra_l_id_0><loc_122><loc_53><loc_525><loc_68><extra_l_id_1> ...'
        """

        text = data['text']
        if 'image' in data:
            image = str_to_img(data['image'])
        elif 'url' in data:
            image = Image.open(BytesIO(requests.get(data['url']).content)).convert('RGB')
        else:
            image = Image.new('RGB', (224, 224), 'rgb(0, 0, 0)')


        # Prepare model inputs
        sentinel_idx = 0
        masked_text = []
        token_boxes = []

        # Mask sentences
        for sentence in text:
            masked_text.append(f'<extra_l_id_{sentinel_idx}>')
            masked_text.append(sentence)
            masked_text.append(f'</extra_l_id_{sentinel_idx}>')

            if sentinel_idx > 100:
                break
            sentinel_idx += 1

        token_boxes = [[0,0,0,0] for _ in range(len(masked_text))]

        # Encode inputs
        encoding = self.processor(images=image, text=[TASK], text_pair=[masked_text], boxes=[token_boxes], return_tensors="pt")
        encoding = {key: value.to(DEVICE) for key, value in encoding.items()}


        # Generate layout predictions
        predicted_ids = self.model.generate(
                    **encoding,
                    num_beams=1,
                    max_length=512
        )

        prediction_text = self.processor.decode(predicted_ids[0][1:-1])

        prediction = {}
        prediction['layout'] = prediction_text
        return prediction
