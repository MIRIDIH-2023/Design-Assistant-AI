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

"""Information about the RICO dataset.

See http://interactionmining.org/rico for more details.
"""

import frozendict

COLORS = {
    "text_0_1": (66, 166, 246),
    "text_0_2": (0, 225, 179),
    "text_0_3": (55, 22, 18),
    "text_0_3_over": (145, 0, 250),
    "text_1_1": (100, 221, 57),
    "text_1_2": (26, 220, 221),
    "text_1_3": (211, 20, 83),
    "text_1_3_over": (13, 71, 162),
    "text_2_1": (216, 120, 227),
    "text_2_2": (187, 104, 201),
    "text_2_3": (0, 256, 205),
    "text_2_3_over": (79, 196, 248),
    "text_3_1": (256, 206, 211),
    "text_3_2": (175, 214, 130),
    "text_3_3": (74, 20, 141),
    "text_3_3_over": (226, 81, 232),
    "image_with_text": (102, 152, 216),
    "image_partially_with_text": (24, 148, 22),
    "image_without_text": (63, 134, 221),
    "background": (205, 102, 154),
    "Chart": (241, 98, 147),
    "GRID": (103, 58, 184)
}

LABEL_NAMES = (
    "text_0_1",
    "text_0_2",
    "text_0_3",
    "text_0_3_over",
    "text_1_1",
    "text_1_2",
    "text_1_3",
    "text_1_3_over",
    "text_2_1",
    "text_2_2",
    "text_2_3",
    "text_2_3_over",
    "text_3_1",
    "text_3_2",
    "text_3_3",
    "text_3_3_over",
    "image_with_text",
    "image_partially_with_text",
    "image_without_text",
    "background",
    "Chart",
    "GRID"
)

TAG_NAMES = ("TEXT",
                "SVG",
                "PHOTO",
                "SHAPESVG",
                "FrameItem",
                "BASICSVG",
                "STICKER",
                "SVG_PRIVATE",
                "SVGIMAGEFRAME",
                "Chart",
                "GRID")

ID_TO_LABEL = frozendict.frozendict(
    {i: v for (i, v) in enumerate(LABEL_NAMES)})

NUMBER_LABELS = len(ID_TO_LABEL)

LABEL_TO_ID_ = frozendict.frozendict(
    {l: i for i, l in ID_TO_LABEL.items()})