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

import frozendict

COLORS = {
    "TEXT": (66, 166, 246),
    "SVG": (0, 225, 179),
    "PHOTO": (55, 22, 18),
    "SHAPESVG": (100, 221, 57),
    "FrameItem": (211, 20, 83),
    "BASICSVG": (13, 71, 162),
    "STICKER": (216, 120, 227),
    "SVG_PRIVATE": (187, 104, 201),
    "SVGIMAGEFRAME": (79, 196, 248),
    "Chart": (226, 81, 232),
    "GRID": (256, 139, 101)
}

LABEL_NAMES = ("TEXT",
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