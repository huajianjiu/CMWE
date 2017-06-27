# -*- coding: utf-8 -*-
import os
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
import numpy as np
import pickle


count = 0
char_vocab = []
shape_vocab = []
char_shape = {}
for line in open("joyo2010.txt").readlines():
    if line[0] == "#":
        continue
    text = line[0]
    im = Image.new("1", (28,28), 0)
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype("/usr/share/fonts/truetype/takao-mincho/TakaoPMincho.ttf", 28)
    dr.text((0, 0), text, font=font, fill=1)
    im.save("1.jpg")
    img = np.array(im, dtype="int32")
    char_vocab.append(text)
    shape_vocab.append(img)
    char_shape[text] = img
    count += 0
shape_vocab_data = {
    "chars": char_vocab,
    "shapes": shape_vocab,
    "char_shape": char_shape
}
with open("shape_vocab.pickle", "wb") as f:
    pickle._dump(shape_vocab_data, f)