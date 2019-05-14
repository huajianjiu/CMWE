# -*- coding: utf-8 -*-

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pickle
from keras.models import Model, Sequential
from keras.layers import Input, MaxPooling2D, Conv2D, Conv1D, MaxPool1D, concatenate, TimeDistributed, \
    Dense, Reshape
from keras.utils.np_utils import to_categorical
from ShapeEmbedding import train_model, test_model
import gc

MAX_SENTENCE_LENGTH = 128


def sentence2image(sentence):
    sentence = sentence[:MAX_SENTENCE_LENGTH]
    output = np.zeros((MAX_SENTENCE_LENGTH, 32, 32, 1))
    for i, character in enumerate(sentence):
        im = Image.new("1", (32, 32), 0)
        dr = ImageDraw.Draw(im)
        font = ImageFont.truetype("/usr/share/fonts/truetype/takao-mincho/TakaoPMincho.ttf", 32)
        dr.text((0, 0), character, font=font, fill=1)
        # im.save("1.jpg")
        img = np.array(im, dtype="int32")
        img = np.reshape(img, (32, 32, 1))
        output[i] = img
    return output


def x2image(texts):
    output = np.zeros((len(texts), MAX_SENTENCE_LENGTH, 32, 32))
    for i, text in enumerate(texts):
        output[i] = sentence2image(text)
    return output


def shimada_character_encoder():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(2, 2), input_shape=(32, 32, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Reshape((64,)))
    model.summary()
    return model


def shimada_sentence_encoder():
    character_encoder = shimada_character_encoder()
    model = Sequential()
    model.add(TimeDistributed(character_encoder, input_shape=(MAX_SENTENCE_LENGTH, 32, 32, 1)))
    model.add(Conv1D(filters=512, kernel_size=3, activation='relu'))
    model.add(MaxPool1D())
    model.add(Conv1D(filters=512, kernel_size=3, activation='relu'))
    model.add(MaxPool1D())
    model.add(Conv1D(filters=512, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=512, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=512, kernel_size=3, activation='relu'))
    model.add(Reshape((-1,)))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    return model


def preparse_data(dataset):
    data_size = (len(dataset['positive']+dataset['negative']))
    print(data_size)
    x_data = np.zeros((data_size, MAX_SENTENCE_LENGTH, 32, 32, 1), dtype=np.float32)
    y_data = np.zeros((data_size, 2), dtype=np.int32)
    for i, document in enumerate(dataset['positive']):
        x_data[i] = sentence2image(document)
        y_data[i][1] = 1
    for j, document in enumerate(dataset['negative']):
        x_data[j + len(dataset['positive'])] = sentence2image(document)
        y_data[j + len(dataset['positive'])][0] = 1
    return x_data, y_data


def shima_unk_experiment_train():
    from unkExperiment import load_data
    train_set, _, validation_set, _, _, _ \
        = load_data()
    x_train, y_train = preparse_data(train_set)
    del train_set
    x_val, y_val = preparse_data(validation_set)
    del validation_set

    model = shimada_sentence_encoder()
    return train_model(model, x_train, y_train, x_val, y_val, 'shimada')

def shima_unk_experiment_test():
    from unkExperiment import load_data
    _, _, _, test_normal_set, test_unk_w_set, test_unk_c_set \
        = load_data()
    x_normal, y_normal = preparse_data(test_normal_set)
    del test_normal_set
    x_unkw, y_unkw = preparse_data(test_unk_w_set)
    del test_unk_w_set
    x_unkc, y_unkc = preparse_data(test_unk_c_set)
    del test_unk_c_set
    model = shimada_sentence_encoder()
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['categorical_crossentropy', "acc"], )
    test_model(model, 'shimada', x_normal, y_normal)
    test_model(model, 'shimada', x_unkw, y_unkw)
    test_model(model, 'shimada', x_unkc, y_unkc)

if __name__=='__main__':
    # model = shimada_sentence_encoder()
    # shima_unk_experiment_train()
    shima_unk_experiment_test()