from vis.utils import utils
from keras import activations
from vis.visualization import visualize_activation
from attention import AttentionWithContext

from matplotlib import pyplot as plt

from ShapeEmbedding import build_fasttext, build_hatt, build_sentence_rnn, build_word_feature_char, \
    build_word_feature_shape, text_to_char_index, get_vocab, _make_kana_convertor, train_model, test_model

# MAX_SENTENCE_LENGTH = 739  # large number as 739 makes cudnn die
MAX_SENTENCE_LENGTH = 500
MAX_WORD_LENGTH = 4
COMP_WIDTH = 3
CHAR_EMB_DIM = 15
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
BATCH_SIZE = 100
WORD_DIM = 600
MAX_RUN = 1
VERBOSE = 0
EPOCHS = 50

full_vocab, real_vocab_number, chara_bukken_revised, additional_translate, hira_punc_number_latin = get_vocab()
n_hira_punc_number_latin = len(hira_punc_number_latin) + 2
model_name = "Radical-CNN-RNN HARC"
print("======MODEL: ", model_name, "======")
model = build_sentence_rnn(real_vocab_number=real_vocab_number, classes=2,
                           char_shape=True, word=False, char=False,
                           cnn_encoder=True, highway='relu', nohighway="linear",
                           attention=True, shape_filter=True, char_filter=True)
model.load_weights("unk_exp/checkpoints/" + model_name + "_bestloss.hdf5")

layer_dict = dict([(layer.name, layer) for layer in model.layers])

print(layer_dict)

# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'dense_1')

# Swap softmax with linear
# model.layers[layer_idx].activation = activations.linear
# model = utils.apply_modifications(model)

plt.rcParams['figure.figsize'] = (18, 6)

# 20 is the imagenet category for 'ouzel'
img = visualize_activation(model, layer_idx)
plt.show(img)
