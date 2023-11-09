import math

import numpy as np
from keras.losses import cosine_similarity
from sklearn.metrics import euclidean_distances
from tensorflow.python.keras.preprocessing import sequence
from transformers import BertTokenizer
import tensorflow as tf
from Pre_utils.bert_setting import BertPreSetting
from sklearn.metrics.pairwise import cosine_similarity  # 余弦距离
from sklearn.metrics.pairwise import euclidean_distances
path = '../Pre-training/BERT_base'
tokenizer = BertTokenizer.from_pretrained(path)


# sentence1= "More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you."
#
# encoded_inputs1= tokenizer(sentence1, return_tensors='tf', padding=True, truncation=True,
#                                         max_length=300)
# encoded_inputs1.data['input_ids']

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[1,6], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[1,6], name='a')
# Cosine similarity
similarity = tf.reduce_sum(a[:, tf.newaxis] * b, axis=-1)
# Only necessary if vectors are not normalized
similarity /= tf.norm(a[:, tf.newaxis], axis=-1) * tf.norm(b, axis=-1)
# If you prefer the distance measure
distance = 1 - similarity
print(distance)
print(cosine_similarity(a,b))
print(euclidean_distances(a,b))

# m = math.sqrt(sum(pow(a - b, 2) for a, b in zip(a, b)))
# print(1 / (1 + m))

# # a.index_select(0, torch.tensor([a, b]))
# a_vecs = tf.unstack(encoded_inputs1.data['input_ids'], axis=1)
# del a_vecs[0]
# del a_vecs[-1]
# a_new = tf.stack(a_vecs, 1)
# a_new =sequence.pad_sequences(a_new, maxlen=35500)
# a_new = tf.reshape(a_new,[355,100])
# print(a_new)

#print(n_output[1:2])
# inputs_ids1 = encoded_inputs1.get('input_ids')
#
# print(inputs_ids1)


# from config import Config
# config = Config()
# args = config.get_parser()
# bert_inputs = BertPreSetting(args, 300)
# inputs_train_ids, inputs_train_mask, inputs_train_tokentype =    bert_inputs.get_inputs(args, "wo are man")
# print(inputs_train_ids)
# # from bert_serving.client import BertClient
# # bc = BertClient()
# # doc_vecs = bc.encode(['First do it', 'then do it right', 'then do it better'])
# # print(doc_vecs)
#
# config_path = '../bertKeras/bert_config.json'
# checkpoint_path = '../bertKeras/bert_model.ckpt'
# dict_path = '../bertKeras/vocab.txt'



def similar_count(vec1, vec2, model="cos"):
    '''
    计算距离
    :param vec1: 句向量1
    :param vec2: 句向量2
    :param model: 用欧氏距离还是余弦距离
    :return: 返回的是两个向量的距离得分
    '''
    if model == "eu":
        return euclidean_distances([vec1, vec2])[0][1]
    if model == "cos":
        return cosine_similarity([vec1, vec2])[0][1]

#
# def main():
#     # 根据配置文件，加载模型
#     model = build_transformer_model(
#         config_path=config_path,
#         checkpoint_path=checkpoint_path,
#         with_pool=True,
#         return_keras_model=True,
#         model="bert"
#     )
#     # 建立分词器
#     tokenizer = Tokenizer(dict_path)
#
#     # 要计算相似句子的标准句1
#     stand_sent1 = "hahah"
#     # 要计算相似句子的标准句2
#     stand_sent2 = "wee"
#
#     # 得到两个句子的 token 和 segment
#     token_ids1, segment_ids1 = tokenizer.encode(stand_sent1, maxlen=128)
#     token_ids2, segment_ids2 = tokenizer.encode(stand_sent2, maxlen=128)
#
#     # 通过模型得到两个句子的向量
#     sentence_vec1 = model.predict([np.array([token_ids1]), np.array([segment_ids1])])[0]
#     sentence_vec2 = model.predict([np.array([token_ids2]), np.array([segment_ids2])])[0]
#
#     # 计算出得分
#     score = similar_count(sentence_vec1, sentence_vec2)
#
#     print(score)


# if __name__ == '__main__':
#     main()
