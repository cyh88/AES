import codecs
import pickle as pk
from sklearn import preprocessing
import re
import nltk
import numpy as np
from sklearn import preprocessing
import logging

logger = logging.getLogger(__name__)
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'

class DatasetProcess:
    def __init__(self, args):
        self.prompt_id = args.prompt_id
        self.vocab_size = args.vocab_size
        self.maxlen = args.maxlen
        self.tokenize_text = True
        self.vocab_path = args.vocab_path
        self.asap_ranges = {
            0: (0, 60),
            1: (2, 12),
            2: (1, 6),
            3: (0, 3),
            4: (0, 3),
            5: (0, 4),
            6: (0, 4),
            7: (0, 30),
            8: (0, 60)
        }

    def convert_to_dataset_friendly_scores(self,scores_array, prompt_id_array):
        arg_type = type(prompt_id_array)
        assert arg_type in {int, np.ndarray}
        if arg_type is int:
            low, high = self.asap_ranges[prompt_id_array]
            scores_array = scores_array * (high - low) + low
            assert np.all(scores_array >= low) and np.all(scores_array <= high)
        else:
            assert scores_array.shape[0] == prompt_id_array.shape[0]
            dim = scores_array.shape[0]
            low = np.zeros(dim)
            high = np.zeros(dim)
            for ii in range(dim):
                low[ii], high[ii] = self.asap_ranges[prompt_id_array[ii]]
            scores_array = scores_array * (high - low) + low
        return scores_array


    # 获得分数的范围
    def get_score_range(self,prompt_id):
        return self.asap_ranges[prompt_id]

    # 获得原始数据
    def get_ref_dtype(self):
        return ref_scores_dtype

    # 将分数转换为 [0 1] 的边界以进行训练和评估（损失计算）
    def get_model_friendly_scores(self,scores_array, prompt_id_array):
        arg_type = type(prompt_id_array)
        assert arg_type in {int, np.ndarray}
        if arg_type is int:
            low, high = self.asap_ranges[prompt_id_array]
            scores_array = (scores_array - low) / (high - low)
        else:
            assert scores_array.shape[0] == prompt_id_array.shape[0]
            dim = scores_array.shape[0]
            low = np.zeros(dim)
            high = np.zeros(dim)
            for ii in range(dim):
                low[ii], high[ii] = self.asap_ranges[prompt_id_array[ii]]
            scores_array = (scores_array - low) / (high - low)
        assert np.all(scores_array >= 0) and np.all(scores_array <= 1)
        return scores_array

    # 载入词表
    def load_vocab(self, vocab_path):
        print('Loading vocabulary from: ' + vocab_path)
        with open(self.vocab_path, 'rb') as vocab_file:
            vocab = pk.load(vocab_file)
        return vocab

    # 是否数字
    def is_number(self, token):
        return bool(num_regex.match(token))

    # 分词 不然双引号会不同！
    def tokenize(self, string):
        tokens = nltk.word_tokenize(string)
        for index, token in enumerate(tokens):
            if token == '@' and (index + 1) < len(tokens):
                tokens[index + 1] = '@' + re.sub('[0-9]+.*', '', tokens[index + 1])
                tokens.pop(index)
        return tokens

    # 获得第一组数据
    def get_data(self, args, vocab_size, tokenize_text=True, to_lower=True, vocab_path=None):
        train_path, dev_path, test_path = args.train_path, args.dev_path, args.test_path
        # 载入词表
        vocab = self.load_vocab(vocab_path)
        if len(vocab) != vocab_size:
            logger.warning(
                'The vocabualry includes %i words which is different from given: %i' % (len(vocab), vocab_size))
        logger.info('  Vocab size: %i' % (len(vocab)))

        train_x, train_y, train_prompts, train_maxlen = self.read_dataset(train_path, vocab, tokenize_text)
        dev_x, dev_y, dev_prompts, dev_maxlen = self.read_dataset(dev_path, vocab, tokenize_text)
        test_x, test_y, test_prompts, test_maxlen = self.read_dataset(test_path, vocab, tokenize_text)

        # 读取题目

        overal_maxlen = max(train_maxlen, dev_maxlen, test_maxlen)

        return ((train_x, train_y, train_prompts), (dev_x, dev_y, dev_prompts),
                (test_x, test_y, test_prompts), vocab, len(vocab), 1)

    # 读取第一组数据
    def read_dataset(self, file_path, vocab, tokenize_text):
        print('Reading dataset from: ' + file_path)
        # 长度截断
        if self.maxlen > 0:
            logger.info('  Removing sequences with more than ' + str(self.maxlen) + ' words')
        data_x, data_y, prompt_ids, contents = [], [], [], []
        num_hit, unk_hit, total = 0., 0., 0.
        t = 0
        maxlen_x = -1
        with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
            next(input_file)
            for line in input_file:
                if line.strip() == '':  # 用于消除空行
                    continue
                tokens = line.strip().split('\t')
                essay_id = int(tokens[0])
                # kouti = hand_feature[int(tokens[0])]
                essay_set = int(tokens[1])  # 文章所属集合/类别
                content = tokens[2].strip()
                # 第6列是得分
                score = float(tokens[6])
                if essay_set == self.prompt_id or self.prompt_id <= 0:
                    content = self.tokenize(content)
                    # print(content)
                    if (score > self.asap_ranges[essay_set][0] + 1) and len(content) < 120 \
                            and content[-1] not in ['.', '!', '?', "''", '``', '..'] \
                            or (len(content) < 80 and score > self.asap_ranges[essay_set][1] - 1):
                        print(content[-1], essay_id)
                        continue
                    if self.maxlen > 0 and len(content) > self.maxlen:
                        continue
                    indices = []
                    pos_indices = []
                    t += 1
                    for word in content:
                        if self.is_number(word):
                            indices.append(vocab['<num>'])
                            num_hit += 1
                        elif word in vocab:
                            indices.append(vocab[word])
                        else:
                            indices.append(vocab['<unk>'])
                            unk_hit += 1
                        total += 1
                    # 把题目给加进去
                    indices.extend(self.get_prompt_id(essay_set,vocab)[0])
                    # 把登录的索引添加到data_x里
                    data_x.append(indices)
                    # pos_x.append(pos_indices)
                    # 把文章的分数添加到data_y里
                    data_y.append(score)
                    # 把文章的类别添加到prompt_ids里
                    prompt_ids.append(essay_set)
                    if maxlen_x < len(indices):
                        maxlen_x = len(indices)
        logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100 * num_hit / total, 100 * unk_hit / total))

        return data_x, data_y, prompt_ids, maxlen_x

    # 获得第二组数据
    def get_pre_data(self, args):
        train_path, dev_path, test_path = args.train_path, args.dev_path, args.test_path
        # 载入词表
        vocab = self.load_vocab(self.vocab_path)
        if len(vocab) != self.vocab_size:
            print(
                'The vocabualry includes %i words which is different from given: %i' % (len(vocab), self.vocab_size))

        print('  Vocab size: %i' % (len(vocab)))
        # 读取数据
        train_pre_x, train_pre_y, train_pre_prompts, train_maxlen,train_x_topic = self.read_pre_data(train_path, vocab)
        dev_pre_x, dev_pre_y, dev_pre_prompts, dev_maxlen ,dev_x_topic= self.read_pre_data(dev_path, vocab)
        test_pre_x, test_pre_y, test_pre_prompts, test_maxlen,text_x_topic = self.read_pre_data(test_path, vocab)
        # 没返回
        overal_maxlen = max(train_maxlen, dev_maxlen, test_maxlen)
        return (train_pre_x, train_pre_y, train_pre_prompts,train_x_topic), (dev_pre_x, dev_pre_y, dev_pre_prompts,dev_x_topic), (
            test_pre_x, test_pre_y, test_pre_prompts,text_x_topic), vocab, len(vocab)

    # 读取第二组数据
    def read_pre_data(self, file_path, vocab):
        print('Reading pretreatment dataset from: ' + file_path)
        # 长度截断
        if self.maxlen:
            print('  Removing sequences with more than ' + str(self.maxlen) + ' words')
        data_x, data_y, prompt_ids, text,data_topic = [], [], [], [],[]
        num_hit, unk_hit, total = 0., 0., 0.
        t = 0
        maxlen_x = -1
        with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
            next(input_file)
            for line in input_file:
                if line.strip() == '':  # 用于消除空行
                    continue
                tokens = line.strip().split('\t')
                essay_id = int(tokens[0])
                essay_set = int(tokens[1])  # 文章所属集合/类别
                content = tokens[2].strip()
                text = content
                # # 文本添加进data_x
                # data_x.append(content)
                # 第6列是得分
                score = float(tokens[6])
                # data_y.append(score)
                if essay_set == self.prompt_id or self.prompt_id <= 0:
                    content = content.lower()
                    # 分词 不然双引号会不同！
                    content = self.tokenize(content)
                    # print(content)
                    if (score > self.asap_ranges[essay_set][0] + 1) and len(content) < 120 \
                            and content[-1] not in ['.', '!', '?', "''", '``', '..'] \
                            or (len(content) < 80 and score > self.asap_ranges[essay_set][1] - 1):
                        print(content[-1], essay_id)
                        continue
                    if self.maxlen > 0 and len(content) > self.maxlen:
                        continue
                    indices = []
                    t += 1
                    for word in content:
                        if self.is_number(word):
                            indices.append(vocab['<num>'])
                            num_hit += 1
                        elif word in vocab:
                            indices.append(vocab[word])
                        else:
                            indices.append(vocab['<unk>'])
                            unk_hit += 1
                        total += 1
                    data_x.append(text)
                    data_topic.append(text)
                    data_topic.append(self.get_prompt_id(essay_set,vocab)[1])
                    # 把文章的分数添加到data_y里
                    data_y.append(score)
                    # 把文章的类别添加到prompt_ids里
                    prompt_ids.append(essay_set)
                    if maxlen_x < len(indices):
                        maxlen_x = len(indices)
        print('Pre_utils:  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (
            100 * num_hit / total, 100 * unk_hit / total))

        return data_x, data_y, prompt_ids, maxlen_x, data_topic

    def get_prompt_id(self,essay_id,vocab):
        essay ,topic = [] ,[]
        with codecs.open('data/prompt.tsv', mode='r', encoding='UTF8') as input_file:
            next(input_file)
            for line in input_file:
                if len(line.strip()) < 10:
                    continue
                indices = []
                content = line[3:].strip()
                content = content.lower()
                text = content
                # 分词 不然双引号会不同！
                content = self.tokenize(content)
                for word in content:
                    if self.is_number(word):
                        indices.append(vocab['<num>'])
                    elif word in vocab:
                        indices.append(vocab[word])
                    else:
                        indices.append(vocab['<unk>'])
                essay.append(indices)
                topic.append(text)
        return (essay[essay_id-1] , topic[essay_id-1])

