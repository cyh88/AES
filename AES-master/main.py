#!/usr/bin/env python
import logging
import pickle
import random
import numpy as np
from time import time
import os
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.losses import huber,mse,cosine_similarity,poisson,mean_squared_logarithmic_error,mean_absolute_error
from Pre_utils.bert_setting import BertPreSetting
from Pre_utils.data_process import DatasetProcess
from Pre_utils.my_utils import store_data, get_store_data, get_bert
from config import Config
from models import ModelsConfigs, Models
from Evaluate.evaluator import Evaluator

# 查看可用设备
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一, 三块GPU
# 最好的验证性能
best_result_fold = []
# 平均性能
mean_fold = []

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
#######################################################################################################################
## Parse arguments
config = Config()
args = config.get_parser()

out_dir = args.out_dir_path

# prompts = [1,2,3]
# datas = [0,1]

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    # tf gpu fix seed, please `pip install tensorflow-determinism` first
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    #os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"]="1" #为了不让加bert之后 有上一句而报错
    os.environ["TF_CUDNN_DETERMINISTIC"]="1"
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
setup_seed(args.seed)


def set_tf_device(device):
    if device == 'cpu':
        print("Training on CPU...")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device == 'gpu':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print("Training on GPU...")
        for gpu in tf.config.experimental.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

#set_tf_device('gpu')



# 自定义损失函数
def my_loss(y_true,y_pred):
    alpha = 0.2
    return (1 - alpha) * mse(y_true, y_pred) + alpha * cosine_similarity(y_true, y_pred)


#bert_model1, bert_model2 = get_bert()

prompts = [1,2,3,4,5,6,7,8]
datas = [0,1,2,3,4]
# 第i折的训练
for i in datas:
    # 最好的数据集性能
    best_result_data = []
    # 第j个数据
    for j in prompts:
        logger.info('################################################# ' + 'this is training: fold ' + str(
            i) + ' and prompt' + str(j) + ' #################################################')

        args.prompt_id = j
        # 指定数据集的路径
        args.train_path = 'data/fold_' + str(i) + '/train.tsv'
        args.dev_path = 'data/fold_' + str(i) + '/dev.tsv'
        args.test_path = 'data/fold_' + str(i) + '/test.tsv'
        # 3090专用
        args.batch_size = 16
        # 设定文本长度
        maxlen_array = [950, 1085, 313, 304, 371, 378, 615, 1324]
        overal_maxlen = maxlen_array[args.prompt_id - 1]
        # 获得数据
        get_Data = DatasetProcess(args)

        #存储数据
        #if  not os.path.exists('store/fold_' + str(i) + '/prompt_' + str(j) + '_train_x_ids.pkl'):
            # 准备第一组数据#(1069,356,357)
        (train_x, train_y, train_pmt), \
        (dev_x, dev_y, dev_pmt), \
        (test_x, test_y, test_pmt), \
        vocab, vocab_size, num_outputs \
            = get_Data.get_data(args, args.vocab_size, tokenize_text=True, vocab_path=args.vocab_path)

        # 对序列长度进行padding,转为ndarray格式
        train_x = sequence.pad_sequences(train_x,maxlen=overal_maxlen)
        dev_x = sequence.pad_sequences(dev_x,maxlen=overal_maxlen)
        test_x = sequence.pad_sequences(test_x,maxlen=overal_maxlen)

        train_y = np.array(train_y, dtype=K.floatx())
        dev_y = np.array(dev_y, dtype=K.floatx())
        test_y = np.array(test_y, dtype=K.floatx())

        train_pmt = np.array(train_pmt, dtype='int32')
        dev_pmt = np.array(dev_pmt, dtype='int32')
        test_pmt = np.array(test_pmt, dtype='int32')

        #准备第二组数据
        (train_pre_x, train_pre_y, train_pre_prompts,train_x_topic), (dev_pre_x, dev_pre_y, dev_pre_prompts,dev_x_topic), (
            test_pre_x, test_pre_y, test_pre_prompts,text_x_topic), vocab, vocab_size = get_Data.get_pre_data(args)

        # 获得第二组数据
        bert_inputs = BertPreSetting(args, overal_maxlen)
        inputs_train_ids, inputs_train_mask, inputs_train_tokentype = bert_inputs.get_inputs(args, train_pre_x)
        inputs_dev_ids, inputs_dev_mask, inputs_dev_tokentype = bert_inputs.get_inputs(args, dev_pre_x)
        inputs_test_ids, inputs_test_mask, inputs_test_tokentype = bert_inputs.get_inputs(args, test_pre_x)

        #x = bert_inputs.get_emb(args,train_x_topic)

        # train_x,_,_ = bert_inputs.get_inputs(args, train_x_topic)
        # dev_x, _,_ = bert_inputs.get_inputs(args, dev_x_topic)
        # test_x, _,_ = bert_inputs.get_inputs(args, text_x_topic)


        # 我们需要原始规模的开发和测试集进行评估
        dev_y_org = dev_y.astype(get_Data.get_ref_dtype())
        test_y_org = test_y.astype(get_Data.get_ref_dtype())

        # 将分数转换为 [0 1] 的边界以进行训练和评估（损失计算）
        train_y = get_Data.get_model_friendly_scores(train_y, train_pmt)
        dev_y = get_Data.get_model_friendly_scores(dev_y, dev_pmt)
        test_y = get_Data.get_model_friendly_scores(test_y, test_pmt)
#             # 存储数据
#             store_data(i,j,train_x,dev_x,test_x,train_y,dev_y,test_y,dev_y_org,test_y_org,inputs_train_ids,
#                        inputs_dev_ids,inputs_test_ids,inputs_train_mask,inputs_dev_mask,inputs_test_mask,
#                        inputs_train_tokentype,inputs_dev_tokentype,inputs_test_tokentype)
#         else:
#             #读取数据
#             train_x, dev_x, test_x, train_y, dev_y, test_y, dev_y_org, test_y_org, inputs_train_ids,\
#             inputs_dev_ids,inputs_test_ids,inputs_train_mask,inputs_dev_mask,inputs_test_mask,\
#                        inputs_train_tokentype,inputs_dev_tokentype,inputs_test_tokentype,vocab = get_store_data(i,j)

        # 建立模型
        creat_model = Models(args, train_y.mean(axis=0), overal_maxlen)
        models_config = ModelsConfigs()
        # optimizer = models_config.get_optimizer(args)
        loss, metric = models_config.get_loss_metric(args)

        model = creat_model.get_model(args, overal_maxlen, vocab)
        # model.compile(optimizer='rmsprop', loss=loss, metrics=metric)
        #model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=metric)
        model.compile(optimizer='Nadam', loss=tf.keras.losses.MeanSquaredError(), metrics=metric)

        # 评价指标
        evl = Evaluator(get_Data, args, out_dir, dev_x, inputs_dev_ids, inputs_dev_mask, inputs_dev_tokentype, test_x,
                        inputs_test_ids, inputs_test_mask, inputs_test_tokentype, dev_y, test_y, dev_y_org, test_y_org)

        # 训练
        total_train_time = 0
        total_eval_time = 0
        t1 = time()

        for epoch in range(args.epochs):
            # Training
            logger.info('Prompt_id: %d, Epoch %d,' % (j, epoch))
            logger.info('Training:')
            t0 = time()
            train_history = model.fit([train_x, inputs_train_ids, inputs_train_mask, inputs_train_tokentype], train_y,
                                      batch_size=args.batch_size, epochs=1, verbose=1)
            tr_time = time() - t0
            total_train_time += tr_time

            # Evaluate
            t0 = time()
            evl.evaluate(model, epoch)
            evl_time = time() - t0
            total_eval_time += evl_time
            total_time = time() - t1

            # Print information
            logger.info('train: %is, evaluation: %is, total_time: %is' % (tr_time, evl_time, total_time))
            train_loss = train_history.history['loss'][0]
            train_metric = train_history.history[metric][0]
            logger.info('[Train] loss: %.4f, metric: %.4f' % (train_loss, train_metric))
            evl.print_info()

        # 结果总结
        logger.info('Training:   %i seconds in total' % total_train_time)
        logger.info('Evaluation: %i seconds in total' % total_eval_time)

        evl.print_final_info()

        best_statistics = evl.get_best_statistics()
        best_result_data.append(best_statistics)
        logger.removeHandler(logging.StreamHandler)

    best_result_fold.append(best_result_data)

mean_fold.append(np.mean(best_result_fold, axis=0))

import time

print(np.mean(best_result_fold, axis=0))
with open("实验数据.txt", "a+", encoding="utf-8") as f:
    f.seek(0)
    f.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + '：' + args.model_type + '，' + args.explain + '\n')
    f.write(str(best_result_fold))
    f.write("\n")
    f.write('均值：')
    f.write(str(mean_fold))
    f.write("\n")
    f.write("平均QWK：" + str(np.mean(mean_fold)))
    f.write("\n")
    f.write("\n")

#os.system('/usr/upload.sh')