# import tensorflow.keras.optimizers as opt
import numpy as np
import tensorflow.keras.optimizers as opt
import tensorflow as tf
from my_layers import GlobalMaxPooling1D, GlobalAveragePooling1D, MultiHeadAttention, Attention, GateModule

'''
    篇章级特征：句法结构信息，构建一个图，通过图神经网络获得向量。来拼
    句法级特征。
'''
'''
freeze就是我初始构思和实现的结构
freeze那部分也冻结了  就是开始在试验模型冻结的部分了
然后看了一下 效果还可以 就开始跑消融实验
因为代码要修改的比较多 所以新建了一个if去做消融实验
最后论文里实验数据的结果 都是在Ablation_experiment里完成的
'''


class ModelsConfigs:
    # 获得优化器
    def get_optimizer(self, args):
        clipvalue = 0
        clipnorm = 10
        if args.algorithm == 'rmsprop':
            optimizer = opt.RMSprop(learning_rate=args.learning_rate, rho=0.9, epsilon=1e-06, clipnorm=clipnorm,
                                    clipvalue=clipvalue)

        return optimizer

    # 获得损失函数和评估方式
    def get_loss_metric(self, args):
        if args.loss == 'mse':
            # 均方误差
            loss = 'mean_squared_error'
            metric = 'mean_absolute_error'
        else:
            loss = 'mean_absolute_error'
            metric = 'mean_squared_error'

        return loss, metric


class Models:
    def __init__(self, args, initial_mean_value, overal_maxlen):
        if initial_mean_value.ndim == 0:
            self.initial_mean_value = np.expand_dims(initial_mean_value, axis=0)
        self.num_outpus = len(self.initial_mean_value)

    def get_model(self, args, overal_maxlen, vocab, essay_len):
        if args.model_type == 'Only bert_input':
            '''
                bert只作为输入和输出，中间什么都没有
            '''
            from tensorflow.keras import Model
            from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, \
                Concatenate, Convolution1D, Dropout, MaxPool1D \
                , LSTM, LeakyReLU, concatenate, Flatten

            from transformers import BertTokenizer, TFBertModel
            path = 'Pre-training/BERT_base'
            bert_model = TFBertModel.from_pretrained(path)

            # 第一个输入
            input1 = Input(shape=(overal_maxlen,), dtype='int32')
            # emb_out1.shape == (None,600,300)
            emb_out1 = Embedding(args.vocab_size, args.emb_dim, name='emb')(input1)

            # 第二个输入
            input_ids = Input(shape=(overal_maxlen,), dtype='int32')
            input_mask = Input(shape=(overal_maxlen,), dtype='int32')
            input_tokentype = Input(shape=(overal_maxlen,), dtype='int32')
            dim_out2 = bert_model(
                {"input_ids": input_ids, "token_type_ids": input_tokentype, "attention_mask": input_mask})
            # ★★★★★ 后面一定要加.pooler_output
            # dim_out2.last_hidden_state.shape == (None,600,768)
            bert_output = dim_out2.last_hidden_state

            trm_out = MultiHeadAttention(3, 100)(emb_out1)
            # trm_out = add([trm_out, trm_out_tem])
            trm_out = Dense(200, activation='relu')(trm_out)

            max1 = GlobalMaxPooling1D()(trm_out)
            avg = GlobalAveragePooling1D()(trm_out)
            x = concatenate([avg, max1], axis=-1)
            x = Dropout(0.1)(x)
            x = Dense(100, activation='relu')(x)
            x = Dense(100, activation='relu')(x)
            x = Dropout(0.1)(x)

            score = Dense(self.num_outpus, activation='sigmoid')(x)

            model = Model(inputs=[input1, input_ids, input_mask, input_tokentype], outputs=score)
            model.summary()

            return model

        if args.model_type == 'bert_pool':
            '''
                取bert的pooler_output  不进行池化，直接和第一个输入池化后拼接的x1 进行拼接
            '''
            from tensorflow.keras import Model
            from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, \
                Concatenate, Convolution1D, Dropout, MaxPool1D \
                , LSTM, LeakyReLU, concatenate, Flatten

            from transformers import BertTokenizer, TFBertModel
            path = 'Pre-training/BERT_base'
            bert_model = TFBertModel.from_pretrained(path)

            # 第一个输入
            input1 = Input(shape=(overal_maxlen,), dtype='int32')
            # emb_out1.shape == (None,600,300)
            emb_out1 = Embedding(args.vocab_size, args.emb_dim, name='emb')(input1)

            # trm_out.shape == (None,600,300)
            trm_out = MultiHeadAttention(3, 100)(emb_out1)
            # trm_out.shape == (None,600,200)
            trm_out = Dense(200, activation='relu')(trm_out)

            # max1.shape == avg1.shape == (None,200)
            max1 = GlobalMaxPooling1D()(trm_out)
            avg1 = GlobalAveragePooling1D()(trm_out)
            # x1.shape == (None,400)
            x1 = concatenate([avg1, max1], axis=-1)

            # 第二个输入
            # shape == (None,600)
            input_ids = Input(shape=(overal_maxlen,), dtype='int32')
            input_mask = Input(shape=(overal_maxlen,), dtype='int32')
            input_tokentype = Input(shape=(overal_maxlen,), dtype='int32')
            dim_out2 = bert_model(
                {"input_ids": input_ids, "token_type_ids": input_tokentype, "attention_mask": input_mask})
            # bert_last_hidden_output = dim_out2.last_hidden_state
            bert_pooler_output = dim_out2.pooler_output

            x2 = bert_pooler_output

            # bert_out = Dense(200,activation='relu')(bert_output)

            # max2.shape == avg2.shape == (None,768)
            # max2 = GlobalMaxPooling1D()(bert_out)
            # avg2 = GlobalAveragePooling1D()(bert_out)
            # x2.shape == 968 (★：为什么不是768+768 = 1536 ?)
            # x2 = concatenate([avg2,max2],axis=-1)

            x = concatenate([x1, x2], axis=-1)
            x = Dropout(0.1)(x)
            x = Dense(100, activation='relu')(x)
            x = Dense(100, activation='relu')(x)
            x = Dropout(0.1)(x)

            score = Dense(self.num_outpus, activation='sigmoid')(x)

            model = Model(inputs=[input1, input_ids, input_mask, input_tokentype], outputs=score)
            model.summary()

            return model

        if args.model_type == 'bert_hiddenState':
            '''
                取bert的last_hidden_state,和经过多头注意力后的第一个数据进行拼接。然后进行双池化，输出
            '''
            from tensorflow.keras import Model
            from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, \
                Concatenate, Convolution1D, Dropout, MaxPool1D \
                , LSTM, LeakyReLU, concatenate, Flatten

            from transformers import BertTokenizer, TFBertModel
            path = 'Pre-training/BERT_base'
            bert_model = TFBertModel.from_pretrained(path)

            # 第一个输入
            input1 = Input(shape=(overal_maxlen,), dtype='int32')
            # emb_out1.shape == (None,600,300)
            emb_out1 = Embedding(args.vocab_size, args.emb_dim, name='emb')(input1)

            # trm_out.shape == (None,600,300)
            x1 = MultiHeadAttention(3, 100)(emb_out1)

            # 第二个输入
            # shape == (None,600)
            input_ids = Input(shape=(overal_maxlen,), dtype='int32')
            input_mask = Input(shape=(overal_maxlen,), dtype='int32')
            input_tokentype = Input(shape=(overal_maxlen,), dtype='int32')
            dim_out2 = bert_model(
                {"input_ids": input_ids, "token_type_ids": input_tokentype, "attention_mask": input_mask})
            # bert_last_hidden_output = dim_out2.last_hidden_state
            bert_pooler_output = dim_out2.last_hidden_state

            x2 = bert_pooler_output

            # 两个输入进行拼接
            out = concatenate([x1, x2], axis=-1)
            # 双池化
            max = GlobalMaxPooling1D()(out)
            avg = GlobalAveragePooling1D()(out)
            # 线性，输出
            x = concatenate([max, avg], axis=-1)
            x = Dropout(0.1)(x)
            x = Dense(100, activation='relu')(x)
            x = Dense(100, activation='relu')(x)
            x = Dropout(0.1)(x)

            score = Dense(self.num_outpus, activation='sigmoid')(x)

            model = Model(inputs=[input1, input_ids, input_mask, input_tokentype], outputs=score)
            model.summary()

            return model

        if args.model_type == 'freeze':
            '''
                同上。冻结bert参数
            '''
            from tensorflow.keras import Model
            from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, \
                Concatenate, Convolution1D, Dropout, MaxPool1D \
                , LSTM, LeakyReLU, concatenate, Flatten
            import transformers

            from transformers import BertTokenizer, TFBertModel
            path = 'Pre-training/BERT_base'
            bert_model = TFBertModel.from_pretrained(path)
            ### 冻结参数
            # 全部冻结参数
            # for k, v in bert_model._get_trainable_state().items():
            #     k.trainable = False
            # bert_model.summary()

            # 冻结embeddings参数
            # for layer in bert_model.layers[:]:
            #     if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
            #         layer.embeddings.trainable = False
            # bert_model.summary()
            # 冻结encoder部分参数
            # for layer in bert_model.layers[:]:
            #     if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
            #         for idx, layer in enumerate(layer.encoder.layer):
            #             # if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            #             if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11]:
            #                 layer.trainable = False
            # bert_model.summary()

            # 第一个输入
            input1 = Input(shape=(overal_maxlen,), dtype='int32')
            # emb_out1.shape == (None,600,300)
            emb_out1 = Embedding(args.vocab_size, args.emb_dim, name='emb')(input1)

            # trm_out.shape == (None,600,300)
            x1 = MultiHeadAttention(3, 100)(emb_out1)

            # 第二个输入
            # shape == (None,600)
            input_ids = Input(shape=(overal_maxlen,), dtype='int32')
            input_mask = Input(shape=(overal_maxlen,), dtype='int32')
            input_tokentype = Input(shape=(overal_maxlen,), dtype='int32')
            dim_out2 = bert_model(
                {"input_ids": input_ids, "token_type_ids": input_tokentype, "attention_mask": input_mask})
            # bert_last_hidden_output = dim_out2.last_hidden_state
            bert_pooler_output = dim_out2.last_hidden_state

            x2 = bert_pooler_output

            # 两个输入进行拼接
            out = concatenate([x1, x2], axis=-1)
            # 双池化
            max = GlobalMaxPooling1D()(out)
            avg = GlobalAveragePooling1D()(out)
            # 线性，输出
            x = concatenate([max, avg], axis=-1)
            x = Dropout(0.1)(x)
            x = Dense(100, activation='relu')(x)
            x = Dense(100, activation='relu')(x)
            x = Dropout(0.1)(x)

            score = Dense(self.num_outpus, activation='sigmoid')(x)

            model = Model(inputs=[input1, input_ids, input_mask, input_tokentype], outputs=score)
            model.summary()

            return model

        if args.model_type == 'only_input':
            '''
                同上。冻结bert参数
            '''
            from tensorflow.keras import Model
            from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, \
                Concatenate, Convolution1D, Dropout, MaxPool1D \
                , LSTM, LeakyReLU, concatenate, Flatten
            import transformers

            from transformers import BertTokenizer, TFBertModel
            path = 'Pre-training/BERT_base'
            bert_model = TFBertModel.from_pretrained(path)
            ### 冻结参数
            # 全部冻结参数
            for k, v in bert_model._get_trainable_state().items():
                k.trainable = False
            bert_model.summary()

            # 冻结embeddings参数
            # for layer in bert_model.layers[:]:
            #     if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
            #         layer.embeddings.trainable = False
            # bert_model.summary()
            # # 冻结encoder部分参数
            # for layer in bert_model.layers[:]:
            #     if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
            #         for idx, layer in enumerate(layer.encoder.layer):
            #             # if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            #             if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11]:
            #                 layer.trainable = False
            bert_model.summary()

            # 第一个输入
            input1 = Input(shape=(overal_maxlen,), dtype='int32')
            # emb_out1.shape == (None,600,300)
            # emb_out1 = Embedding(args.vocab_size, args.emb_dim, name='emb')(input1)
            # # args.emb_dim == 300
            # # 维度降低
            # emb_out1 = Embedding(args.vocab_size, 50, name='emb')(input1)

            # # trm_out.shape == (None,600,300)
            # # output_dim == emb_dim == heads * head_size == 3 *100 == 300
            # x1 = MultiHeadAttention(3, 100)(emb_out1)
            # x1 = MultiHeadAttention(2, 25)(emb_out1)

            # 第二个输入
            # shape == (None,600)
            input_ids = Input(shape=(overal_maxlen,), dtype='int32')
            input_mask = Input(shape=(overal_maxlen,), dtype='int32')
            input_tokentype = Input(shape=(overal_maxlen,), dtype='int32')
            dim_out2 = bert_model(
                {"input_ids": input_ids, "token_type_ids": input_tokentype, "attention_mask": input_mask})
            # bert_last_hidden_output = dim_out2.last_hidden_state
            bert_pooler_output = dim_out2.last_hidden_state

            x2 = bert_pooler_output

            # 不拼接了，只用随机初始化
            # # 两个输入进行拼接
            # out = concatenate([x1, x2], axis=-1)
            out = x2

            # 双池化
            max = GlobalMaxPooling1D()(out)
            avg = GlobalAveragePooling1D()(out)
            # 线性，输出
            x = concatenate([max, avg], axis=-1)
            x = Dropout(0.1)(x)
            x = Dense(100, activation='elu')(x)
            x = Dense(100, activation='elu')(x)
            x = Dropout(0.1)(x)

            score = Dense(self.num_outpus, activation='sigmoid')(x)
            theme = Dense(300, activation='sigmoid')(x)

            model = Model(inputs=[input1, input_ids, input_mask, input_tokentype], outputs=[score, theme])
            model.summary()

            return model

        if args.model_type == 'test':
            '''
                同上。冻结bert参数
            '''
            from tensorflow.keras import Model
            from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, \
                Concatenate, Convolution1D, Dropout, MaxPool1D \
                , LSTM, LeakyReLU, concatenate, Flatten
            import transformers

            from transformers import BertTokenizer, TFBertModel
            path = 'Pre-training/BERT_base'
            bert_model = TFBertModel.from_pretrained(path)
            ### 冻结参数
            # 全部冻结参数
            # for k, v in bert_model._get_trainable_state().items():
            #     k.trainable = False
            # bert_model.summary()

            #######   冻结11层
            # 冻结embeddings参数
            for layer in bert_model.layers[:]:
                if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
                    layer.embeddings.trainable = False
            # bert_model.summary()
            # 冻结encoder部分参数
            for layer in bert_model.layers[:]:
                if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
                    for idx, layer in enumerate(layer.encoder.layer):
                        # if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                        if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                            layer.trainable = False
            bert_model.summary()

            # 第一个输入
            input1 = Input(shape=(overal_maxlen,), dtype='int32')
            # emb_out1.shape == (None,600,300)
            emb_out1 = Embedding(args.vocab_size, args.emb_dim, name='emb')(input1)

            # trm_out.shape == (None,600,300)
            x1 = MultiHeadAttention(3, 100)(emb_out1)
            # one_merge = Dense(300, activation='elu', kernel_initializer='he_normal')(x1)
            # two_merge = Dense(200, activation='elu', kernel_initializer='he_normal')(one_merge)
            # three_merge = Dense(100, activation='elu', kernel_initializer='he_normal')(two_merge)
            # # concate
            # concat = concatenate([one_merge, two_merge, three_merge], axis=-1)
            # x1= concat

            '''
            one_merge=Dense(512,activation='elu',kernel_initializer='he_normal')(concat)
            two_merge=Dense(256,activation='elu',kernel_initializer='he_normal')(one_merge)
            three_merge=Dense(64,activation='elu',kernel_initializer='he_normal')(two_merge)
            #concate
            concat = concatenate([one_merge,two_merge,three_merge], axis=-1)
            '''

            # 第二个输入
            # shape == (None,600)
            input_ids = Input(shape=(overal_maxlen,), dtype='int32')
            input_mask = Input(shape=(overal_maxlen,), dtype='int32')
            input_tokentype = Input(shape=(overal_maxlen,), dtype='int32')
            dim_out2 = bert_model(
                {"input_ids": input_ids, "token_type_ids": input_tokentype, "attention_mask": input_mask})
            # bert_last_hidden_output = dim_out2.last_hidden_state
            bert_pooler_output = dim_out2.last_hidden_state
            # x2.shape == (None,600,768)
            x2 = bert_pooler_output

            # out.shape == (None,600,1368)
            x_feature = concatenate([x1, x2], axis=-1)
            # 把Bert的输出作为初始化门偏置
            matrix = Dense(1068, activation='sigmoid')(emb_out1)
            out = x_feature * matrix

            max = GlobalMaxPooling1D()(out)
            avg = GlobalAveragePooling1D()(out)

            x = concatenate([max, avg], axis=-1)

            # # 分别双池化，然后门控，然后全连接
            # max_x1 = GlobalMaxPooling1D()(x1)
            # avg_x1 = GlobalAveragePooling1D()(x1)
            # # x1.shape == (None,300+300) == (None,600)
            # x1 = concatenate([max_x1,avg_x1],axis=-1)
            #
            # max_x2 = GlobalMaxPooling1D()(x2)
            # avg_x2 = GlobalAveragePooling1D()(x2)
            # # x2.shape == (None,768+768) == (None,1536)
            # x2 = concatenate([max_x2, avg_x2], axis=-1)
            #
            # # x3:x1和x2两个双池化一拼(上下),x3.shape == (None,1536+600) == (None,2136)
            # x_feature = concatenate([x1,x2],axis=-1)
            #
            # matrix = Dense(2136,activation='sigmoid')(input1)
            #
            # x = x_feature * matrix

            #
            # # 两个输入进行拼接
            # out = concatenate([x1, x2], axis=-1)
            # # 双池化
            # max = GlobalMaxPooling1D()(out)
            # avg = GlobalAveragePooling1D()(out)
            # # 线性，输出
            # x = concatenate([max, avg], axis=-1)

            x = Dropout(0.1)(x)
            x = Dense(100, activation='swish')(x)
            x = Dense(100, activation='swish')(x)
            x = Dropout(0.1)(x)

            score = Dense(self.num_outpus, activation='sigmoid')(x)

            model = Model(inputs=[input1, input_ids, input_mask, input_tokentype], outputs=score)
            model.summary()

            return model

        if args.model_type == 'Ablation_experiment':
            '''
                消融实验
            '''
            from tensorflow.keras import Model
            from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, \
                Concatenate, Convolution1D, Dropout, MaxPool1D \
                , LSTM, LeakyReLU, concatenate, Flatten
            # import transformers
            #
            # from transformers import BertTokenizer, TFBertModel
            # path = './Pre-training/BERT_base'
            # bert_model = TFBertModel.from_pretrained(path)
            ### 冻结参数
            # 全部冻结参数
            # for k, v in bert_model._get_trainable_state().items():
            #     k.trainable = False
            # bert_model.summary()

            # ##冻结embeddings参数
            # for layer in bert_model.layers[:]:
            #     if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
            #         layer.embeddings.trainable = False
            # bert_model.summary()

            ######   冻结10层
            ##冻结encoder部分参数
            # for layer in bert_model.layers[:]:
            #     if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
            #         for idx, layer in enumerate(layer.encoder.layer):
            #             # if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            #             if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            #                 layer.trainable = False
            # bert_model.summary()

            # 第一个输入
            input1 = Input(shape=(overal_maxlen,), dtype='int32')
            # emb_out1.shape == (None,600,300)
            emb_out1 = Embedding(args.vocab_size, args.emb_dim, name='emb')(input1)

            # trm_out.shape == (None,600,300)
            x1 = MultiHeadAttention(3, 100)(emb_out1)

            # 第二个输入
            # shape == (None,600)
            input_ids = Input(shape=(overal_maxlen,), dtype='int32')
            input_mask = Input(shape=(overal_maxlen,), dtype='int32')
            input_tokentype = Input(shape=(overal_maxlen,), dtype='int32')
            dim_out2 = bert_model(
                {"input_ids": input_ids, "token_type_ids": input_tokentype, "attention_mask": input_mask})

            # # bert_last_hidden_output = dim_out2.last_hidden_state
            # bert_pooler_output = dim_out2.last_hidden_state
            # # x2.shape == (None,600,768)
            # x2 = bert_pooler_output

            # out.shape == (None,600,1368)
            # x_feature = concatenate([x1, x2], axis=-1)
            # 把Bert的输出作为初始化门偏置
            # matrix = Dense(1068, activation='sigmoid')(emb_out1)
            # out = x_feature * matrix

            out = x1
            max = GlobalMaxPooling1D()(out)
            avg = GlobalAveragePooling1D()(out)

            x = concatenate([max, avg], axis=-1)

            x = Dropout(0.1)(x)
            x = Dense(100, activation='swish')(x)
            x = Dense(100, activation='swish')(x)
            x = Dropout(0.1)(x)

            score = Dense(self.num_outpus, activation='sigmoid')(x)

            model = Model(inputs=[input1, input_ids, input_mask, input_tokentype], outputs=score)
            model.summary()

            return model
        if args.model_type == 'essay_experiment':
            '''
                论文模型实验
            '''
            from tensorflow.keras import Model
            from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, \
                Concatenate, Convolution1D, Dropout, MaxPool1D \
                , LSTM, LeakyReLU, concatenate, Flatten
            import transformers

            from transformers import BertTokenizer, TFBertModel
            path = './Pre-training/BERT_base'
            bert_model = TFBertModel.from_pretrained(path)

            # 第一个输入
            input1 = Input(shape=(overal_maxlen,), dtype='int32')
            # emb_out1.shape == (None,600,300)
            emb_out1 = Embedding(args.vocab_size, args.emb_dim, name='emb')(input1)
            # trm_out.shape == (None,600,300)
            # emb_out = LSTM(300, return_sequences=True)(emb_out1)
            # x1 = MultiHeadAttention(3, 100)(emb_out)
            # x1 = MultiHeadAttention(3, 100)(x1)

            # 第二个输入
            # shape == (None,600)
            input_ids = Input(shape=(overal_maxlen,), dtype='int32')
            input_mask = Input(shape=(overal_maxlen,), dtype='int32')
            input_tokentype = Input(shape=(overal_maxlen,), dtype='int32')
            dim_out2 = bert_model(
                {"input_ids": input_ids, "token_type_ids": input_tokentype, "attention_mask": input_mask})

            input_label_ids = Input(shape=(overal_maxlen,), dtype='int32')
            input_label_mask = Input(shape=(overal_maxlen,), dtype='int32')
            input_label_tokentype = Input(shape=(overal_maxlen,), dtype='int32')
            dim_lebal_out2 = bert_model(
                {"input_ids": input_label_ids, "token_type_ids": input_label_tokentype,
                 "attention_mask": input_label_mask})

            # bert_last_hidden_output = dim_out2.last_hidden_state
            bert_pooler_output = dim_out2.last_hidden_state
            # x2.shape == (None,600,768)
            x2 = bert_pooler_output
            x2 = MultiHeadAttention(3, 256)(x2)

            # out.shape == (None,600,1368)
            # x_feature = concatenate([x1, x2], axis=-1)
            # 把Bert的输出作为初始化门偏置
            matrix = Dense(768, activation='sigmoid')(emb_out1)
            out = x2 * matrix

            max = GlobalMaxPooling1D()(out)
            avg = GlobalAveragePooling1D()(out)

            x = concatenate([max, avg], axis=-1)
            x = Dropout(0.1)(x)
            x = Dense(100, activation='swish')(x)
            x = Dense(100, activation='swish')(x)
            x = Dropout(0.1)(x)

            score = Dense(self.num_outpus, activation='sigmoid')(x)

            model = Model(inputs=[input1, input_ids, input_mask, input_tokentype, input_label_ids, input_label_mask,
                                  input_label_tokentype], outputs=score)
            model.summary()

            return model
        if args.model_type == 'cyh':
            from my_layers import HSMMBottom, HSMMTower, Self_Attention, CrossAttention, Adapter
            from tensorflow.keras import Model
            from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, \
                Concatenate, Convolution1D, Dropout, MaxPool1D \
                , LSTM, LeakyReLU, concatenate, Flatten
            from transformers import BertTokenizer, TFBertModel
            import tensorflow.keras.backend as K
            from transformers.models.bert.modeling_tf_bert import TFBertModel
            from transformers import BertTokenizer, TFBertModel, BertConfig
            import transformers
            path = './Pre-training/BERT_base'
            bert_model2 = TFBertModel.from_pretrained(path)

            for layer in bert_model2.layers[:]:
                if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
                    layer.embeddings.trainable = False
            bert_model2.summary()

            # 第一个输入
            input1 = Input(shape=(essay_len,), dtype='int32')

            # emb_out1.shape == (None,600,300)
            emb_output = Embedding(args.vocab_size, 300, name='emb')(input1)
            mul_out = MultiHeadAttention(3, 100)(emb_output)  # (None，600,50)
            max0 = GlobalMaxPooling1D()(mul_out)
            avg0 = GlobalAveragePooling1D()(mul_out)
            mul_out = concatenate([avg0, max0], axis=-1)
            mul_out = Dropout(0.2)(mul_out)
            fully_connected_layers = [200, 300]
            for fl in fully_connected_layers:
                mul_out = Dense(fl, activation='relu')(mul_out)

            input_dependency = Input(shape=(overal_maxlen, overal_maxlen,), dtype=K.floatx())

            max = GlobalMaxPooling1D()(input_dependency)
            avg = GlobalAveragePooling1D()(input_dependency)
            parser_out = concatenate([avg, max], axis=-1)
            parser_out = Dropout(0.2)(parser_out)
            fully_connected_layers = [200, 300]
            for fl in fully_connected_layers:
                att_lstm_out = Dense(fl, activation='relu')(parser_out)


            input_ids = Input(shape=(overal_maxlen,), dtype='int32')
            input_mask = Input(shape=(overal_maxlen,), dtype='int32')
            input_tokentype = Input(shape=(overal_maxlen,), dtype='int32')


            ######   只冻结0-10层
            #冻结encoder部分参数
            for layer in bert_model2.layers[:]:
                if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
                    for idx, layer_input in enumerate(layer.encoder.layer):
                        layer_input.trainable = True
                        print(idx)
                        if idx in [0,1,2,3,4,5,6,7,8,9,10]:
                            layer_input.trainable = False
            bert_model2.summary()
            dim_out2 = bert_model2(
                {"input_ids": input_ids, "token_type_ids": input_tokentype, "attention_mask": input_mask})
            bert_2 = dim_out2.last_hidden_state
            max = GlobalMaxPooling1D()(bert_2)
            avg = GlobalAveragePooling1D()(bert_2)
            #bert_2 = concatenate([avg,mul_out], axis=-1)
            bert_2 = Dropout(0.2)(avg)
            fully_connected_layers = [200, 200]
            for fl in fully_connected_layers:
                bert_2 = Dense(fl, activation='relu')(bert_2)

            x = concatenate([att_lstm_out, bert_2], axis=-1)
            x = Dropout(0.2)(x)
            x = Dense(100, activation='swish')(x)
            x = Dense(100, activation='swish')(x)
            x = Dropout(0.1)(x)

            score = Dense(self.num_outpus, activation='sigmoid', name='score')(x)

            model = Model(inputs=[input1, input_dependency, input_ids, input_mask, input_tokentype], outputs=score)

#             bias_value = (np.log(self.initial_mean_value) - np.log(1 - self.initial_mean_value)).astype(K.floatx())
#             model.layers[-1].bias = bias_value  # 更改至 keras 2.0.8
            model.summary()
            from Pre_utils.w2vEmbReader import W2VEmbReader as EmbReader
            print('Initializing lookup table')
            emb_reader = EmbReader('embeddings.w2v.txt', emb_dim=50)
            # embedding_matrix = emb_reader.get_emb_matrix_given_vocab(vocab)
            # model.layers[model.emb_index].W.set_value(emb_reader.get_emb_matrix_given_vocab(vocab, model.layers[model.emb_index].W.get_value()))
#             model.get_layer(name='emb').set_weights(
#                 emb_reader.get_emb_matrix_given_vocab(vocab, model.get_layer(name='emb').get_weights()))  # 升级至2.0.8
            print('  Done')
            return model


'''
	BERT的最后输出有两种
last_hidden_state：维度【batch_size, seq_length, hidden_size】，这是训练后每个token的词向量。
pooler_output：维度是【batch_size, hidden_size】，每个sequence第一个位置CLS的向量输出，用于分类任务。
'''
