import codecs
import pickle


def store_data(i,j,train_x,dev_x,test_x,
               train_y,dev_y,test_y,dev_y_org,test_y_org,
               inputs_train_ids,inputs_dev_ids,inputs_test_ids,
               inputs_train_mask,inputs_dev_mask,inputs_test_mask,
               inputs_train_tokentype,inputs_dev_tokentype,inputs_test_tokentype):
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_train_x_ids.pkl', 'wb') as fout:
        pickle.dump(train_x, fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_dev_x_ids.pkl', 'wb') as fout:
        pickle.dump(dev_x, fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_test_x_ids.pkl', 'wb') as fout:
        pickle.dump(test_x, fout)
    fout.close()

    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_train_y.pkl', 'wb') as fout:
        pickle.dump(train_y, fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_dev_y.pkl', 'wb') as fout:
        pickle.dump(dev_y, fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_test_y.pkl', 'wb') as fout:
        pickle.dump(test_y, fout)
    fout.close()

    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_dev_y_org.pkl', 'wb') as fout:
        pickle.dump(dev_y_org, fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_test_y_org.pkl', 'wb') as fout:
        pickle.dump(test_y_org, fout)
    fout.close()

    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_inputs_train_ids.pkl', 'wb') as fout:
        pickle.dump(inputs_train_ids, fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_inputs_dev_ids.pkl', 'wb') as fout:
        pickle.dump(inputs_dev_ids, fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_inputs_test_ids.pkl', 'wb') as fout:
        pickle.dump(inputs_test_ids, fout)
    fout.close()

    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_inputs_train_mask.pkl', 'wb') as fout:
        pickle.dump(inputs_train_mask, fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_inputs_dev_mask.pkl', 'wb') as fout:
        pickle.dump(inputs_dev_mask, fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_inputs_test_mask.pkl', 'wb') as fout:
        pickle.dump(inputs_test_mask, fout)
    fout.close()

    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_inputs_train_tokentype.pkl', 'wb') as fout:
        pickle.dump(inputs_train_tokentype, fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_inputs_dev_tokentype.pkl', 'wb') as fout:
        pickle.dump(inputs_dev_tokentype, fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_inputs_test_tokentype.pkl', 'wb') as fout:
        pickle.dump(inputs_test_tokentype, fout)
    fout.close()

def get_store_data(i,j):
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_train_x_ids.pkl', 'rb') as fout:
        train_x = pickle.load(fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_dev_x_ids.pkl', 'rb') as fout:
        dev_x = pickle.load(fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_test_x_ids.pkl', 'rb') as fout:
        test_x = pickle.load(fout)
    fout.close()

    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_train_y.pkl', 'rb') as fout:
        train_y=pickle.load(fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_dev_y.pkl', 'rb') as fout:
        dev_y =pickle.load(fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_test_y.pkl', 'rb') as fout:
        test_y =pickle.load(fout)
    fout.close()

    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_dev_y_org.pkl', 'rb') as fout:
        dev_y_org=pickle.load(fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_test_y_org.pkl', 'rb') as fout:
        test_y_org=pickle.load(fout)
    fout.close()

    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_inputs_train_ids.pkl', 'rb') as fout:
        inputs_train_ids=pickle.load(fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_inputs_dev_ids.pkl', 'rb') as fout:
        inputs_dev_ids=pickle.load(fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_inputs_test_ids.pkl', 'rb') as fout:
        inputs_test_ids=pickle.load(fout)
    fout.close()

    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_inputs_train_mask.pkl', 'rb') as fout:
        inputs_train_mask =pickle.load(fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_inputs_dev_mask.pkl', 'rb') as fout:
        inputs_dev_mask =pickle.load(fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_inputs_test_mask.pkl', 'rb') as fout:
        inputs_test_mask =pickle.load(fout)
    fout.close()

    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_inputs_train_tokentype.pkl', 'rb') as fout:
        inputs_train_tokentype =pickle.load(fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_inputs_dev_tokentype.pkl', 'rb') as fout:
        inputs_dev_tokentype =pickle.load(fout)
    fout.close()
    with open('store/fold_' + str(i) + '/prompt_' + str(j) + '_inputs_test_tokentype.pkl', 'rb') as fout:
        inputs_test_tokentype = pickle.load(fout)
    fout.close()

    with open('output_dir/vocab.pkl', 'rb') as vocab_file:
        vocab = pickle.load(vocab_file)
    vocab_file.close()

    return  train_x, dev_x, test_x, train_y, dev_y, test_y, dev_y_org, test_y_org, inputs_train_ids,\
        inputs_dev_ids,inputs_test_ids,inputs_train_mask,inputs_dev_mask,inputs_test_mask,\
                   inputs_train_tokentype,inputs_dev_tokentype,inputs_test_tokentype,vocab


def get_bert():
    # 先获取两个bert
    from transformers.models.bert.modeling_tf_bert import TFBertModel
    from transformers import BertTokenizer, TFBertModel, BertConfig
    import transformers

    path = './Pre-training/BERT_base'
    path1 = './Pre-training/BERT_base_1'

    bert_model1 = TFBertModel.from_pretrained(path1)
    bert_model2 = TFBertModel.from_pretrained(path)

    ##冻结embeddings参数
    for layer in bert_model1.layers[:]:
        if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
            layer.embeddings.trainable = False
    bert_model1.summary()

    for layer in bert_model2.layers[:]:
        if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
            layer.embeddings.trainable = False
    bert_model2.summary()

    for layer in bert_model1.layers[:]:
        if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
            for idx, layer_inner in enumerate(layer.encoder.layer):
                print(idx)
                if idx in [0, 1, 2, 3, 4, 5, 6]:
                    layer_inner.trainable = False
    bert_model1.summary()

    for layer in bert_model2.layers[:]:
        if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
            for idx, layer_input in enumerate(layer.encoder.layer):
                layer_input.trainable = True
                print(idx)
                if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    layer_input.trainable = False
    bert_model2.summary()
    return bert_model1, bert_model2