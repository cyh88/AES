import transformers
from transformers import BertTokenizer,AutoTokenizer
from transformers.models.bert.modeling_tf_bert import TFBertModel

class BertPreSetting:
    def __init__(self, args, overal_maxlen):
        if args.model_type == r'ernie':
            self.path = './Pre-training/ernie'
            self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        if args.model_type == r'roberta':
            self.path = './Pre-training/roberta'
            self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        else:
            self.path = r'./Pre-training/BERT_base'
            self.tokenizer = BertTokenizer.from_pretrained(self.path)
        self.args = args
        self.max_length = overal_maxlen

    def get_inputs(self, args, input_pre_x):
        encoded_inputs = self.tokenizer(input_pre_x, return_tensors='tf', padding=True, truncation=True,
                                        max_length=self.max_length)

        inputs_ids = encoded_inputs.get('input_ids')
        inputs_mask = encoded_inputs.get('attention_mask')
        inputs_tokentype = encoded_inputs.get('token_type_ids')

        return inputs_ids, inputs_mask, inputs_tokentype

    def get_emb(self,args,input_pre_x):
        model = TFBertModel.from_pretrained('./Pre-training/BERT_base')
        # 使用tokenizer对句子进行编码
        tokens = self.tokenizer.encode_plus(input_pre_x, add_special_tokens=True, return_tensors='tf',max_length= self.max_length)

        # 将编码后的tokens传递给BERT模型获取词向量
        outputs = model(tokens)

        # 获取句子的词向量
        sentence_embeddings = outputs[0]

        # 打印句子的词向量
        print(sentence_embeddings)

        return sentence_embeddings


