import argparse


class Config:
    def __init__(self):
        self.prompts = [6, 7, 8]
        self.dates = [0, 1, 2, 3, 4]
        parser = argparse.ArgumentParser()
        parser.add_argument("-tr", "--train", dest="train_path", type=str, metavar='<str>', required=True, help="训练集路径")
        parser.add_argument("-tu", "--tune", dest="dev_path", type=str, metavar='<str>', required=True, help="验证集路径")
        parser.add_argument("-ts", "--test", dest="test_path", type=str, metavar='<str>', required=True, help="测试集路径")
        parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True,
                            help="输出目录")
        parser.add_argument("-p", "--prompt", dest="prompt_id", type=int, metavar='<int>', default=0,
                            help="用来指定哪个数据集. '0' means all prompts.")
        parser.add_argument("-t", "--type", dest="model_type", type=str, metavar='<str>', default='regp',
                            help="Model type (reg|regp|breg|bregp) (default=regp)")
        parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='rmsprop',
                            help="优化器 (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
        parser.add_argument("-l", "--loss", dest="loss", type=str, metavar='<str>', default='mse',
                            help="损失函数 (mse|mae) (default=mse)")
        parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=300,
                            help="Embedding维度 (default=50)")
        parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, metavar='<int>', default=1,
                            help="Batch size (default=32)")
        parser.add_argument("-lr", dest="learning_rate", type=float, metavar='<float>', default=1e-5,
                            help="学习率. 若禁用，给予一个负数 (default=0.001)")
        parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=5000,
                            help="词表大小 (default=5000)")
        parser.add_argument("--aggregation", dest="aggregation", type=str, metavar='<str>', default='mot',
                            help="The aggregation method for regp and bregp types (mot|attsum|attmean) (default=mot)")
        parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.5,
                            help="dropout，禁用的话给予一个负数 (default=0.5)")
        parser.add_argument("--vocab-path", dest="vocab_path", type=str, metavar='<str>',
                            help="(Optional)现有词表的路径(*.pkl)")
        parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>',
                            help="embedding文件的路径 (Word2Vec format)")
        parser.add_argument("--tsp", dest="training_set_path", type=str, metavar='<str>', help="训练集路径")
        parser.add_argument("--pp", dest="prompt_path", type=str, metavar='<str>', help="用到的数据集的路径")
        parser.add_argument("--swp", dest="stop_word_path", type=str, metavar='<str>', help="停用词路径")
        parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=20,
                            help="epochs (default=60)")
        parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0,
                            help="最大字长，0为无限制 (default=0)")
        parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="随机种子 (default=1234)")
        parser.add_argument("-exp", "--explain", dest="explain", type=str, metavar='<str>', default="请输入说明",required=True, help="写入文本说明")
        parser.add_argument("--non_gate", dest="non_gate", action='store_true',
                            help="Model type (SWEM|regp|breg|bregp) (default=SWEM)")
        self.args = parser.parse_args()

    def get_parser(self):
        return self.args

    def get_prompt(self):
        if self.args.prompt_id != 0:
            prompt = [self.prompts[self.args.prompt_id - 1]]
        else:
            prompt = self.prompts
        return prompt
