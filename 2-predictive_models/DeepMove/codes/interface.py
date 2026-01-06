import os
from os.path import join
from enum import Enum
from random import random

from luigi import (
    Task, ExternalTask, WrapperTask,
    LocalTarget, 
    Parameter, IntParameter, FloatParameter, EnumParameter,
)

from DeepMove.codes.preprocess import preprocess
from DeepMove.codes.main import run as train_model
from DeepMove.codes.perplexity


class DataFile(ExternalTask):
    path = Parameter()

    def output(self):
        return LocalTarget(self.path)

class DataDirectory(WrapperTask):
    path = Parameter()

    def requires(self):
        return [DataFile(join(self.path, file)) for file in os.listdir(self.path) if file.endswith('.csv')]

class Preprocess(Task):
    in_dir = Parameter()
    out_dir = Parameter()
    training_set_name = Parameter(default='training_set')

    @property
    def files(self):
        return [file for file in os.listdir(self.in_dir) if file.endswith('.csv')]

    def run(self):
        preprocess(
            in_dir=self.in_dir,
            out_dir=self.out_dir,
            training_set_name=self.training_set_name,
        )

    def requires(self):
        return DataDirectory(self.in_dir)

    def output(self):
        os.makedirs(join(self.out_dir, 'preprocessed'), exist_ok=True)
        preproc = join(self.out_dir, 'preprocessed')
        return [LocalTarget(join(self.out_dir, 'metadata.json'))] + \
            [LocalTarget(join(preproc, file.replace('.csv', '.pk'))) for file in self.files]


"""
Training parameters from the previous argparse interface:
parser.add_argument('--loc_emb_size', type=int, default=500, help="location embeddings size")
parser.add_argument('--uid_emb_size', type=int, default=40, help="user id embeddings size")
parser.add_argument('--voc_emb_size', type=int, default=50, help="words embeddings size")
parser.add_argument('--tim_emb_size', type=int, default=10, help="time embeddings size")
parser.add_argument('--hidden_size', type=int, default=500)
parser.add_argument('--dropout_p', type=float, default=0.3)
parser.add_argument('--data_name', type=str, default='foursquare')
parser.add_argument('--learning_rate', type=float, default=1/3 * 1e-3)
parser.add_argument('--lr_step', type=int, default=4, help="how many epochs before reducing learning rate")
parser.add_argument('--lr_decay', type=float, default=1/3)
parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
parser.add_argument('--L2', type=float, default=1 * 1e-5, help=" weight decay (L2 penalty)")
parser.add_argument('--clip', type=float, default=5.0)
parser.add_argument('--epoch_max', type=int, default=20)
parser.add_argument('--early_stopping', type=int, default=5, help="number of epochs to wait for validation loss improvement before early stopping")
parser.add_argument('--history_mode', type=str, default='avg', choices=['max', 'avg', 'whole'])
parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU', 'RNN'])
parser.add_argument('--attn_type', type=str, default='dot', choices=['general', 'concat', 'dot'])
parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--save_dir', type=str, default='rez/')
parser.add_argument('--save_model_path', type=str,
                    help="path to save the trained model")
parser.add_argument('--model_mode', type=str, default='simple_long',
                    choices=['simple', 'simple_long', 'attn_avg_long_user', 'attn_local_long'])
parser.add_argument('--pretrain', type=int, default=0)
parser.add_argument('--metadata_json', type=str, default='data/foursquare/metadata.json', help="path to metadata json file")
parser.add_argument('--pretrain_model_path', type=str, default='../pretrain/simple_long/res.m',
                    help="path to pretrained model file")
"""
class ModelMode(Enum):
    simple = 'simple'
    simple_long = 'simple_long'
    attn_avg_long_user = 'attn_avg_long_user'
    attn_local_long = 'attn_local_long'

class Optimizer(Enum):
    Adam = 'Adam'
    SGD = 'SGD'

class RNNType(Enum):
    LSTM = 'LSTM'
    GRU = 'GRU'
    RNN = 'RNN'

class AttentionType(Enum):
    general = 'general'
    concat = 'concat'
    dot = 'dot'

class HistoryMode(Enum):
    max = 'max'
    avg = 'avg'
    whole = 'whole'



class Train(Task):
    in_dir = Parameter()
    out_dir = Parameter(default=f'runs/run-{int(random()*1e10):09d}')
    training_set_name = Parameter(default='training_set')

    loc_emb_size = IntParameter(default=500, description="location embeddings size")
    uid_emb_size = IntParameter(default=40, description="user id embeddings size")
    voc_emb_size = IntParameter(default=50, description="words embeddings size")
    tim_emb_size = IntParameter(default=10, description="time embeddings size")
    hidden_size = IntParameter(default=500, description="hidden size")
    dropout_p = FloatParameter(default=0.3, description="dropout probability")
    learning_rate = FloatParameter(default=1/3 * 1e-3, description="learning rate")
    lr_step = IntParameter(default=4, description="how many epochs before reducing learning rate")
    lr_decay = FloatParameter(default=1/3, description="learning rate decay factor")
    optim = EnumParameter(enum=Optimizer, default=Optimizer.Adam)
    L2 = FloatParameter(default=1 * 1e-5, description=" weight decay (L2 penalty)")
    clip = FloatParameter(default=5.0, description="gradient clipping value")
    epoch_max = IntParameter(default=20, description="maximum number of epochs")
    early_stopping = IntParameter(default=5, description="number of epochs to wait for validation loss improvement before early stopping")
    history_mode = EnumParameter(enum=HistoryMode, default=HistoryMode.avg)
    rnn_type = EnumParameter(enum=RNNType, default=RNNType.LSTM)
    attn_type = EnumParameter(enum=AttentionType, default=AttentionType.dot)
    model_mode = EnumParameter(enum=ModelMode, default=ModelMode.simple_long)

    pretrain = IntParameter(default=0, description="whether to use pretraining (1) or not (0)")
    pretrain_model_path = Parameter(default='../pretrain/simple_long/res.m', description="path to pretrained model file")


    def run(self):
        args = {param[0]: getattr(self, param[0]) for param in self.get_params()}
        # turn values of EnumParameter into their .value
        args = {k: (v.value if isinstance(v, Enum) else v) for k, v in args.items()}
        preproc_dir = os.path.dirname(self.input()[-1].path)
        train_model(
            data_path=join(preproc_dir, self.training_set_name) + '.pk',
            save_dir=self.out_dir,
            metadata_json=self.input()[0].path,
            **args
        )
    
    def requires(self):
        return Preprocess(self.in_dir, self.out_dir, self.training_set_name)

    def output(self):
        model = LocalTarget(join(self.out_dir, 'res.m'))
        learning_curve = LocalTarget(join(self.out_dir, 'res.rs'))
        return [model, learning_curve]
    

class PerplexitySingle(Task):
    in_file = Parameter()
    out_dir = Parameter(default=f'runs/run-{int(random()*1e10):09d}/perplexities')

    def output(self):
        os.makedirs(self.out_dir, exist_ok=True)
        filename = os.path.basename(self.in_file)
        out_filename = filename.split('.')[0] + '_perplexity.csv'
        return LocalTarget(join(self.out_dir, out_filename))
    
    def requires(self):
        in_dir = os.path.dirname(self.in_file)
        return Train(in_dir, self.out_dir)

    def run(self):
        pass