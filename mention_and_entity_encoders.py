import torch
import torch.nn as nn
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper, BagOfEmbeddingsEncoder
from allennlp.modules.seq2vec_encoders import BertPooler
# from overrides import overrides
from allennlp.nn.util import get_text_field_mask, add_positional_features
# from transformers.models.roberta.modeling_roberta import RobertaPooler
# from transformers import RobertaModel

class Entity_Pool(Seq2VecEncoder):
    def __init__(self, args, word_embedder):
        super(Entity_Pool, self).__init__()
        self.args = args
        # self.huggingface_nameloader()
        self.bert_weight_filepath = 'xlm-roberta-base'
        self.bertpooler_sec2vec = BertPooler(pretrained_model=self.bert_weight_filepath)
        # self.bertpooler_sec2vec = RobertaModel.from_pretrained('xlm-roberta-base').pooler
        self.word_embedder = word_embedder
        self.word_embedding_dropout = nn.Dropout(self.args.word_embedding_dropout)

    def forward(self, entity_def_text):
        # print("cano_and_def_concatnated_text:",cano_and_def_concatnated_text)
        mask_sent = get_text_field_mask(entity_def_text)
        entity_emb = self.word_embedder(entity_def_text)
        entity_emb = self.word_embedding_dropout(entity_emb)
        entity_emb = self.bertpooler_sec2vec(entity_emb, mask_sent)

        # print(entity_emb)
        return entity_emb

class Mention_Pool(Seq2VecEncoder):
    def __init__(self, args, word_embedder):
        super(Mention_Pool, self).__init__()
        self.args = args
        # self.huggingface_nameloader()
        self.bert_weight_filepath = 'bert-base-uncased'
        self.bertpooler_sec2vec = BertPooler(pretrained_model=self.bert_weight_filepath)
        # self.bertpooler_sec2vec = RobertaModel.from_pretrained('xlm-roberta-base').pooler
        self.word_embedder = word_embedder
        self.word_embedding_dropout = nn.Dropout(self.args.word_embedding_dropout)

    def forward(self, contextualized_mention):
        mask_sent = get_text_field_mask(contextualized_mention)
        mention_emb = self.word_embedder(contextualized_mention)
        mention_emb = self.word_embedding_dropout(mention_emb)
        mention_emb = self.bertpooler_sec2vec(mention_emb, mask_sent)
        return mention_emb

class Encoder_Pool(Seq2VecEncoder):
    def __init__(self, args, word_embedder):
        super(Encoder_Pool, self).__init__()
        self.args = args
        # self.huggingface_nameloader()
        self.bert_weight_filepath = 'xlm-roberta-base'
        self.bertpooler_sec2vec = BertPooler(pretrained_model=self.bert_weight_filepath)
        # self.bertpooler_sec2vec = RobertaModel.from_pretrained('xlm-roberta-base').pooler
        self.word_embedder = word_embedder
        self.word_embedding_dropout = nn.Dropout(self.args.word_embedding_dropout)

    def forward(self, text):
        mask_sent = get_text_field_mask(text)
        emb = self.word_embedder(text)
        emb = self.word_embedding_dropout(emb)
        emb = self.bertpooler_sec2vec(emb, mask_sent)
        return emb
