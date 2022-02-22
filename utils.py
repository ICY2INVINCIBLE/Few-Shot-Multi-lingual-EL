import os, json, time, gc, copy, shutil, random, pickle, sys, pdb
from datetime import datetime
import numpy as np
# from allennlp.common.tee_logger import TeeLogger
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from pytz import timezone
import faiss
import torch
import torch.nn as nn
from tqdm import tqdm

def cuda_device_parser(str_ids):
    return [int(stridx) for stridx in str_ids.strip().split(',')]

def from_original_sentence2left_mention_right_tokens_before_berttokenized(sentence):
    mention_start = '<target>'
    mention_end = '</target>'
    print('sentence:',sentence)
    original_tokens = sentence.split(' ')
    print('original_tokens:',original_tokens)
    mention_start_idx = int(original_tokens.index(mention_start))
    mention_end_idx = int(original_tokens.index(mention_end))
    if mention_end_idx == len(sentence) - 1 :
        return original_tokens[:mention_start_idx], original_tokens[mention_start_idx+1:mention_end_idx], []
    else:
        return original_tokens[:mention_start_idx], original_tokens[mention_start_idx+1:mention_end_idx], original_tokens[mention_end_idx+1:]

def parse_cuidx2encoded_emb_for_debugging(cuidx2encoded_emb, original_uni_to_id):
    print('/////Some entities embs are randomized for debugging./////')
    for cuidx in tqdm(original_uni_to_id.values()):
        if cuidx not in cuidx2encoded_emb:
            cuidx2encoded_emb.update({cuidx:np.random.randn(*cuidx2encoded_emb[0].shape)})
    return cuidx2encoded_emb

def parse_cuidx2encoded_emb_2_cui2emb(cuidx2encoded_emb, original_uni_to_id):
    cui2emb = {}
    for cui, idx in original_uni_to_id.items():
        cui2emb.update({cui:cuidx2encoded_emb[idx]})
    return cui2emb

def experiment_logger(args):
    '''
    :param args: from biencoder_parameters
    :return: dirs for experiment log
    '''
    experimet_logdir = args.experiment_logdir # / is included

    timestamp = datetime.now(timezone('Asia/Tokyo'))
    str_timestamp = '{0:%Y%m%d_%H%M%S}'.format(timestamp)[2:]

    dir_for_each_experiment = experimet_logdir + str_timestamp

    if os.path.exists(dir_for_each_experiment):
        dir_for_each_experiment += '_d'

    dir_for_each_experiment += '/'
    logger_path = dir_for_each_experiment + 'teelog.log'
    os.mkdir(dir_for_each_experiment)

    # if not args.debug:
    #     sys.stdout = TeeLogger(logger_path, sys.stdout, False)  # default: False
    #     sys.stderr = TeeLogger(logger_path, sys.stderr, False)  # default: False

    return dir_for_each_experiment

def from_jsonpath_2_str2idx(json_path):
    str2intidx = {}
    with open(json_path, 'r') as f:
        tmp_str2stridx = json.load(f)
    for str_key, str_idx in tmp_str2stridx.items():
        str2intidx.update({str_key:int(str_idx)})
    return str2intidx

def from_jsonpath_2_idx2str(json_path):
    intidx2str = {}
    with open(json_path, 'r') as f:
        tmp_stridx2str = json.load(f)
    for str_idx, value_str in tmp_stridx2str.items():
        intidx2str.update({int(str_idx):value_str})
    return intidx2str

def from_jsonpath_2_str2strorlist(json_path):
    with open(json_path, 'r',encoding='utf-8') as f:
        raw_json = json.load(f)
    return raw_json

def pklloader(pkl_path):
    with open(pkl_path, 'rb') as p:
        loaded = pickle.load(p)
    return loaded

class EmbLoader:
    def __init__(self, args):
        self.args = args

    def emb_returner(self):
        if self.args.bert_name == 'bert-base-uncased':
            huggingface_model = 'bert-base-uncased'
        else:
            huggingface_model = 'xlm-roberta-base'
        # elif self.args.bert_name == 'biobert':
        #     assert self.args.ifbert_use_whichmodel == 'biobert'
        #     huggingface_model = './biobert_transformers/'
        # else:
        #     huggingface_model = 'dummy'
        #     print(self.args.bert_name,'are not supported')
        #     exit()
        bert_embedder = PretrainedTransformerEmbedder(model_name=self.args.bert_name)
        return bert_embedder, bert_embedder.get_output_dim(), BasicTextFieldEmbedder({'tokens': bert_embedder},)
                                                                                     #allow_unmatched_keys=True)

class DatasetLoader:

    def __init__(self, args):
        self.args = args
        self.dataset = self.args.dataset
        # data/en_dataset/
        # data/de_dataset/
        self.src_dataset_dir = './data/' + self.args.srclang+'_'+self.dataset + '/'
        self.tgt_dataset_dir = './data/' + self.args.tgtlang + '_' + self.dataset + '/'

    def fixed_idxnized_datapath_returner(self):
        # data/en_dataset/en_mention_line.json
        src_mention_line_json_path = self.src_dataset_dir +self.args.srclang+'_mention_line.json'
        src_train_mentionidpath = self.src_dataset_dir + self.args.srclang+'_train_mentionid.pkl'
        src_dev_mentionidpath = self.src_dataset_dir + self.args.srclang+'_dev_mentionid.pkl'
        src_test_mentionidpath = self.src_dataset_dir + self.args.srclang+'_test_mentionid.pkl'
        # data/de_dataset/de_mention_line.json
        tgt_mention_line_json_path = self.tgt_dataset_dir + self.args.tgtlang + '_mention_line.json'
        tgt_train_mentionidpath = self.tgt_dataset_dir + self.args.tgtlang + '_train_mentionid.pkl'
        tgt_dev_mentionidpath = self.tgt_dataset_dir + self.args.tgtlang + '_dev_mentionid.pkl'
        tgt_test_mentionidpath = self.tgt_dataset_dir + self.args.tgtlang+'_test_mentionid.pkl'

        return src_mention_line_json_path,src_train_mentionidpath,src_dev_mentionidpath,src_test_mentionidpath,\
               tgt_mention_line_json_path,tgt_train_mentionidpath,tgt_dev_mentionidpath,tgt_test_mentionidpath
        # return id2line_json_path, train_mentionidpath, dev_mentionidpath, test_mentionidpath,trainQ_mentionidpath

    def id2line_path_2_intid2line(self, mention_line_json_path):
        with open(mention_line_json_path, 'r',encoding='utf-8') as id2l:
            tmp_id2l = json.load(id2l)
        intid2line = {}
        for str_idx, line_mention in tmp_id2l.items():
            intid2line.update({int(str_idx): line_mention})

        return intid2line

    def train_dev_test_mentionid_returner(self, src_train_mentionidpath, src_dev_mentionidpath, src_test_mentionidpath,
                                          tgt_train_mentionidpath, tgt_dev_mentionidpath, tgt_test_mentionidpath):
        with open(src_train_mentionidpath, 'rb') as trp:
            src_train_mentionid = pickle.load(trp)
        with open(src_dev_mentionidpath, 'rb') as drp:
            src_dev_mentionid = pickle.load(drp)
        with open(src_test_mentionidpath, 'rb') as terp:
            src_test_mentionid = pickle.load(terp)

        with open(tgt_train_mentionidpath, 'rb') as trp:
            tgt_train_mentionid = pickle.load(trp)
        with open(tgt_dev_mentionidpath, 'rb') as drp:
            tgt_dev_mentionid = pickle.load(drp)
        with open(tgt_test_mentionidpath, 'rb') as terp:
            tgt_test_mentionid = pickle.load(terp)

        return src_train_mentionid,src_dev_mentionid,src_test_mentionid,\
               tgt_train_mentionid,tgt_dev_mentionid,tgt_test_mentionid
        # if self.args.debug:
        #     return train_mentionid[:300], dev_mentionid[:200], test_mentionid[:400]
        # else:
        #     print(type(train_mentionid))
        #     return train_mentionid, dev_mentionid, test_mentionid,trainQ_mentionid

    def id2line_trn_dev_test_loader(self):
        # mention_json, and pkl file
        src_mention_line_json_path, src_train_mentionidpath, src_dev_mentionidpath, src_test_mentionidpath, \
        tgt_mention_line_json_path, tgt_train_mentionidpath, tgt_dev_mentionidpath, tgt_test_mentionidpath= self.fixed_idxnized_datapath_returner()

        src_id2line = self.id2line_path_2_intid2line(mention_line_json_path=src_mention_line_json_path)
        tgt_id2line = self.id2line_path_2_intid2line(mention_line_json_path=tgt_mention_line_json_path)
        # src_id2line: {0: '0\ten\tNorway\t<li>redirect <a>Syrian-born youth killed while building bomb in  <target> Norway </target>
        # tgt_id2line {0: '0\tja\tフジテレビ\t女子<a>バレーボール・ワールドグランプリ2005</a>の中継を担当した <target> フジテレビ </target>
        '''
            "id": 1,
            "context": "Which books by Kerouac were  <A> publis </A> hed by Viking Press?",
            "mention": "publish",
            "candidate": ['Publishing is the activity of making information, literature, music, software and other content available to the public for sale or for free.',
                            'P, or p, is the sixteenth letter of the modern English alphabet and the ISO basic Latin alphabet.']
        '''

        # [0, 1] [2, 3] [4, 5]
        src_train_mentionid, src_dev_mentionid, src_test_mentionid,\
        tgt_train_mentionid, tgt_dev_mentionid, tgt_test_mentionid = self.train_dev_test_mentionid_returner(
            src_train_mentionidpath=src_train_mentionidpath,
            src_dev_mentionidpath=src_dev_mentionidpath,
            src_test_mentionidpath=src_test_mentionidpath,
            tgt_train_mentionidpath=tgt_train_mentionidpath,
            tgt_dev_mentionidpath=tgt_dev_mentionidpath,
            tgt_test_mentionidpath=tgt_test_mentionidpath
        )

        return src_id2line, src_train_mentionid, src_dev_mentionid, src_test_mentionid,\
               tgt_id2line, tgt_train_mentionid, tgt_dev_mentionid, tgt_test_mentionid

class KBConstructor_fromKGemb:
    def __init__(self, args):
        self.args = args
        self.kbemb_dim = self.args.kbemb_dim
        self.original_kbloader_to_memory()

    def original_kbloader_to_memory(self):
        # uni_to_id_path, id_to_uni_path, cui2emb_path, uni_to_name_path, uni_to_def_path = self.from_datasetname_return_related_dicts_paths()
        src_uni_to_id_path, src_id_to_uni_path, src_uni_to_name_path, src_uni_to_def_path,\
        tgt_uni_to_id_path, tgt_id_to_uni_path, tgt_uni_to_name_path, tgt_uni_to_def_path= self.from_datasetname_return_related_dicts_paths()
        print('set value and load original KB')
        self.src_original_uni_to_id = from_jsonpath_2_str2idx(src_uni_to_id_path)
        self.src_original_id_to_uni = from_jsonpath_2_idx2str(src_id_to_uni_path)
        # self.original_cui2emb = pklloader(cui2emb_path)
        # print('self.original_cui2emb:',self.original_cui2emb)
        self.src_original_uni_to_name = from_jsonpath_2_str2strorlist(src_uni_to_name_path)
        self.src_original_uni_to_def = from_jsonpath_2_str2strorlist(src_uni_to_def_path)

        self.tgt_original_uni_to_id = from_jsonpath_2_str2idx(tgt_uni_to_id_path)
        self.tgt_original_id_to_uni = from_jsonpath_2_idx2str(tgt_id_to_uni_path)
        # self.original_cui2emb = pklloader(cui2emb_path)
        # print('self.original_cui2emb:',self.original_cui2emb)
        self.tgt_original_uni_to_name = from_jsonpath_2_str2strorlist(tgt_uni_to_name_path)
        self.tgt_original_uni_to_def = from_jsonpath_2_str2strorlist(tgt_uni_to_def_path)

    def return_original_KB(self):
        return self.src_original_uni_to_id, self.src_original_id_to_uni, self.src_original_uni_to_name, self.src_original_uni_to_def,\
               self.tgt_original_uni_to_id, self.tgt_original_id_to_uni, self.tgt_original_uni_to_name, self.tgt_original_uni_to_def
        # return self.original_uni_to_id, self.original_id_to_uni, self.original_cui2emb, self.original_uni_to_name, self.original_uni_to_def

    def from_datasetname_return_related_dicts_paths(self):
        src_uni_to_id_path = './' + 'data/' + self.args.srclang + '_dataset/' + self.args.srclang + '_uni_to_id.json'
        src_id_to_uni_path = './' + 'data/' + self.args.srclang + '_dataset/' + self.args.srclang + '_id_to_uni.json'
        # cui2emb_path = './dataset/cui2emb.pkl'
        src_uni_to_name_path = './' + 'data/' + self.args.srclang + '_dataset/' + self.args.srclang + '_uni_to_name.json'
        src_uni_to_def_path = './' + 'data/' + self.args.srclang + '_dataset/' + self.args.srclang + '_uni_to_def.json'

        tgt_uni_to_id_path = './' + 'data/' + self.args.tgtlang + '_dataset/' + self.args.tgtlang + '_uni_to_id.json'
        tgt_id_to_uni_path = './' + 'data/' + self.args.tgtlang + '_dataset/' + self.args.tgtlang + '_id_to_uni.json'
        # cui2emb_path = './dataset/cui2emb.pkl'
        tgt_uni_to_name_path = './' + 'data/' + self.args.tgtlang + '_dataset/' + self.args.tgtlang + '_uni_to_name.json'
        tgt_uni_to_def_path = './' + 'data/' + self.args.tgtlang + '_dataset/' + self.args.tgtlang + '_uni_to_def.json'

        return src_uni_to_id_path, src_id_to_uni_path, src_uni_to_name_path, src_uni_to_def_path, \
               tgt_uni_to_id_path, tgt_id_to_uni_path, tgt_uni_to_name_path, tgt_uni_to_def_path

        # else:
        #     # uni_to_id_path, id_to_uni_path, cui2emb_path, uni_to_name_path, uni_to_def_path = ['dummy' for i in range(5)]
        #     uni_to_id_path, id_to_uni_path, uni_to_name_path, uni_to_def_path = ['dummy' for i in range(4)]
        #     print(self.args.dataset, 'are currently not supported')
        #     exit()

        # return uni_to_id_path, id_to_uni_path, cui2emb_path, uni_to_name_path, uni_to_def_path
        # return uni_to_id_path, id_to_uni_path,uni_to_name_path, uni_to_def_path


    def load_original_KBmatrix_alignedwith_id_to_uni(self):
        KBemb = np.random.randn(len(self.original_cui2emb.keys()), self.kbemb_dim).astype('float32')

        for idx, cui in self.original_id_to_uni.items():
            KBemb[idx] = self.original_cui2emb[cui]

        return KBemb

    def indexed_faiss_loader_for_constructing_smallKB(self):
        if self.args.search_method_for_faiss_during_construct_smallKBfortrain == 'indexflatl2':  # L2
            self.indexed_faiss = faiss.IndexFlatL2(self.kbemb_dim)
        elif self.args.search_method_for_faiss_during_construct_smallKBfortrain == 'indexflatip':  # innerdot * Beforehand-Normalization must be done.
            self.indexed_faiss = faiss.IndexFlatIP(self.kbemb_dim)
        elif self.args.search_method_for_faiss_during_construct_smallKBfortrain == 'cossim':  # innerdot * Beforehand-Normalization must be done.
            self.indexed_faiss = faiss.IndexFlatIP(self.kbemb_dim)
        else:
            print('currently',self.args.search_method_for_faiss_during_construct_smallKBfortrain, 'are not supported')
            exit()

        return self.indexed_faiss

class ForOnlyFaiss_KBIndexer:
    def __init__(self, args, input_uni_to_id, input_id_to_uni, input_cui2emb, search_method_for_faiss, entity_emb_dim=300):
        self.args = args
        self.kbemb_dim = entity_emb_dim
        self.uni_to_id = input_uni_to_id
        self.id_to_uni = input_id_to_uni
        self.cui2emb = input_cui2emb
        self.search_method_for_faiss = search_method_for_faiss
        self.indexed_faiss_loader()
        self.KBmatrix = self.KBmatrixloader()
        self.entity_num = len(input_uni_to_id)
        self.indexed_faiss_KBemb_adder(KBmatrix=self.KBmatrix)

    def KBmatrixloader(self):
        KBemb = np.random.randn(len(self.uni_to_id.keys()), self.kbemb_dim).astype('float32')
        for idx, cui in self.id_to_uni.items():
            KBemb[idx] = self.cui2emb[cui].astype('float32')

        return KBemb

    def indexed_faiss_loader(self):
        if self.search_method_for_faiss == 'indexflatl2':  # L2
            self.indexed_faiss = faiss.IndexFlatL2(self.kbemb_dim)
        elif self.search_method_for_faiss == 'indexflatip':  #
            self.indexed_faiss = faiss.IndexFlatIP(self.kbemb_dim)
        elif self.search_method_for_faiss == 'cossim':  # innerdot * Beforehand-Normalization must be done.
            self.indexed_faiss = faiss.IndexFlatIP(self.kbemb_dim)

    def indexed_faiss_KBemb_adder(self, KBmatrix):
        if self.search_method_for_faiss == 'cossim':
            KBemb_normalized_for_cossimonly = np.random.randn(self.entity_num, self.kbemb_dim).astype('float32')
            for idx, emb in enumerate(KBmatrix):
                if np.linalg.norm(emb, ord=2, axis=0) != 0:
                    KBemb_normalized_for_cossimonly[idx] = emb / np.linalg.norm(emb, ord=2, axis=0)
            self.indexed_faiss.add(KBemb_normalized_for_cossimonly)
        else:
            self.indexed_faiss.add(KBmatrix)

    def indexed_faiss_returner(self):
        return self.indexed_faiss

    def KBembeddings_loader(self):
        KBembeddings = nn.Embedding(self.entity_num, self.kbemb_dim, padding_idx=0)
        KBembeddings.weight.data.copy_(torch.from_numpy(self.KBmatrix))
        KBembeddings.weight.requires_grad = False
        return KBembeddings

