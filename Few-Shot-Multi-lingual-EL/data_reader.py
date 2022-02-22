import numpy as np
from tqdm import tqdm
import torch
import pdb
from typing import Iterator
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.fields import SpanField, ListField, TextField, MetadataField, ArrayField, SequenceLabelField, LabelField
from allennlp.data.tokenizers import Token
from utils import DatasetLoader, KBConstructor_fromKGemb
from overrides import overrides
import random
import transformers
from utils import from_original_sentence2left_mention_right_tokens_before_berttokenized

# SEEDS are FIXED
torch.backends.cudnn.deterministic = True
seed = 777
np.random.seed(seed)
torch.manual_seed(seed)

class TokenizeReader(DatasetReader):
    def __init__(self,args, entity_def_connection, token_indexers=None):
        super().__init__(args.allen_lazyload)
        self.args = args
        self.max_context_len = args.max_context_len
        self.max_canonical_len = args.max_canonical_len
        self.max_def_len = args.max_def_len

        self.token_indexers = self.token_indexer_returner()
        self.XLMRobertatokenizer = self.XLMRobertatokenizer_returner()

        linking_dataset_loader = DatasetLoader(args=args)
        self.src_id2line, self.src_train_mention_id, self.src_dev_mention_id, self.src_test_mention_id,\
        self.tgt_id2line, self.tgt_train_mention_id, self.tgt_dev_mention_id, self.tgt_test_mention_id= linking_dataset_loader.id2line_trn_dev_test_loader()

        print('loading KB')
        self.kbclass = KBConstructor_fromKGemb(args=self.args)
        self.setting_original_KB()
        print('original KB loaded')
        self.mention_start_token, self.mention_end_token = '[unused1]', '[unused2]'
        self.entity_def_connection = entity_def_connection

    def setting_original_KB(self):
        self.src_uni_to_id, self.src_id_to_uni, self.src_uni_to_name, self.src_uni_to_def,\
        self.tgt_uni_to_id, self.tgt_id_to_uni, self.tgt_uni_to_name, self.tgt_uni_to_def= self.kbclass.return_original_KB()
        # self.uni_to_id, self.id_to_uni, self.cui2emb, self.uni_to_name, self.uni_to_def = self.kbclass.return_original_KB()

    def currently_stored_KB_dataset_returner(self):
        return self.src_uni_to_id, self.src_id_to_uni, self.src_uni_to_name, self.src_uni_to_def,\
               self.tgt_uni_to_id, self.tgt_id_to_uni, self.tgt_uni_to_name, self.tgt_uni_to_def

    def huggingfacename_returner(self):
        if self.args.bert_name == 'bert-base-uncased':
            return 'bert-base-uncased', True
        else:
            return 'xlm-roberta-base', True

    def token_indexer_returner(self):
        huggingface_name, do_lower_case = self.huggingfacename_returner()
        return {'tokens': PretrainedTransformerIndexer(
                    model_name=huggingface_name,)
                    #do_lowercase=do_lower_case)
                }

    def XLMRobertatokenizer_returner(self):
        bos_token = '[BOS]'
        eos_token = '[EOS]'
        sep_token = '[SEP]'
        cls_token = '[CLS]'
        unk_token = '[UNK]'
        pad_token = '[pad]'
        mask_token = '[MASK]'
        from transformers import XLMRobertaTokenizer
        do_lower_case = True
        print(XLMRobertaTokenizer.from_pretrained("xlm-roberta-base").vocab_file)
        return transformers.XLMRobertaTokenizer(
            vocab_file=XLMRobertaTokenizer.from_pretrained("xlm-roberta-base").vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=True,
            never_split=['<A>','</A>'],
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            cls_token=cls_token,
            sep_token=sep_token,
            pad_token=pad_token,
            mask_token=mask_token
        )

    def tokenizer_custom(self, txt):
        target_anchors = ['<A>', '</A>']
        print(txt)
        original_tokens = txt.split(' ')
        new_tokens = list()

        for token in original_tokens:
            if token in target_anchors:
                new_tokens.append(token)
                continue
            else:
                split_to_subwords = self.XLMRobertatokenizer.tokenize(token) # token is oneword, split_tokens
                if ['[CLS]'] in  split_to_subwords:
                    split_to_subwords.remove('[CLS]')
                if ['[SEP]'] in  split_to_subwords:
                    split_to_subwords.remove('[SEP]')
                if split_to_subwords == []:
                    new_tokens.append('[UNK]')
                else:
                    new_tokens += split_to_subwords

        return new_tokens

    def mention_and_contexttokenizer_followblinkimplementation(self, txt):
        mention_start = '<A>'
        mention_end = '</A>'
        left, mention, right = from_original_sentence2left_mention_right_tokens_before_berttokenized(txt)

        new_tokens = list()
        new_tokens.append('[CLS]')

        if len(left) != 0:
            left_tokens = []
            for one_token in left:
                left_tokens += self.XLMRobertatokenizer.tokenize(one_token)
            new_tokens += left_tokens[:self.args.max_left_context_len]

        new_tokens.append(self.mention_start_token)
        if len(mention) != 0:
            mention_tokens = []
            for one_token in mention:
                mention_tokens += self.XLMRobertatokenizer.tokenize(one_token)
            new_tokens += mention_tokens[:self.args.max_mention_len]
        new_tokens.append(self.mention_end_token)

        if len(right) != 0:
            right_tokens = []
            for one_token in right:
                right_tokens += self.XLMRobertatokenizer.tokenize(one_token)
            new_tokens += right_tokens[:self.args.max_right_context_len]
        new_tokens.append('[SEP]')
        return new_tokens

    def find_anchor(self,split_txt,tobefoundtoken):
        for i, word in enumerate(split_txt):
            if word == tobefoundtoken:
                return i
        return -1

    def left_right_mention_sentence_from_anchorincludedsentence_returner(self, split_txt):
        i = self.find_anchor(split_txt=split_txt, tobefoundtoken='<A>') # mention start
        j = self.find_anchor(split_txt=split_txt, tobefoundtoken='</A>') # mention end

        sfm_mention = split_txt[i+1:j]
        raw_sentence_noanchor = [token for token in split_txt if not token in ['<A>', '</A>']]

        left_context_include_mention = split_txt[:j]
        left_context_include_mention.remove('<A>')
        right_context_include_mention = split_txt[i+1:]
        right_context_include_mention.remove('</A>')

        return raw_sentence_noanchor, sfm_mention, left_context_include_mention, right_context_include_mention

    @overrides
    def _read(self, train_dev_testflag) -> Iterator[Instance]:
        mention_ids = list()
        print('_read______:',train_dev_testflag)
        if train_dev_testflag == self.args.srclang+'_train':
            mention_ids += self.src_train_mention_id
            # Because original data is sorted with pmid documents, we have to shuffle data points for in-batch training.
            # random.shuffle(mention_ids)
        elif train_dev_testflag == self.args.srclang+'_dev':
            mention_ids += self.src_dev_mention_id
        elif train_dev_testflag == self.args.srclang+'_test':
            mention_ids += self.src_test_mention_id

        elif train_dev_testflag == self.args.tgtlang+'_train':
            mention_ids += self.tgt_train_mention_id
        elif train_dev_testflag == self.args.tgtlang+'_dev':
            mention_ids += self.tgt_dev_mention_id
        elif train_dev_testflag == self.args.tgtlang+'_test':
            mention_ids += self.tgt_test_mention_id

        if train_dev_testflag==self.args.srclang+'_train' or train_dev_testflag==self.args.srclang+'_dev' or train_dev_testflag==self.args.srclang+'_test':
            self.id2line = self.src_id2line
            self.temp_lang='src'
        else:
            self.id2line = self.tgt_id2line
            self.temp_lang = 'tgt'

        print('mention_id:',mention_ids)
        for idx, mention_uniq_id in tqdm(enumerate(mention_ids)):
            print('idx:',idx)
            print('mention_uniq_id:',mention_uniq_id)
            data = self.linesparser_for_blink_implementation(line=self.id2line[mention_uniq_id],
                                                             mention_uniq_id=mention_uniq_id,lang=self.temp_lang)
            yield self.text_to_instance(data=data)

    def lineparser_for_local_mentions(self, line, mention_uniq_id):
        gold_cui, gold_type, gold_surface_mention, targetanchor_included_sentence = line.split('\t')
        tokenized_context_including_target_anchors = self.tokenizer_custom(txt=targetanchor_included_sentence)
        raw_sentence_noanchor, sfm_mention, left_context_include_mention, right_context_include_mention = self.left_right_mention_sentence_from_anchorincludedsentence_returner(
            split_txt=tokenized_context_including_target_anchors)

        data = {}

        data['mention_uniq_id'] = mention_uniq_id
        data['gold_ids'] = gold_cui  # str
        data['gold_id_idx_with_uni_to_id'] = int(self.uni_to_id[gold_cui])
        data['mention_raw'] = gold_surface_mention
        data['raw_sentence_without_anchor_str'] = ' '.join(raw_sentence_noanchor)

        data['context'] = [Token(word) for word in raw_sentence_noanchor][:self.args.max_context_len]
        data['mention_preprocessed'] = [Token(word) for word in sfm_mention][:self.max_context_len]

        if len(left_context_include_mention) <= self.max_context_len:
            data['left_context_include_mention'] = [Token(word) for word in left_context_include_mention]
        else:
            data['left_context_include_mention'] = [Token(word) for word in left_context_include_mention][
                                                   len(left_context_include_mention) - self.max_context_len:]

        data['right_context_include_mention'] = [Token(word) for word in right_context_include_mention][:self.max_context_len]

        data['context'].insert(0, Token('[CLS]'))
        data['context'].insert(len(data['context']), Token('[SEP]'))
        data['mention_preprocessed'].insert(0, Token('[CLS]'))
        data['mention_preprocessed'].insert(len(data['mention_preprocessed']), Token('[SEP]'))
        data['left_context_include_mention'].insert(0, Token('[CLS]'))
        data['left_context_include_mention'].insert(len(data['left_context_include_mention']), Token('[SEP]'))
        data['right_context_include_mention'].insert(0, Token('[CLS]'))
        data['right_context_include_mention'].insert(len(data['right_context_include_mention']), Token('[SEP]'))

        data['gold_entity_def'] = self.gold_entity_def_returner(gold_cui=gold_cui)

        return data

    def linesparser_for_blink_implementation(self, line, mention_uniq_id,lang):
        gold_cui, gold_type, gold_surface_mention, targetanchor_included_sentence = line.split('\t')
        gold_cui = gold_cui.replace('UMLS:', '')
        print("line split:")
        print(gold_cui, gold_type, gold_surface_mention, targetanchor_included_sentence)
        tokenized_context_including_target_anchors = self.mention_and_contexttokenizer_followblinkimplementation(txt=targetanchor_included_sentence)
        tokenized_context_including_target_anchors = [Token(split_token) for split_token in tokenized_context_including_target_anchors]
        data = {}
        if lang == 'src':
            self.uni_to_id=self.src_uni_to_id
        else:
            self.uni_to_id=self.tgt_uni_to_id
        data['context'] = tokenized_context_including_target_anchors
        data['gold_entity_def'] = self.gold_entity_def_returner(gold_cui=gold_cui,lang=lang)
        print('self.uni_to_id[gold_cui]:',self.uni_to_id[gold_cui])
        data['gold_cuidx'] = int(self.uni_to_id[gold_cui])
        data['mention_uniq_id'] = int(mention_uniq_id)
        return data

    def gold_entity_def_returner(self, gold_cui, lang):
        if lang == 'src':
            self.uni_to_name=self.src_uni_to_name
            self.uni_to_def=self.src_uni_to_def
        else:
            self.uni_to_id=self.tgt_uni_to_id
            self.uni_to_def=self.tgt_uni_to_def
        canonical = self.tokenizer_custom(txt=self.uni_to_name[gold_cui])
        # choose the first def
        definition = self.tokenizer_custom(txt=self.uni_to_def[gold_cui][0])

        concatenated = ['[CLS]']
        # concatenated += canonical[:self.max_canonical_len]
        # concatenated.append(self.entity_def_connection)
        concatenated += definition[:self.max_def_len]
        concatenated.append('[SEP]')

        return [Token(tokenized_word) for tokenized_word in concatenated]

    # def to_be_ignored_mention_idx_checker(self):
    #     to_be_ignored_mention_idxs = []
    #     all_mention_idxs = list()
    #     all_mention_idxs += self.train_mention_id
    #     all_mention_idxs += self.dev_mention_id
    #     all_mention_idxs += self.test_mention_id
    #     print('self.uni_to_id:',self.uni_to_id)
    #     for mention_idx in all_mention_idxs:
    #         gold_cui_or_dui = self.id2line[mention_idx].split('\t')[0].replace('UMLS:', '')
    #         print('gold_cui_or_dui:',gold_cui_or_dui)
    #         if gold_cui_or_dui not in self.uni_to_id:
    #             to_be_ignored_mention_idxs.append(mention_idx)
    #     return to_be_ignored_mention_idxs

    @overrides
    def text_to_instance(self, data=None) -> Instance:
        if self.args.model_for_training == 'blink_implementation_inbatchencoder':
            context_field = TextField(data['context'], self.token_indexers)
            fields = {"context": context_field}
            fields['gold_entity_def'] = TextField(data['gold_entity_def'], self.token_indexers)
            fields['gold_cuidx'] = ArrayField(np.array(data['gold_cuidx']))
            fields['mention_uniq_id'] = ArrayField(np.array(data['mention_uniq_id']))
        else:
            context_field = TextField(data['context'], self.token_indexers)
            fields = {"context": context_field}
            surface_mention_field = TextField(data['mention_preprocessed'], self.token_indexers)
            fields['left_context_include_mention'] = TextField(data['left_context_include_mention'], self.token_indexers)
            fields['right_context_include_mention'] = TextField(data['right_context_include_mention'], self.token_indexers)
            fields['mention_processed'] = surface_mention_field
            fields['gold_entity_def'] = TextField(data['gold_entity_def'], self.token_indexers)
            fields['gold_id_for_knn'] = ArrayField(np.array(data['gold_id_idx_with_uni_to_id']))
        return Instance(fields)
'''
For encoding all entities, we need another datasetreader
'''
class AllEntityCanonical_and_Defs_loader(DatasetReader):
    def __init__(self, args, id_to_uni, uni_to_name, uni_to_def,
                 textfield_embedder, pretrained_tokenizer, tokenindexer, canonical_and_def_connect_token):
        super().__init__(args.allen_lazyload)

        self.args = args
        self.id_to_uni = id_to_uni
        self.uni_to_name = uni_to_name
        self.uni_to_def = uni_to_def
        self.textfield_embedder = textfield_embedder
        self.pretrained_tokenizer = pretrained_tokenizer
        self.token_indexers = tokenindexer
        self.canonical_and_def_connect_token = canonical_and_def_connect_token

    @overrides
    def _read(self,file_path=None) -> Iterator[Instance]:
        for idx, cui in tqdm(self.id_to_uni.items()):
            if self.args.debug_for_entity_encoder and idx==2100:
                break
            data = self.cui2data(cui=cui, idx=idx)
            yield self.text_to_instance(data=data)

    @overrides
    def text_to_instance(self, data=None) -> Instance:
        cano_and_def_concatenated = TextField(data['cano_and_def_concatenated'], self.token_indexers)
        fields = {"cano_and_def_concatenated": cano_and_def_concatenated, 'cui_idx':ArrayField(np.array(data['cui_idx'], dtype='int32'))}

        return Instance(fields)

    def tokenizer_custom(self, txt):
        original_tokens = txt.split(' ')
        new_tokens = list()

        for token in original_tokens:
            split_to_subwords = self.pretrained_tokenizer.tokenize(token) # token is oneword, split_tokens
            if ['[CLS]'] in split_to_subwords:
                split_to_subwords.remove('[CLS]')
            if ['[SEP]'] in split_to_subwords:
                split_to_subwords.remove('[SEP]')
            if split_to_subwords == []:
                new_tokens.append('[UNK]')
            else:
                new_tokens += split_to_subwords

        return new_tokens

    def cui2data(self, cui, idx):
        canonical_plus_definition = []
        canonical_plus_definition.append('[CLS]')

        canonical = self.uni_to_name[cui]
        canonical_tokens = [split_word for split_word in self.tokenizer_custom(txt=canonical)]
        canonical_plus_definition += canonical_tokens[:self.args.max_canonical_len]

        canonical_plus_definition.append(self.canonical_and_def_connect_token)

        definition = self.uni_to_def[cui][0]
        definition_tokens = [split_word for split_word in self.tokenizer_custom(txt=definition)]
        canonical_plus_definition += definition_tokens[:self.args.max_def_len]

        canonical_plus_definition.append('[SEP]')

        return {'cano_and_def_concatenated':[ Token(split_word_) for split_word_ in canonical_plus_definition],
                'cui_idx': idx}
