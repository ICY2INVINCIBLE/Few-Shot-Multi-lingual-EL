import pdb, time, math
from allennlp.data.data_loaders import MultiProcessDataLoader, SimpleDataLoader
from parameters import Biencoder_params
from utils import experiment_logger, EmbLoader, ForOnlyFaiss_KBIndexer, cuda_device_parser, \
    parse_cuidx2encoded_emb_for_debugging
from utils import parse_cuidx2encoded_emb_2_cui2emb
from data_reader import TokenizeReader, AllEntityCanonical_and_Defs_loader
from allennlp.data.vocabulary import Vocabulary
from tqdm import tqdm
import logging
import torch.nn.functional as Fun

log = logging.getLogger(__name__)
from mention_and_entity_encoders import Entity_Pool,Mention_Pool,Encoder_Pool
from biencoder_models import Entity_Classifier, EntityenDecoding, BLINKBiencoder_OnlyforEncodingMentions
import torch.optim as optim
from KB_encoder import InKBAllEntitiesEncoder
from evaluator import BiEncoder_Retriever_Top, DevandTest_BLINKBiEncoder_IterateEvaluator
from Net import *
import random

CANONICAL_AND_DEF_CONNECTTOKEN = '[unused3]'

def main():
    Parameters = Biencoder_params()
    opts = Parameters.get_params()
    exp_start_time = time.time()
    experiment_logdir = experiment_logger(args=opts)
    Parameters.dump_params(experiment_dir=experiment_logdir)

    read_entity_for_mentions = TokenizeReader(args=opts,entity_def_connection=CANONICAL_AND_DEF_CONNECTTOKEN)
    src_train_data = list(read_entity_for_mentions._read(opts.srclang + '_train'))
    src_dev_data = list(read_entity_for_mentions._read(opts.srclang + '_dev'))
    # the train data was input to training process to conduct adversarial training
    # only put the mention
    tgt_train_data = list(read_entity_for_mentions._read(opts.tgtlang + '_train'))
    tgt_dev_data = list(read_entity_for_mentions._read(opts.tgtlang + '_dev'))
    tgt_test_data = list(read_entity_for_mentions._read(opts.tgtlang + '_test'))

    # print('type(train_data):',train_data)
    src_train_loader = SimpleDataLoader(src_train_data, opts.batch_size_for_train, shuffle=True)
    src_train_loader_Q = SimpleDataLoader(src_train_data, opts.batch_size_for_train, shuffle=True)

    tgt_train_loader = SimpleDataLoader(tgt_train_data, opts.batch_size_for_train, shuffle=True)
    tgt_train_loader_Q = SimpleDataLoader(tgt_train_data, opts.batch_size_for_train, shuffle=True)

    src_dev_loader = SimpleDataLoader(src_dev_data, opts.batch_size_for_train, shuffle=True)
    tgt_dev_loader = SimpleDataLoader(tgt_dev_data, opts.batch_size_for_train, shuffle=True)
    tgt_test_loader = SimpleDataLoader(tgt_test_data, opts.batch_size_for_train, shuffle=True)

    vocab = Vocabulary.from_instances(src_train_data)
    src_train_loader.index_with(vocab)
    src_train_loader_Q.index_with(vocab)

    vocab = Vocabulary.from_instances(tgt_train_data)
    tgt_train_loader.index_with(vocab)
    tgt_train_loader_Q.index_with(vocab)

    vocab = Vocabulary()
    # vocab = Vocabulary.from_instances(src_dev_data)
    src_dev_loader.index_with(vocab)
    # vocab = Vocabulary.from_instances(tgt_dev_data)
    tgt_dev_loader.index_with(vocab)
    # vocab = Vocabulary.from_instances(tgt_test_data)
    tgt_test_loader.index_with(vocab)

    src_train_iter_Q = iter(src_train_loader_Q)
    tgt_train_iter = iter(tgt_train_loader)
    tgt_train_iter_Q = iter(tgt_train_loader_Q)

    embloader = EmbLoader(args=opts)
    emb_mapper, emb_dim, textfieldEmbedder = embloader.emb_returner()

    mention_encoder = Mention_Pool(args=opts, word_embedder=textfieldEmbedder)
    src_current_uni_to_id, src_current_id_to_uni, src_current_uni_to_name, src_current_uni_to_def, \
    tgt_current_uni_to_id, tgt_current_id_to_uni, tgt_current_uni_to_name, tgt_current_uni_to_def = read_entity_for_mentions.currently_stored_KB_dataset_returner()
    entity_encoder = Entity_Pool(args=opts, word_embedder=textfieldEmbedder)

    R = Encoder_Pool(args=opts, word_embedder=textfieldEmbedder)
    F = LangDetectNet(opts.vocab_size, opts.kbemb_dim, opts.Q_layers, opts.hidden_size, opts.dropout)
    C = Entity_Classifier(args=opts,mention_encoder=mention_encoder, entity_encoder=entity_encoder,
                              vocab=vocab)

    optimizer = optim.Adam(filter(lambda param: param.requires_grad, list(C.parameters()) + list(F.parameters())),
                           lr=opts.lr, eps=opts.epsilon,
                           weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2), amsgrad=opts.amsgrad)
    optimizerF = optim.Adam(filter(lambda param: param.requires_grad, F.parameters()), lr=opts.lr, eps=opts.epsilon,
                            weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2), amsgrad=opts.amsgrad)

    for epoch in range(opts.num_epochs):
        R.train()
        C.train()
        F.train()
        src_train_iter = iter(src_train_loader)
        correct, total = 0, 0
        sum_src_q, sum_tgt_q = (0, 0.0), (0, 0.0)
        print('enumerate(src_train_iter):', enumerate(src_train_iter))
        i = 0
        for content in tqdm((src_train_iter), total=len(src_train_data) // opts.batch_size):
            src_input_context = content['context']
            src_input_gold_cui_cano_and_def_concatenated = content['gold_cui_cano_and_def_concatenated']
            src_input_gold_cuidx = content['gold_cuidx']
            src_input_mention_uniq_id = content['mention_uniq_id']
            # print(context,gold_cui_cano_and_def_concatenated,gold_cuidx,mention_uniq_id)
            try:
                tgt_src_content = next(tgt_train_iter)
                tgt_input_context = tgt_src_content['context']
                tgt_input_gold_cui_cano_and_def_concatenated = tgt_src_content['gold_cui_cano_and_def_concatenated']
                tgt_input_gold_cuidx = tgt_src_content['gold_cuidx']
                tgt_input_mention_uniq_id = tgt_src_content['mention_uniq_id']
            except StopIteration:
                tgt_train_iter=iter(tgt_train_loader)
                tgt_src_content = next(tgt_train_iter)
                tgt_input_context = tgt_src_content['context']
                tgt_input_gold_cui_cano_and_def_concatenated = tgt_src_content['gold_cui_cano_and_def_concatenated']
                tgt_input_gold_cuidx = tgt_src_content['gold_cuidx']
                tgt_input_mention_uniq_id = tgt_src_content['mention_uniq_id']
            # tgt_input_gold_cuidx,tgt_input_mention_uniq_id, tgt_input_context,tgt_input_gold_cui_cano_and_def_concatenated=next(tgt_train_iter)
            # print(tgt_input_gold_cuidx,tgt_input_mention_uniq_id, tgt_input_context,tgt_input_gold_cui_cano_and_def_concatenated)
            n_critic = opts.n_critic
            if n_critic > 0 and ((epoch == 0 and i <= 25) or (i % 500 == 0)):
                n_critic = 10
            for qiter in range(n_critic):
                try:
                    src_train_iter_Q=iter(src_train_loader_Q)
                    src_train_content_Q = next(src_train_iter_Q)
                    q_src_input_context = src_train_content_Q['context']
                    q_src_input_gold_cui_cano_and_def_concatenated = src_train_content_Q[
                        'gold_cui_cano_and_def_concatenated']
                    q_src_input_gold_cuidx = src_train_content_Q['gold_cuidx']
                    q_src_input_mention_uniq_id = src_train_content_Q['mention_uniq_id']
                except StopIteration:
                    src_train_iter_Q = iter(src_train_loader_Q)
                    src_train_content_Q = next(src_train_iter_Q)
                    q_src_input_context = src_train_content_Q['context']
                    q_src_input_gold_cui_cano_and_def_concatenated = src_train_content_Q[
                        'gold_cui_cano_and_def_concatenated']
                    q_src_input_gold_cuidx = src_train_content_Q['gold_cuidx']
                    q_src_input_mention_uniq_id = src_train_content_Q['mention_uniq_id']
                    # q_src_input_context, q_src_input_gold_cui_cano_and_def_concatenated, q_src_input_gold_cuidx, q_src_input_mention_uniq_id = next(src_train_iter_Q)
                try:
                    tgt_train_content_Q = next(tgt_train_iter_Q)
                    q_tgt_input_context = tgt_train_content_Q['context']
                    q_tgt_input_gold_cui_cano_and_def_concatenated = tgt_train_content_Q[
                        'gold_cui_cano_and_def_concatenated']
                    q_tgt_input_gold_cuidx = tgt_train_content_Q['gold_cuidx']
                    q_tgt_input_mention_uniq_id = tgt_train_content_Q['mention_uniq_id']
                except StopIteration:
                    tgt_train_iter_Q = iter(tgt_train_loader_Q)
                    tgt_train_content_Q = next(tgt_train_iter_Q)
                    q_tgt_input_context = tgt_train_content_Q['context']
                    q_tgt_input_gold_cui_cano_and_def_concatenated = tgt_train_content_Q[
                        'gold_cui_cano_and_def_concatenated']
                    q_tgt_input_gold_cuidx = tgt_train_content_Q['gold_cuidx']
                    q_tgt_input_mention_uniq_id = tgt_train_content_Q['mention_uniq_id']

                mu = random.random()
                print(q_src_input_context)
                representation_src_Q = R(q_src_input_context)
                representation_tgt_Q = R(q_tgt_input_context)
                print('represent:', representation_src_Q)
                # representation_hat = mu*representation_src+(1.0-mu)*representation_tgt

                o_src_ad = F(representation_src_Q.long())
                l_src_ad = torch.mean(o_src_ad)
                (-l_src_ad).backward()

                sum_tgt_q = (sum_src_q[0] + 1, sum_tgt_q[1] + l_src_ad.item())

                optimizerF.step()
            i += 1

            representation_src = R(src_input_context)
            representation_tgt = R(tgt_input_context)
            o_src_sent = C(representation_src, src_input_gold_cui_cano_and_def_concatenated, src_input_gold_cuidx, src_input_mention_uniq_id)
            print(o_src_sent)

            device = torch.device('cpu')
            batch_num = src_input_context['tokens']['token_ids'].size(0)
            target_src = torch.LongTensor(torch.arange(batch_num)).to(device)
            l_src_sent =  Fun.cross_entropy(o_src_sent, target_src, reduction="mean")

            # l_src_sent = functional.nll_loss(o_src_sent, target_src)
            l_src_sent.backward(retain_graph=True)
            o_src_ad = F(representation_src.long())
            l_src_ad = torch.mean(o_src_ad)
            (opts.lambd * l_src_ad).backward(retain_graph=True)

            _, pred = torch.max(o_src_sent, 1)
            total += target_src.size(0)
            correct += (pred == target_src).sum().item()

            representation_tgt = R(tgt_input_context)
            o_tgt_ad = F(representation_tgt.long())
            l_tgt_ad = torch.mean(o_tgt_ad)
            (-opts.lambd * l_tgt_ad).backward()

            optimizer.step()


if __name__ == '__main__':
    main()
