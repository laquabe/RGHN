import argparse

import torch
from align.util import gcn_load_data
from mynet_torch import TrainerTorch

parser = argparse.ArgumentParser(description='MyNet')
parser.add_argument('--input', type=str, default="../data/OpenEA/D_Y_15K_V1/721_5fold/1/")
parser.add_argument('--output', type=str, default='../output/results/')

parser.add_argument('--embedding_module', type=str, default='TrainerTorch')
parser.add_argument('--gcn', type=str, default='RGC')
parser.add_argument('--layer_dims', type=list, default=[256, 256 ,256, 256, 256])  
parser.add_argument('--num_features_nonzero', type=float, default=0.0)

parser.add_argument('--neg_multi', type=int, default=50)  # for negative sampling
parser.add_argument('--neg_margin', type=float, default=1.5)  # margin value for negative loss
parser.add_argument('--neg_param', type=float, default=0.1)  # weight for negative loss
parser.add_argument('--rel_param', type=float, default=0.2)  # weight for relation loss
parser.add_argument('--rel_align_param', type=float, default=1)  # weight for relation alignment loss
parser.add_argument('--threshold', type=float, default=0.5) # soft label
parser.add_argument('--truncated_epsilon', type=float, default=0.98)  # epsilon for truncated negative sampling
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--model_name', type=str, default='baseline')
parser.add_argument('--inverse_relation', type=str, default='lin')

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--min_rel_win', type=int, default=15)
parser.add_argument('--max_epoch', type=int, default=1000)

parser.add_argument('--is_save', type=bool, default=False)
parser.add_argument('--start_valid', type=int, default=25)

parser.add_argument('--eval_metric', type=str, default='inner')
parser.add_argument('--hits_k', type=list, default=[1, 5, 10, 50])
parser.add_argument('--eval_threads_num', type=int, default=10)
parser.add_argument('--eval_normalize', type=bool, default=True)
parser.add_argument('--eval_csls', type=int, default=10)
parser.add_argument('--eval_freq', type=int, default=25)
parser.add_argument('--adj_number', type=int, default=1)


class ModelFamily(object):
    TrainerTorch = TrainerTorch


def get_model(model_name):
    return getattr(ModelFamily, model_name)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    is_two = False
    _, kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, tri_num, ent_num, rel_num, _, rel_ht_dict, rel1_list, rel2_list = \
        gcn_load_data(args.input, is_two=is_two)
    import pickle
    with open('ref_ent.pkl','wb') as f:
        pickle.dump({'ref1':ref_ent1, 'ref2':ref_ent2}, f)
    
    linked_entities = set(sup_ent1 + sup_ent2 + ref_ent1 + ref_ent2)

    gcn_model = get_model(args.embedding_module)(kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2,
                                                 tri_num, ent_num, rel_num, rel1_list, rel2_list, rel_ht_dict, args)
    gcn_model.train(args.batch_size, max_epochs=args.max_epoch, start_valid=args.start_valid, eval_freq=args.eval_freq)
    gcn_model.test()
