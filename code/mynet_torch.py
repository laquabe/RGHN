import random
import time
import gc
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from model import MyNet_Torch

from align.test import greedy_alignment
from align.sample import generate_neighbours
from align.preprocess import enhance_triples, remove_unlinked_triples

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

class TrainerTorch:
    def __init__(self, kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, tri_num, ent_num, rel_num, rel1_list, rel2_list, rel_ht_dict, args):
        self.kg1 = kg1
        self.kg2 = kg2
        self.sup_ent1 = sup_ent1
        self.sup_ent2 = sup_ent2
        self.ref_ent1 = ref_ent1
        self.ref_ent2 = ref_ent2
        self.tri_num = tri_num
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.rel1_list = rel1_list
        self.rel2_list = rel2_list

        self.rel_ht_dict = rel_ht_dict
        self.rel_win_size = args.batch_size // len(rel_ht_dict)
        if self.rel_win_size <= 1:
            self.rel_win_size = args.min_rel_win

        self.learning_rate = args.learning_rate

        self.neg_multi = args.neg_multi
        self.neg_margin = args.neg_margin
        self.neg_param = args.neg_param
        self.rel_param = args.rel_param
        self.rel_align_param = args.rel_align_param
        self.truncated_epsilon = args.truncated_epsilon
        self.threshold = args.threshold

        self.eval_metric = args.eval_metric
        self.hits_k = args.hits_k
        self.eval_threads_num = args.eval_threads_num
        self.eval_normalize = args.eval_normalize
        self.eval_csls = args.eval_csls

        self.linked_ents = set(sup_ent1 + sup_ent2 + ref_ent1 + ref_ent2)
        enhanced_triples1, enhanced_triples2 = enhance_triples(self.kg1, self.kg2, self.sup_ent1, self.sup_ent2)
        triples = self.kg1.triple_list + self.kg2.triple_list + list(enhanced_triples1) + list(enhanced_triples2)
        triples = remove_unlinked_triples(triples, self.linked_ents)

        self.model = MyNet_Torch(kg1, kg2, sup_ent1, sup_ent2, triples, ent_num, rel_num, rel_ht_dict, args)
        self.model = self.model.cuda()

        self.pos_link_batch = None
        self.neg_link_batch = None

        sup_ent1 = np.array(self.sup_ent1).reshape((len(self.sup_ent1), 1))
        sup_ent2 = np.array(self.sup_ent2).reshape((len(self.sup_ent2), 1))
        weight = np.ones((len(self.sup_ent1), 1), dtype=np.float)
        self.sup_links = np.hstack((sup_ent1, sup_ent2, weight))
        self.sup_links_set = set()
        for i in range(len(sup_ent1)):
            self.sup_links_set.add((self.sup_ent1[i], self.sup_ent2[i]))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model_name = args.model_name
        self.save_path = './checkpoints/best_model_{}.ckpt'.format(self.model_name)
        print(self.save_path)
        self.best_epoch = 0
        self.best_valid = 0

    @staticmethod
    def early_stop(flag1, flag2, flag):
        if flag < flag2 < flag1:
            return flag2, flag, True
        else:
            return flag2, flag, False

    def eval_embeds(self):
        self.model.eval()
        with torch.no_grad():
            output_embeds, _, _, _ = self.model.forward()
            return output_embeds

    def load_model(self):
        self.model.load_state_dict(torch.load(self.save_path))

    def valid(self):
        output_embeds = self.eval_embeds()
        embeds1 = output_embeds[self.ref_ent1, :]
        embeds1 = embeds1.cpu().detach().numpy()
        embeds2 = output_embeds[self.ref_ent2, :]
        embeds2 = embeds2.cpu().detach().numpy()
        alignment_rest, hits1_12, mr_12, mrr_12 = greedy_alignment(embeds1, embeds2, self.hits_k, self.eval_threads_num,
                                                                   self.eval_metric, False, 0, False)
        return hits1_12

    def test(self):
        print('best epoch {}:'.format(self.best_epoch))
        self.model.load_state_dict(torch.load(self.save_path))
        output_embeds = self.eval_embeds()
        embeds1 = output_embeds[self.ref_ent1, :]
        embeds1 = embeds1.cpu().detach().numpy()
        embeds2 = output_embeds[self.ref_ent2, :]
        embeds2 = embeds2.cpu().detach().numpy()
        alignment_rest, _, _, _ = greedy_alignment(embeds1, embeds2, self.hits_k, self.eval_threads_num,
                                                   self.eval_metric, False, 0, True)
        alignment_rest, _, _, _ = greedy_alignment(embeds1, embeds2, self.hits_k, self.eval_threads_num,
                                                   self.eval_metric, False, self.eval_csls, True)
        import pickle
        with open('result.pkl','wb') as f:
            pickle.dump(alignment_rest, f)

    def generate_input_batch(self, batch_size, neighbors1=None, neighbors2=None):
        if batch_size > len(self.sup_ent1):
            batch_size = len(self.sup_ent1)
        index = np.random.choice(len(self.sup_ent1), batch_size)
        pos_links = self.sup_links[index,]
        neg_links = list()
        if neighbors1 is None:
            neg_ent1 = list()
            neg_ent2 = list()
            for i in range(self.neg_multi):
                neg_ent1.extend(random.sample(self.sup_ent1 + self.ref_ent1, batch_size))
                neg_ent2.extend(random.sample(self.sup_ent2 + self.ref_ent2, batch_size))
            neg_links.extend([(neg_ent1[i], neg_ent2[i]) for i in range(len(neg_ent1))])
        else:
            for i in range(batch_size):
                e1 = pos_links[i, 0]
                candidates = random.sample(neighbors1.get(e1), self.neg_multi)
                neg_links.extend([(e1, candidate) for candidate in candidates])
                e2 = pos_links[i, 1]
                candidates = random.sample(neighbors2.get(e2), self.neg_multi)
                neg_links.extend([(candidate, e2) for candidate in candidates])
        neg_links = set(neg_links) - self.sup_links_set
        neg_links = np.array(list(neg_links))
        return pos_links, neg_links

    def generate_rel_batch(self):
        hs, rs, ts = list(), list(), list()
        for r, hts in self.rel_ht_dict.items():
            hts_batch = [random.choice(hts) for _ in range(self.rel_win_size)]
            for h, t in hts_batch:
                hs.append(h)
                ts.append(t)
                rs.append(r)
        return hs, rs, ts

    def find_neighbors(self):
        if self.truncated_epsilon <= 0.0:
            return None, None
        start = time.time()
        output_embeds = self.eval_embeds()
        ents1 = self.sup_ent1 + self.ref_ent1
        ents2 = self.sup_ent2 + self.ref_ent2

        embeds1 = output_embeds[ents1, :]
        embeds2 = output_embeds[ents2, :]
        embeds1 = embeds1.cpu().detach().numpy()
        embeds2 = embeds2.cpu().detach().numpy()
        num = int((1-self.truncated_epsilon) * len(ents1))
        print("neighbors num", num)
        neighbors1 = generate_neighbours(embeds1, ents1, embeds2, ents2, num, threads_num=self.eval_threads_num)
        neighbors2 = generate_neighbours(embeds2, ents2, embeds1, ents1, num, threads_num=self.eval_threads_num)
        print('finding neighbors for sampling costs time: {:.4f}s'.format(time.time() - start))
        del embeds1, embeds2
        gc.collect()
        return neighbors1, neighbors2

    def generate_rel_label(self, rel_hent_index, rel_tent_index, mask, rel1_list, rel2_list, ent_embeds, threshold=0.5):
        ent_embeds = ent_embeds.detach()
        rel_hent_index = rel_hent_index.detach()
        rel_tent_index = rel_tent_index.detach()
        rel_head_embeds = ent_embeds[rel_hent_index, :] #[R, max_N, dim]
        rel_tail_embeds = ent_embeds[rel_tent_index, :] #[R, max_N, dim]
        mask = torch.unsqueeze(mask, dim=2) #[R, max_N, 1]
        rel_embeds = torch.cat([rel_head_embeds, rel_tail_embeds], dim=-1)  #[R, max_N, dim * 2]
        rel_embeds = torch.mul(rel_embeds, mask)    #[R, max_N, dim *2]
        rel_embeds = torch.sum(rel_embeds, dim=1)   #[R, dim * 2]
        rel_embeds = F.normalize(rel_embeds, dim=1)
        
        rel1 = rel_embeds[rel1_list, :] #[R1, dim]
        rel2 = rel_embeds[rel2_list, :] #[R2, dim]
        score = torch.mm(rel1, rel2.transpose(0,1))
        label = score.gt(threshold)
        label = label.detach().int()
        
        return label


    def train(self, batch_size, max_epochs=1000, start_valid=10, eval_freq=10):
        flag1 = 0
        flag2 = 0
        steps = len(self.sup_ent2) // batch_size
        neighbors1, neighbors2 = None, None
        if steps == 0:
            steps = 1
        rel_label = None
        for epoch in range(1, max_epochs + 1):
            self.model.train()
            start = time.time()
            epoch_loss = 0.0

            for _ in range(steps):
                self.pos_link_batch, self.neg_link_batch = self.generate_input_batch(batch_size,
                                                                                     neighbors1=neighbors1,
                                                                                     neighbors2=neighbors2)
                self.pos_link_batch = torch.tensor(self.pos_link_batch, dtype=torch.long)
                self.pos_link_batch = self.pos_link_batch.cuda()
                self.neg_link_batch = torch.tensor(self.neg_link_batch, dtype=torch.long)
                self.neg_link_batch = self.neg_link_batch.cuda()

                self.model.train()
                output_embeds, rel_embed, last_output_embeed, last_rel_embed = self.model.forward()
                batch_loss = self.model.compute_loss(self.pos_link_batch, self.neg_link_batch, output_embeds)
                if self.rel_param > 0.0:
                    hs, _, ts = self.generate_rel_batch()
                    hs = torch.tensor(hs, dtype=torch.long)
                    hs = hs.cuda()
                    ts = torch.tensor(ts, dtype=torch.long)
                    ts = ts.cuda()
                    rel_loss = self.model.compute_rel_loss(hs, ts, output_embeds)
                    batch_loss += rel_loss
                if self.rel_align_param > 0.0:
                    if rel_label != None:
                        rel_align_loss = self.model.compute_rel_align_unsupervise(self.rel1_list, self.rel2_list, last_rel_embed, rel_label)
                        batch_loss += rel_align_loss * self.rel_align_param

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                epoch_loss += batch_loss

            print('epoch {}, loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

            if epoch % eval_freq == 0 and epoch >= start_valid:
                flag = self.valid()
                if flag > self.best_valid:
                    torch.save(self.model.state_dict(), self.save_path)
                    print('save model{}'.format(self.save_path))
                    self.best_epoch = epoch
                    self.best_valid = flag

                flag1, flag2, is_stop = self.early_stop(flag1, flag2, flag)
                if is_stop:
                     print("\n == training stop == \n")
                     break
                neighbors1, neighbors2 = self.find_neighbors()
                self.rel_head, self.rel_tail, self.mask = self.model.get_relation_entities_tensor()
                self.rel_head = self.rel_head.cuda()
                self.rel_tail = self.rel_tail.cuda()
                self.mask = self.mask.cuda()
                rel_label = self.generate_rel_label(self.rel_head, self.rel_tail, self.mask, self.rel1_list, self.rel2_list, last_output_embeed, self.threshold)
    
    def visual(self, vis_relation=False):
        if vis_relation == False:
            array_path = './tsne/{}'.format(self.model_name)
            if os.path.exists(array_path + '.npy'):
                ent_tsne = np.load(array_path + '.npy')
            else:
                self.model.load_state_dict(torch.load(self.save_path))
                ent_embedding = self.model.ent_embedding.weight.data.cpu().numpy()
                tsne = TSNE()
                ent_tsne = tsne.fit_transform(ent_embedding)
                np.save(array_path, ent_tsne)
            ent_1 = ent_tsne[:15000, :]
            ent_2 = ent_tsne[15000:, :]
        else:
            array_path = './tsne/{}_relation'.format(self.model_name)
            if os.path.exists(array_path + '.npy'):
                ent_tsne = np.load(array_path + '.npy')
            else:
                self.model.load_state_dict(torch.load(self.save_path))
                ent_embedding = self.model.rel_embedding.weight.data.cpu().numpy()
                tsne = TSNE()
                ent_tsne = tsne.fit_transform(ent_embedding)
                np.save(array_path, ent_tsne)
            ent_1 = ent_tsne[self.rel1_list, :]
            ent_2 = ent_tsne[self.rel2_list, :]

        save_path = './pic/{}_rel.png'.format(self.model_name)
        plt.clf()
        plt.scatter(ent_1[:,0], ent_1[:,1], c='#F78E53')
        plt.scatter(ent_2[:,0], ent_2[:,1], c='#8065A2')
        plt.savefig(save_path)