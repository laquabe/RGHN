import torch
from torch import nn
import torch.nn.functional as F
from layer_torch import RelationGatedConv
import random
from torch_geometric.nn.conv import MessagePassing, GCNConv, GATv2Conv

class MyNet_Torch(torch.nn.Module):
    def __init__(self, kg1, kg2, sup_ent1, sup_ent2, triples, ent_num, rel_num, rel_ht_dict, args):
        super(MyNet_Torch, self).__init__()
        self.kg1 = kg1
        self.kg2 = kg2
        self.sup_ent1 = sup_ent1
        self.sup_ent2 = sup_ent2
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.gcn_name = args.gcn

        self.neg_margin = args.neg_margin
        self.neg_param = args.neg_param
        self.rel_param = args.rel_param
        self.inverse_relation = args.inverse_relation
        self.num_bias = 5

        self.rel_ht_dict = rel_ht_dict
        self.rel_win_size = args.batch_size // len(rel_ht_dict)
        if self.rel_win_size <= 1:
            self.rel_win_size = args.min_rel_win

        self.layer_dims = args.layer_dims
        self.layer_num = len(args.layer_dims) - 1
        self.num_features_nonzero = args.num_features_nonzero
        self.rel_dim = self.layer_dims[0]
        self.attention_dim = 128
        self.attention_head = 1

        self.eval_metric = args.eval_metric
        self.hits_k = args.hits_k
        self.eval_threads_num = args.eval_threads_num
        self.eval_normalize = args.eval_normalize
        self.eval_csls = args.eval_csls

        self.sup_links_set = set()
        self.sup_links_dict = dict()
        for i in range(len(sup_ent1)):
            self.sup_links_set.add((self.sup_ent1[i], self.sup_ent2[i]))
            self.sup_links_dict[self.sup_ent1[i]] = self.sup_ent2[i]
            self.sup_links_dict[self.sup_ent2[i]] = self.sup_ent1[i]

        self.edge_index, self.edge_attr = self.get_edge_index(triples, self.rel_num)
        self.cross_edge_index, _= self.get_edge_index(triples, self.rel_num, cross=True)
        self.cross_edge_index = self.cross_edge_index.cuda()
        self.edge_attr = self.edge_attr.cuda()
        self.edge_index = self.edge_index.cuda()

        self.ent_embedding = nn.Embedding(self.ent_num, self.layer_dims[0])
        if self.inverse_relation == 'indep':
            self.rel_embedding = nn.Embedding(self.rel_num*2, self.layer_dims[0])
        else:
            if self.gcn_name == 'RGCN':
                self.rel_embedding = nn.Embedding(self.rel_num, self.num_bias)
            else:
                self.rel_embedding = nn.Embedding(self.rel_num, self.layer_dims[0])
            

        nn.init.xavier_normal_(self.ent_embedding.weight)
        nn.init.xavier_normal_(self.rel_embedding.weight)
        
        if self.inverse_relation == 'lin':
            if self.gcn_name == 'RGCN':
                self.rel_inv_W = nn.Linear(self.num_bias, self.num_bias)
            else:
                self.rel_inv_W = nn.Linear(self.layer_dims[0], self.layer_dims[0])

        self.convs = []
        self.bns = []
        self.rel_lins = []
        self.ent_lins = []
        self.s_convs = []
        self.ent_lins_cross = []
        final_ent_dim = self.layer_dims[0]
        for i in range(len(self.layer_dims) - 1):
            if self.gcn_name == 'RGC':
                print('now_gcn_layer:RGC')
                conv = RelationGatedConv(in_channels=self.layer_dims[i], out_channels=self.layer_dims[i + 1])
                s_conv = RelationGatedConv(in_channels=self.layer_dims[i], out_channels=self.layer_dims[i+1])
            
            self.convs.append(conv.cuda())
            self.s_convs.append(s_conv.cuda())
            bn = nn.BatchNorm1d(self.layer_dims[i])
            self.bns.append(bn.cuda())
            if self.gcn_name == 'RGCN':
                rel_lin = nn.Linear(self.num_bias, self.num_bias)
            else:
                rel_lin = nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])
            self.rel_lins.append(rel_lin.cuda())
            ent_lin = nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])
            self.ent_lins.append(ent_lin.cuda())
            s_lin = nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])
            self.ent_lins_cross.append(s_lin.cuda())
            final_ent_dim += self.layer_dims[i + 1]

        self.att_layer = nn.MultiheadAttention(self.layer_dims[-1] * 2, self.attention_head, batch_first=True, kdim=self.layer_dims[-1] , vdim=self.layer_dims[-1])

    def forward(self):
        in_ent_embed = self.ent_embedding.weight
        rel_embed = self.rel_embedding.weight
        if self.inverse_relation == 'minus':
            inv_rel_embed = -rel_embed
            rel_embed_all = torch.cat([rel_embed,inv_rel_embed], dim=0)
            edge_attr = rel_embed_all[self.edge_attr,:]
        elif self.inverse_relation == 'lin':
            inv_rel_embed = self.rel_inv_W(rel_embed)
            rel_embed_all = torch.cat([rel_embed,inv_rel_embed], dim=0)
            edge_attr = rel_embed_all[self.edge_attr,:]
        elif self.inverse_relation == 'indep':
            edge_attr = self.rel_embedding(self.edge_attr)

        out_ent_embed_list = [F.normalize(in_ent_embed, dim=1)]
        out_ent_embed = in_ent_embed
        s_out_ent_embed = in_ent_embed

        rel_embed_list = [F.normalize(rel_embed_all, dim=1)]

        for i in range(len(self.layer_dims) - 1):
            out_ent_embed = self.bns[i](out_ent_embed)
            s_out_ent_embed = self.bns[i](s_out_ent_embed)

            rel_embed_all = self.rel_lins[i](rel_embed_all)
            edge_attr = rel_embed_all[self.edge_attr,:]

            layout_in_ent_embed = self.ent_lins[i](out_ent_embed)
            s_layout_in_ent_embed = self.ent_lins_cross[i](s_out_ent_embed)
            if self.gcn_name == 'GCN' or self.gcn_name == 'GAT':
                layout_ent_embed = self.convs[i](x=layout_in_ent_embed, edge_index=self.edge_index)
                s_layout_ent_embed = self.s_convs[i](x=s_layout_in_ent_embed, edge_index=self.cross_edge_index)
            else:
                layout_ent_embed = self.convs[i](x=layout_in_ent_embed, edge_index=self.edge_index, edge_attr=edge_attr)
                s_layout_ent_embed = self.s_convs[i](x=s_layout_in_ent_embed, edge_index=self.cross_edge_index, edge_attr=edge_attr)
            
            out_ent_embed = s_layout_ent_embed
            s_out_ent_embed = layout_ent_embed
            out_ent_embed_list.append(F.normalize(out_ent_embed, dim=1))
            out_ent_embed_list.append(F.normalize(s_out_ent_embed, dim=1))
            rel_embed_list.append(F.normalize(rel_embed_all, dim=1))

        last_ent_embed = out_ent_embed_list[-2:]
        last_ent_embed = torch.cat(last_ent_embed, dim=1)
        last_ent_embed = F.normalize(last_ent_embed, dim=1)
        last_rel_embed = rel_embed_list[-1]

        final_embed = torch.cat(out_ent_embed_list, dim=1)
        final_embed = F.normalize(final_embed, dim=1)
        rel_embed_final = torch.cat(rel_embed_list, dim=1)
        rel_embed_final = F.normalize(rel_embed_final, dim=1)

        return final_embed, rel_embed_final, last_ent_embed, last_rel_embed

    def get_edge_index(self, triples, rel_num, equal=False, cross=False):
        h_list = []
        t_list = []
        r_list = []
        for (h, r, t) in triples:
            if cross:
                h_list.append(self.sup_links_dict.get(h, h))
                h_list.append(self.sup_links_dict.get(t, t))
                t_list.append(self.sup_links_dict.get(t, t))
                t_list.append(self.sup_links_dict.get(h, h))
                r_list.append(r)
                r_list.append(r + rel_num)

            else:
                h_list.append(h)
                h_list.append(t)
                t_list.append(t)
                t_list.append(h)
                r_list.append(r)
                r_list.append(r + rel_num)

        if equal:
            for (e1, e2) in self.sup_links_set:
                h_list.append(e1)
                h_list.append(e2)
                t_list.append(e2)
                t_list.append(e1)
                r_list.append(rel_num * 2)
                r_list.append(rel_num * 2)

        return torch.tensor([h_list, t_list], dtype=torch.long), torch.tensor(r_list, dtype=torch.long)
    
    def get_entity_relations_tensor(self, triples, rel_num, sum=True, inverse=False):
        e_rel_dict = {}
        for (h, r, t) in triples:
            lst = e_rel_dict.get(h, [])
            lst.append(r)
            e_rel_dict[h] = lst
            lst = e_rel_dict.get(t, [])
            lst.append(r + rel_num)
            e_rel_dict[t] = lst
        
        max_len = 0
        for (e, lst) in e_rel_dict.items():
            if len(lst) > max_len:
                max_len = len(lst)

        e_rel_tensor = torch.zeros(self.ent_num, max_len, dtype=torch.long)
        mask_tensor = torch.zeros(self.ent_num, max_len, dtype=torch.float)
        for (e, lst) in e_rel_dict.items():
            if sum == True:
                mask = [1 / len(lst)] * len(lst) + [0] * (max_len - len(lst))
            elif inverse == False:
                mask = [1] * len(lst) + [0] * (max_len - len(lst))
            elif inverse == True:
                mask = [0] * len(lst) + [1] * (max_len - len(lst))

            lst_ext = lst + [0]*(max_len - len(lst))
            e_rel_tensor[e] = torch.tensor(lst_ext, dtype=torch.long)
            mask_tensor[e] = torch.tensor(mask, dtype=torch.float)

        return e_rel_tensor, mask_tensor

    def get_relation_entities_tensor(self, sum = True, max_len = 100):
        r_hent_tensor = torch.zeros(self.rel_num, max_len, dtype=torch.long) #[R, N_max]
        r_tent_tensor = torch.zeros(self.rel_num, max_len, dtype=torch.long)
        mask_tensor = torch.zeros(self.rel_num, max_len, dtype=torch.float)
        for r, hts in self.rel_ht_dict.items():
            if len(hts) > max_len:
                hts = random.sample(hts, max_len)
            hs = []
            ts = []
            for (h,t) in hts:
                hs.append(h)
                ts.append(t)
            if sum == True:
                mask = [1 / len(hs)] * len(hs) + [0] * (max_len - len(hs))
            else:
                mask = [1] * len(hs) + [0] * (max_len - len(hs))
            hs = hs + [0]*(max_len - len(hs))
            ts = ts + [0]*(max_len - len(ts))

            r_hent_tensor[r] = torch.tensor(hs, dtype=torch.long)
            r_tent_tensor[r] = torch.tensor(ts, dtype=torch.long)
            mask_tensor[r] = torch.tensor(mask, dtype=torch.float)

        return r_hent_tensor, r_tent_tensor, mask_tensor

    def compute_loss(self, pos_links, neg_links, output_embeds):
        index1 = pos_links[:, 0]    
        index2 = pos_links[:, 1]   
        neg_index1 = neg_links[:, 0]
        neg_index2 = neg_links[:, 1]

        embeds1 = output_embeds[index1, :]
        embeds2 = output_embeds[index2, :]
        pos_loss = torch.sum(torch.sum(torch.square(embeds1 - embeds2), dim=1))

        embeds1 = output_embeds[neg_index1, :]
        embeds2 = output_embeds[neg_index2, :]
        neg_distance = torch.sum(torch.square(embeds1 - embeds2), dim=1)
        neg_loss = torch.sum(F.relu(self.neg_margin - neg_distance))

        return pos_loss + self.neg_param * neg_loss

    def compute_rel_loss(self, hs, ts, output_embeds):
        h_embeds = output_embeds[hs, :]
        t_embeds = output_embeds[ts, :]
        r_temp_embeds = torch.reshape(h_embeds - t_embeds, [-1, self.rel_win_size, output_embeds.shape[-1]])
        r_temp_embeds = torch.mean(r_temp_embeds, dim=1, keepdim=True)
        r_embeds = torch.tile(r_temp_embeds, [1, self.rel_win_size, 1])
        r_embeds = torch.reshape(r_embeds, [-1, output_embeds.shape[-1]])
        r_embeds = F.normalize(r_embeds, dim=1)
        return torch.sum(torch.sum(torch.square(h_embeds - t_embeds - r_embeds), dim=1)) * self.rel_param

    def compute_rel_align_loss(self, pos_links, rel_embed, ent_embed, attention=False):
        index1 = pos_links[:, 0]    # [B]
        index2 = pos_links[:, 1]    # [B]

        rel_index1 = self.ent_rel_tensor[index1, :] #[B, max_N]
        rel_index2 = self.ent_rel_tensor[index2, :] 
        rel_mask1 = self.ent_rel_mask[index1, :] #[B, max_N]
        rel_mask2 = self.ent_rel_mask[index2, :] 

        rel1 = rel_embed[rel_index1, :] #[B, max_N, dim]
        rel2 = rel_embed[rel_index2, :] #[B, max_N, dim]

        # sum, maybe attention
        if attention == False:
            rel1 = torch.mul(rel1, rel_mask1)   #[B, max_N, dim]
            rel2 = torch.mul(rel2, rel_mask2)
        if attention == True:
            embed1 = ent_embed[index1, :]   #[B, dim]
            embed2 = ent_embed[index2, :]   
            embed1 = torch.unsqueeze(embed1, 1) #[B, 1, dim]
            embed2 = torch.unsqueeze(embed2, 1)
            rel1, _ = self.att_layer(embed1, rel1, rel1, need_weights=False, key_padding_mask=rel_mask1)   #[B, 1, dim]
            rel2, _ = self.att_layer(embed2, rel2, rel2, need_weights=False, key_padding_mask=rel_mask2)
            rel1 = torch.squeeze(rel1)
            rel2 = torch.squeeze(rel2)

        pos_loss = torch.sum(torch.sum(torch.square(rel1 - rel2), dim=1))

        return pos_loss


    def compute_rel_align_unsupervise(self, rel1_list, rel2_list, rel_embeds, label):
        rel1 = rel_embeds[rel1_list, :] #[R1, dim]
        rel2 = rel_embeds[rel2_list, :] #[R2, dim]
        score = torch.mm(rel1, rel2.transpose(0,1))

        loss_func = nn.MultiLabelSoftMarginLoss()
        loss = loss_func(score, label)
        
        return loss
