import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from TIE import TIE
from datetime import datetime


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class TMMSRec(torch.nn.Module):
    def __init__(self, item2img, textfeature, localfeature, globalfeature, user_num, item_num, args):
        super(TMMSRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.length = args.maxlen
        self.hidden = args.hidden_units
        self.item2img = item2img
        self.textfeature = textfeature
        self.localfeature = localfeature
        self.globalfeature = globalfeature
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.pool1 = nn.AvgPool1d(9)
        self.rotape = TIE(self.hidden,self.dev)
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.sig = torch.nn.Sigmoid()
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    def getItemEmbedding_From_V_T(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        self.textfaeture = torch.tensor(self.textfeature)
        textfaeture_table = self.textfaeture[torch.LongTensor(log_seqs)].to(self.dev)
        imagefeature_table = self.globalfeature[torch.LongTensor(log_seqs)].to(self.dev)
        return seqs,textfaeture_table,imagefeature_table

    def convert_unix_to_datetime(self, unix_timestamp):
        dt_object = datetime.utcfromtimestamp(unix_timestamp)
        year = dt_object.year
        day = dt_object.day  # Extract day
        month = dt_object.month
        hour_minute = dt_object.hour  # Extract hour and minute

        return day, month, year

    def getposition(self,seqs,log_seqs):
        time_seqs = log_seqs[:,:,1]
        item_seqs = log_seqs[:,:,0]
        day_table = torch.zeros([log_seqs.shape[0],log_seqs.shape[1]])
        month_table = torch.zeros([log_seqs.shape[0],log_seqs.shape[1]])
        hour_table = torch.zeros([log_seqs.shape[0],log_seqs.shape[1]])

        for p,time in enumerate(time_seqs):
            for q,t in enumerate(time):
                if t!= 0:
                    d,m,y = self.convert_unix_to_datetime(t)
                    day_table[p,q] ,hour_table[p,q], month_table[p,q] = d, y, m
        return seqs+self.rotape(seqs,hour_table.to(self.dev),month_table.to(self.dev),day_table.to(self.dev))

    def log2feats_ROTAPE(self, log_seqs, seqs):
        time_seqs = log_seqs[:,:,1]
        item_seqs = log_seqs[:,:,0]
        seqs = self.emb_dropout(seqs)
        seqs = self.getposition(seqs,log_seqs,method='NP')
        timeline_mask = torch.BoolTensor(item_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim
        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats
 

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training     
        item_seqs = log_seqs[:,:,0]
        idfeature,textfeature,visualfeature = self.getItemEmbedding_From_V_T(item_seqs)

        id_user = self.log2feats_ROTAPE(log_seqs,idfeature) # user_ids hasn't been used yet
        t_user = self.log2feats_ROTAPE(log_seqs,textfeature)
        v_user = self.log2feats_ROTAPE(log_seqs,visualfeature)
        id_pos,t_pos,v_pos = self.getItemEmbedding_From_V_T(pos_seqs)
        id_neg,t_neg,v_neg = self.getItemEmbedding_From_V_T(neg_seqs)
        pos_logits = (id_user * id_pos).sum(dim=-1)  + (v_user * v_pos).sum(dim=-1) + (t_user * t_pos).sum(dim=-1)
        neg_logits = (id_user * id_neg).sum(dim=-1)  + (v_user * v_neg).sum(dim=-1) + (t_user * t_neg).sum(dim=-1)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def getallembedding_from_V_T(self,log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))

        self.textfaeture = torch.tensor(self.textfeature)
        textfaeture_table = self.textfaeture[torch.LongTensor(log_seqs)].to(self.dev)
        imagefeature_table = self.globalfeature[torch.LongTensor(log_seqs)].to(self.dev)
        
        return seqs,textfaeture_table,imagefeature_table

    def fusion(self, idfeature, textfeature, visualfeature):
        return self.emb_dropout(idfeature+textfeature+visualfeature)


    def predict(self, user_ids, log_seqs, item_indices): # for inference
        item_seqs = log_seqs[:,:,0]
        idfeature,textfeature,visualfeature = self.getItemEmbedding_From_V_T(item_seqs)
        id_user = self.log2feats_ROTAPE(log_seqs,idfeature) # user_ids hasn't been used yet
        t_user = self.log2feats_ROTAPE(log_seqs,textfeature)
        v_user = self.log2feats_ROTAPE(log_seqs,visualfeature)

        final_id = id_user[:, -1, :] # only use last QKV classifier, a waste
        final_t = t_user[:,-1,:]
        final_v = v_user[:,-1,:]

        id_item,t_item,v_item = self.getallembedding_from_V_T(item_indices)
        logits = id_item.matmul(final_id.unsqueeze(-1)).squeeze(-1)  + v_item.matmul(final_v.unsqueeze(-1)).squeeze(-1) + t_item.matmul(final_t.unsqueeze(-1)).squeeze(-1)

        return logits # preds # (U, I)


        