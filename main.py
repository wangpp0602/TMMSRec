import os
import time
import torch
import argparse
import json
from model import TMMSRec
from utils import *
from torch.optim.lr_scheduler import ExponentialLR
def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="Luxury_Beauty")
parser.add_argument('--train_dir', default="default")
parser.add_argument('--batch_size', default=30, type=int)
parser.add_argument('--lr', default=0.001,type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=768, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--device', default="cuda:0", type=str)
args = parser.parse_args()

if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()


if __name__ == '__main__':
    # global dataset
    # 倒数第二个做验证集，倒数第一个做测试集
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    textfeature,localfeature,globalfeature = getFeature(args)
    with open(f"{args.dataset}/item2img.json") as f:
        item2img = json.load(f)
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

    # 为训练集采样负样本
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = TMMSRec(item2img, textfeature, localfeature, globalfeature, usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass 
    
    model.train() 
    
    epoch = 0
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    gamma = 0.9  # 指数衰减率
    scheduler = ExponentialLR(adam_optimizer, gamma=gamma)
    T = 0.0
    t0 = time.time()
    best_hr_valid = 0.0
    best_ndcg_valid = 0.0
    best_hr_test = 0.0
    best_ndcg_test = 0.0
    num = 0
    flag = True
    while flag:
        epoch += 1
        scheduler.step()
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            #mml = getModalLoss(model.textfaeture,model.sumfeature,seq[:,:,0])
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            loss.backward()
            adam_optimizer.step()
    
        model.eval()
        t1 = time.time() - t0
        T += t1
        print('Evaluating', end='')
        t_valid = evaluate_valid(model, dataset, args)
        print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1]))
        if t_valid[0] > best_ndcg_valid:
            best_ndcg_valid = t_valid[0]
            best_hr_valid = t_valid[1]
            num = 0
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model, os.path.join(folder, fname))
        else:
            num += 1
        f.write(str(t_valid) +'\n')
        f.flush()
        t0 = time.time()
        model.train()
        if num == 20:
            print(f"--------------the best result--------------")
            model = torch.load(os.path.join(folder, fname))
            t_test = evaluate(model, dataset, args)
            best_hr_test = t_test[1]
            best_ndcg_test = t_test[0]
            print("valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)" 
                      % (best_ndcg_valid, best_hr_valid, best_ndcg_test, best_hr_test))
            flag = False
        torch.cuda.empty_cache()
    f.close()
    sampler.close()
    print("Done")
