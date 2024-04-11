import os
import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue


def getFeature(args):
    f_t = f"./{args.dataset}/textEmbedding/"
    f_l = f"./{args.dataset}/LocalImgEmb/"
    f_g = f"./{args.dataset}/GlobalImgEmb/"
    if os.listdir(f_t):
        text_list = [torch.load(f_t+f"{i}.json",map_location=torch.device('cpu')) for i in range(1)]
        text_list.insert(0,torch.zeros([1,768]))
        textfeature = torch.cat(text_list,dim=0)
    else:
        text_list = None
    if os.listdir(f_l):
        local_list = [torch.load(f_l+f"{i}.json",map_location=torch.device('cpu')) for i in range(7)]
        local_list.insert(0,torch.zeros([1,9,768]))
        localfeature = torch.cat(local_list,dim=0)
    else:
        local_list = None
    if os.listdir(f_g):
        global_list = [torch.load(f_g+f"{i}.json",map_location=torch.device('cpu')) for i in range(1)]
        global_list.insert(0,torch.zeros([1,768]))
        globalfeature = torch.cat(global_list,dim=0)
    else:
        global_list = None
    print(textfeature.shape,localfeature.shape,globalfeature.shape)
    return textfeature,localfeature,globalfeature


# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def getbatchnum(total,batch):
    if total%batch==0:
        return int(total/batch)
    else:
        return int(total/batch+1)

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen,2], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        # next item
        nxt = user_train[user][-1][0]
        idx = maxlen - 1
        ts = set(user_train[user][0])
        for i,t in reversed(user_train[user][:-1]):
            seq[idx,0] = i
            seq[idx,1] = t
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break
        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i, t = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        t = int(t)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append([i,t])

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0


    users = range(1, usernum + 1)
    evaluate_user = []
    evaluate_seq = []
    evaluate_item = []
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen,2], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        # 为了得到最近的交互
        for i,t in reversed(train[u]):
            seq[idx,0] = i
            seq[idx,1] = t
            idx -= 1
            if idx == -1: break
        item_elements = [item[0] for item in train[u]]
        rated = set(item_elements)
        rated.add(0)
        item_idx = [test[u][0][0]]
        other = list(set(range(1,itemnum+1))-set(item_idx))
        item_idx = item_idx + other
        '''
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        '''
        evaluate_user.append([u])
        evaluate_seq.append(seq)
        evaluate_item.append(item_idx)
        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    evaluate_user = np.array(evaluate_user)
    evaluate_seq = np.array(evaluate_seq)
    evaluate_item = np.array(evaluate_item)
    
    batchnum = getbatchnum(valid_user,args.batch_size)
    for b in range(batchnum):
        start = b*args.batch_size
        if start+args.batch_size>valid_user:
            end = int(valid_user)
        else:
            end = start+args.batch_size

        predictions = -model.predict(evaluate_user[start:end],evaluate_seq[start:end],evaluate_item[start:end])
        rank = predictions.argsort().argsort()[:,0]
        for r in rank:
            if r < 10:
                NDCG += 1 / np.log2(r.cpu() + 2)
                HT += 1
    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0

    users = range(1, usernum + 1)
    evaluate_user = []
    evaluate_seq = []
    evaluate_item = []
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue
        seq = np.zeros([args.maxlen,2], dtype=np.int32)
        idx = args.maxlen - 1
        # 两个函数的区别在于 上一个函数的seq包括了valid
        for i,t in reversed(train[u]):
            seq[idx,0] = i
            seq[idx,1] = t
            idx -= 1
            if idx == -1: break
        item_elements = [item[0] for item in train[u]]
        rated = set(item_elements)
        rated.add(0)
        item_idx = [valid[u][0][0]]
        item_idx = [test[u][0][0]]
        other = list(set(range(1,itemnum+1))-set(item_idx))
        item_idx = item_idx + other
        '''
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        '''
        evaluate_user.append([u])
        evaluate_seq.append(seq)
        evaluate_item.append(item_idx)
        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    evaluate_user = np.array(evaluate_user)
    evaluate_seq = np.array(evaluate_seq)
    evaluate_item = np.array(evaluate_item)

    batchnum = getbatchnum(valid_user,args.batch_size)
    for b in range(batchnum):
        start = b*args.batch_size
        if start+args.batch_size>valid_user:
            end = int(valid_user)
        else:
            end = start+args.batch_size

        predictions = -model.predict(evaluate_user[start:end],evaluate_seq[start:end],evaluate_item[start:end])
        rank = predictions.argsort().argsort()[:,0]
        for r in rank:
            if r < 10:
                NDCG += 1 / np.log2(r.cpu() + 2)
                HT += 1
    print(NDCG,HT,valid_user)
    return NDCG / valid_user, HT / valid_user


def loss_sample(seq):
    seq = torch.tensor(seq)
    pos_set = set(seq.flatten().numpy())
    s = len(pos_set)
    pos_set.discard(0)
    neg_sample = []
    for p in seq:
        for i in p:
            if i.item()==0:
                pad = torch.full([s-1],0)
                neg_sample.append(pad)
            else:
                copy_set = pos_set.copy()
                copy_set.discard(i)
                neg_sample.append(torch.tensor(list(copy_set)))
    neg = torch.cat(neg_sample)
    neg = neg.reshape(seq.shape[0],seq.shape[1],-1)
    return neg


