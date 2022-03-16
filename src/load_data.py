from typing import DefaultDict
import numpy as np 
from collections import defaultdict
import scipy.sparse as sp
from time import time
import random 
import copy
from scipy.sparse import vstack


class Data(object):
    def __init__(self, path, batch_size, maxlen):
        self.path = path
        self.batch_size = batch_size
        self.maxlen = maxlen

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        self.n_users, self.n_nodes = 0, 0
        self.n_train, self.n_test = 0, 0 
        self.neg_pools = {}

        self.exist_users = []
        
        with open(train_file) as f:
            for l in f.readlines():
                l = l.strip('\n').split('\t')
                uid = int(l[0])
                iid = int(l[1]) 
                self.exist_users.append(uid)
                self.n_nodes = max(self.n_nodes, int(iid))
                self.n_users = max(self.n_users, int(uid))
                self.n_train += 1
        

        with open(test_file) as f:
            for l in f.readlines():
                uid, iid = l.strip().split('\t')
                self.n_nodes = max(self.n_nodes, int(iid))
                self.n_test += 1
        self.n_nodes += 1
        self.n_users += 1
        self.n_items = self.n_nodes - self.n_users
        self.print_statistics()

        self.train_data, self.test_data, self.train_dict, self.test_dict, self.train_user_items_dict, self.train_item_users_dict, self.test_user_items_dict, self.test_item_users_dict = self.load_data_set(train_file, test_file)
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        # adjacent matrix
        with open(train_file) as f_train:
            for l in f_train:
                uid, iid = l.strip().split('\t')
                self.R[int(uid), int(iid)-self.n_users] = 1.
        # build temporal table 
        self.train_temporal_table = self.build_temporal_table(self.maxlen, self.train_user_items_dict, self.train_item_users_dict) # [M+N, S] 
        self.test_temporal_table = self.build_temporal_table(self.maxlen, self.test_user_items_dict, self.test_item_users_dict) # [M+N, S] 
        
    def load_data_set(self, train_file, test_file):
        train_dict = defaultdict(list)
        test_dict = defaultdict(list)
        train_user_items_dict = defaultdict(list)
        train_item_users_dict = defaultdict(list)
        test_user_items_dict = defaultdict(list)
        test_item_users_dict = defaultdict(list)
        train_data = []
        test_data = []
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train:
                    uid, iid = l.strip().split('\t')
                    train_dict[int(uid)].append(int(iid)-self.n_users)
                    train_user_items_dict[int(uid)].append(int(iid))
                    train_item_users_dict[int(iid)].append(int(uid))
                    train_data.append((int(uid), int(iid)))


                for l in f_test.readlines():
                    uid, iid = l.strip().split('\t')
                    test_dict[int(uid)].append(int(iid)-self.n_users)
                    test_user_items_dict[int(uid)].append(int(iid))
                    test_item_users_dict[int(iid)].append(int(uid))
                    test_data.append((int(uid), int(iid)))

        return train_data, test_data, train_dict, test_dict, train_user_items_dict, train_item_users_dict, test_user_items_dict, test_item_users_dict 


    def get_adj_mat(self):
        try:
            t1 = time()
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            print('already load adj matrix', norm_adj_mat.shape, time() - t1)
        
        except Exception:
            norm_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            
        return norm_adj_mat

    # get adj matrix
    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()
        # prevent memory from overflowing
        for i in range(5):
            adj_mat[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5), self.n_users:] =\
            R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)]
            adj_mat[self.n_users:,int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)] =\
            R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)].T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        
        t2 = time()
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()
        
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        
        print('already normalize adjacency matrix', time() - t2)
        return norm_adj_mat.tocsr()

    def build_temporal_order(self, maxlen, sequence):
        order = []
        last_node = sequence[-1]
        k = len(sequence)
        if k >= maxlen: 
            order.extend(sequence[k-maxlen:]) # recent maxlen items
        else:
            order.extend(sequence[:k] + [last_node] * (maxlen - k)) # padding unexist item id 
       
        return order


    def build_temporal_table(self, maxlen, user_items, item_users):
        # return [M+N, S]
        temporal_orders = defaultdict(list)

        for user_id in list(user_items.keys()): 
            item_list = user_items[user_id] 
            order = self.build_temporal_order(maxlen, item_list)
            temporal_orders[user_id].extend(order)
        
        
        for item_id in list(item_users.keys()): 
            user_list = item_users[item_id] 
            order = self.build_temporal_order(maxlen, user_list)
            temporal_orders[item_id].extend(order)
        
        # add last node
        padding_node = list(set(range(0, self.n_nodes)) - set(temporal_orders.keys()))
        for idx in padding_node:
            temporal_orders[idx].extend([idx]*maxlen)
        
        # build temporal_table [M+N, S]
        temporal_table = [temporal_orders[key] for key in sorted(temporal_orders.keys())]

        return np.array(temporal_table)

    def get_batch_data(self, data_list):
        random.shuffle(data_list) 
        data_len = len(data_list)
        batch_num = int(data_len / self.batch_size) + 1
        batch_data = []
        for i in range(batch_num):
            start = i * self.batch_size
            end = (i+1) * self.batch_size
            if end <= data_len:
                batch_data.append(data_list[start:end]) 
            else:
                # add_data
                add_data = data_list[0:batch_num*self.batch_size-data_len]
                last_batch = []
                last_batch.extend(data_list[start:data_len])
                last_batch.extend(add_data)
                batch_data.append(last_batch)

        return batch_data

    def neg_sampling(self, user):
        while True:
            neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_id not in self.train_dict[user] and neg_id not in self.test_dict[user]:
                neg_id = neg_id + self.n_users # item idx: M ~ (M+N-1)
                return neg_id

    def sample(self, batch_data):
        users = []
        pos_items = []
        neg_items = []
        for u, v in batch_data:
            users.append(u)
            pos_items.append(v)
            # negative sampling
            v_n = self.neg_sampling(u)
            neg_items.append(v_n)

        return users, pos_items, neg_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))


