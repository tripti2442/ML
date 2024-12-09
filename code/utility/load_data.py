'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
from utility.data_partition import *
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity 

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)
class Data(object):
    def __init__(self, path, batch_size,part_type,part_num,part_T):
        self.path = path
        self.batch_size = batch_size
        self.part_type = part_type
        self.part_num =part_num

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []
        
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.
                        
                    self.train_items[uid] = train_items
                    
                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    
                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        if self.part_type != 0:
            try:
                with open(self.path + '/C_type-' +str(part_type)+'_num-'+ str(part_num)+'.pk', 'r') as f:
                    self.C = pickle.load(f)
                with open(self.path + '/C_U_type-' +str(part_type)+'_num-'+ str(part_num)+'.pk', 'r') as f:
                    self.C_U = pickle.load(f)
                with open(self.path + '/C_I_type-' +str(part_type)+'_num-'+ str(part_num)+'.pk', 'r') as f:
                    self.C_I = pickle.load(f)
            except Exception:
                if part_type==1:
                    self.C, self.C_U,self.C_I = data_partition_1(self.train_items,part_num,part_T)
                if part_type==2:
                    self.C, self.C_U,self.C_I = data_partition_2(self.train_items,part_num,part_T)
                if part_type==3:
                    self.C, self.C_U,self.C_I = data_partition_3(self.train_items,part_num)

                with open(self.path + '/C_type-' +str(part_type)+'_num-'+ str(part_num)+'.pk', 'w') as f:
                    pickle.dump(self.C,f)
                with open(self.path + '/C_U_type-' +str(part_type)+'_num-'+ str(part_num)+'.pk', 'w') as f:
                    pickle.dump(self.C_U,f)
                with open(self.path + '/C_I_type-' +str(part_type)+'_num-'+ str(part_num)+'.pk', 'w')as f:
                    pickle.dump(self.C_I, f)

            self.n_C=[]

            for i in range(len(self.C)):
                t=0
                for j in self.C[i]:
                    t+=len(self.C[i][j])
                self.n_C.append(t)

        self.print_statistics()

    def load_embeddings(self):
        """
        Loads user/item embeddings from precomputed pickle files if they exist.
        If they don't exist, this method creates placeholders.
        """
        try:
            with open(self.path + "/user_content_features.pk", "rb") as f:
                self.user_embeddings = pickle.load(f)
                print("Loaded user embeddings from pickle.")
        except FileNotFoundError:
            self.user_embeddings = np.random.rand(self.n_users, 64)  # Fallback to random embeddings
            print("User embeddings not found. Generated random embeddings.")

        try:
            with open(self.path + "/movie_content_features.pk", "rb") as f:
                self.item_embeddings = pickle.load(f)
                print("Loaded item embeddings from pickle.")
        except FileNotFoundError:
            self.item_embeddings = np.random.rand(self.n_items, 64)  # Fallback to random embeddings
            print("Item embeddings not found. Generated random embeddings.")

    def create_user_user_adj_matrix(self):
        with open("user_content_features.pk", "rb") as f:
         user_content_embeddings = pickle.load(f)
        # Compute pairwise cosine similarity between user embeddings
        user_similarity = cosine_similarity(user_content_embeddings)
    
        # Normalize similarities to be between 0 and 1 (if needed)
        user_similarity = (user_similarity - user_similarity.min()) / (user_similarity.max() - user_similarity.min())
    
        # Convert to sparse matrix for memory efficiency
        user_user_adj = sp.csr_matrix(user_similarity)
        return user_user_adj
    
    def create_item_item_adj_matrix(self):
        with open("movie_content_features.pk", "rb") as f:
         item_content_embeddings = pickle.load(f)
    
        # Compute pairwise cosine similarity between item embeddings
        item_similarity = cosine_similarity(item_content_embeddings)
    
        # Normalize similarities to be between 0 and 1 (if needed)
        item_similarity = (item_similarity - item_similarity.min()) / (item_similarity.max() - item_similarity.min())
    
        # Convert to sparse matrix for memory efficiency
        item_item_adj = sp.csr_matrix(item_similarity)
        return item_item_adj

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
        
        # Load user-user and item-item adjacency matrices
        user_user_adj = self.create_user_user_adj_matrix()  # Load user-user similarity matrix
        item_item_adj = self.create_item_item_adj_matrix()  # Load item-item similarity matrix
    
        # Add user-user similarity to the top-left quadrant of the adjacency matrix
        adj_mat[:self.n_users, :self.n_users] = user_user_adj.tolil()
    
       # Add item-item similarity to the bottom-right quadrant of the adjacency matrix
        adj_mat[self.n_users:, self.n_users:] = item_item_adj.tolil()
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        return adj_mat.tocsr() 

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)
        
        except Exception:
            adj_mat= self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            
        try:
            pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
        except Exception:
            adj_mat=adj_mat
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat = norm_adj.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
            
        return pre_adj_mat

    def get_adj_mat_local(self,local):
        try:
            f_path = self.path +'/type_' +str(self.part_type)+'/nums_'+str(self.part_num)+'/s_adj_mat_local_'+str(local)+'.npz'
            adj_mat = sp.load_npz(f_path)
        except Exception as e:
            adj_mat= self.create_adj_mat_local(local)
            f_path = self.path +'/type_' +str(self.part_type)+'/nums_'+str(self.part_num)+'/s_adj_mat_local_'+str(local)+'.npz'
            ensureDir(f_path)
            sp.save_npz(f_path, adj_mat)
            
        try:
            pre_adj_mat = sp.load_npz(self.path +'/type_' +str(self.part_type)+'/nums_'+str(self.part_num)+ '/s_pre_adj_mat_local_'+str(local)+'.npz')
        except Exception:
            adj_mat=adj_mat
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat = norm_adj.tocsr()

            f_path = self.path +'/type_' +str(self.part_type)+'/nums_'+str(self.part_num)+ '/s_pre_adj_mat_local_'+str(local)+'.npz'
            ensureDir(f_path)

            sp.save_npz(f_path, norm_adj)
            
        return pre_adj_mat
    
    def create_adj_mat_local(self, local):
   
        t1 = time()
    
        # Initialize the adjacency matrix
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        # Create the local user-item interaction matrix (R_local)
        R_local = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        for u in self.C[local]:
          for i in self.C[local][u]:
            R_local[u, i] = 1.0
        R_local = R_local.tolil()

        # Add local user-item interactions
        for i in range(5):  # Process in chunks to prevent memory overflow
          adj_mat[int(self.n_users * i / 5.0):int(self.n_users * (i + 1.0) / 5), self.n_users:] = \
            R_local[int(self.n_users * i / 5.0):int(self.n_users * (i + 1.0) / 5)]
          adj_mat[self.n_users:, int(self.n_users * i / 5.0):int(self.n_users * (i + 1.0) / 5)] = \
            R_local[int(self.n_users * i / 5.0):int(self.n_users * (i + 1.0) / 5)].T

        # Load user-user and item-item adjacency matrices
         # Load user-user and item-item adjacency matrices
        user_user_adj = self.create_user_user_adj_matrix()  # Load user-user similarity matrix
        item_item_adj = self.create_item_item_adj_matrix()  # Load item-item similarity matrix
    
        # Add user-user similarity to the top-left quadrant of the adjacency matrix
        adj_mat[:self.n_users, :self.n_users] = user_user_adj.tolil()
    
       # Add item-item similarity to the bottom-right quadrant of the adjacency matrix
        adj_mat[self.n_users:, self.n_users:] = item_item_adj.tolil()
        # Convert to sparse format
        adj_mat = adj_mat.todok()
        print('Created local adjacency matrix for locality:', local, adj_mat.shape, time() - t1)
    
        return adj_mat.tocsr()


    '''def create_adj_mat_local(self,local):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        R_local = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        for u in self.C[local]:
            for i in self.C[local][u]:
                R_local[u, i] = 1.
        R_local = R_local.tolil()

        for i in range(5):
            adj_mat[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5), self.n_users:] =\
            R_local[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)]
            adj_mat[self.n_users:,int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)] =\
            R_local[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)].T
        adj_mat = adj_mat.todok()

        return adj_mat.tocsr()'''

     
   
    '''def create_adj_mat(self):
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

        return adj_mat.tocsr() '''
      
    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)


    def local_sample(self,local):
        if self.batch_size <= len(self.C_U[local]):
            users = rd.sample(self.C_U[local], self.batch_size)
        else:
            users = [rd.choice(self.C_U[local]) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.C[local][u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]
                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            can_items=self.C_I[local]
            n_can_items=len(can_items)
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=n_can_items,size=1)[0]
                neg_i_id = can_items[neg_id]
                if neg_i_id not in self.train_items[u] and neg_i_id not in neg_items:
                    neg_items.append(neg_i_id)
            return neg_items

        def sample_neg_items_for_u2(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]


        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
    
    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in (self.test_set[u]+self.train_items[u]) and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items
    
        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
    
    
    
    
    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))
        if self.part_type != 0:
            print('training nums of each local data:')
            print self.n_C


    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state
