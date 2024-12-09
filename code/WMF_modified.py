import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix
from utility.batch_test import *
import pickle
import copy
import time
import sys

# Load content features
with open('../data/ml-1m/user_profile.pk', 'rb') as f:
    user_content_features = pickle.load(f)

with open('../data/ml-1m/movie_profile.pk', 'rb') as f:
    movie_content_features = pickle.load(f)


class WMF:
    def _init_(self, user_num, item_num, max_item_pu, user_content_dim, item_content_dim):
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_size = args.embed_size
        self.max_item_pu = max_item_pu
        self.weight1 = args.negative_weight
        self.lambda_bilinear = [1e-4, 1e-4]  # Regularization
        self.lr = args.lr
        self.Ks = eval(args.Ks)
        self.user_content_dim = user_content_dim
        self.item_content_dim = item_content_dim

    def _create_placeholders(self):
        self.input_u = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_ur = tf.placeholder(tf.int32, [None, None], name="input_ur")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))

    def _create_variables(self):
        # Collaborative embeddings
        self.uidW = tf.Variable(tf.truncated_normal(shape=[self.user_num, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="uidW")
        self.iidW = tf.Variable(tf.truncated_normal(shape=[self.item_num + 1, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="iidW")

        # Content feature embeddings
        self.user_contentW = tf.Variable(tf.truncated_normal(
            shape=[self.user_num, self.user_content_dim], mean=0.0, stddev=0.01), dtype=tf.float32, name="user_contentW")
        self.item_contentW = tf.Variable(tf.truncated_normal(
            shape=[self.item_num + 1, self.item_content_dim], mean=0.0, stddev=0.01), dtype=tf.float32, name="item_contentW")

    def _create_inference(self):
        # Base embeddings
        self.uid = tf.nn.embedding_lookup(self.uidW, self.input_u)
        self.uid = tf.reshape(self.uid, [-1, self.embedding_size])
        self.uid = tf.nn.dropout(self.uid, rate=1 - self.dropout_keep_prob)

        self.pos_item = tf.nn.embedding_lookup(self.iidW, self.input_ur)
        self.pos_num_r = tf.cast(tf.not_equal(self.input_ur, self.item_num), 'float32')
        self.pos_item = tf.einsum('ab,abc->abc', self.pos_num_r, self.pos_item)

        # Content features
        user_ids = tf.squeeze(self.input_u, axis=1)
        content_features_u = tf.nn.embedding_lookup(self.user_contentW, user_ids)
        content_features_i = tf.nn.embedding_lookup(self.item_contentW, tf.reshape(self.input_ur, [-1]))

        # Reduce content dimensions and combine embeddings
        content_features_u = tf.layers.dense(content_features_u, self.embedding_size // 2, activation=tf.nn.relu)
        content_features_i = tf.layers.dense(content_features_i, self.embedding_size // 2, activation=tf.nn.relu)

        self.uid = tf.layers.dense(tf.concat([self.uid, content_features_u], axis=1), self.embedding_size, activation=tf.nn.relu)
        content_features_i = tf.reshape(content_features_i, [-1, self.max_item_pu, self.embedding_size // 2])
        self.pos_item = tf.layers.dense(tf.concat([self.pos_item, content_features_i], axis=2), self.embedding_size, activation=tf.nn.relu)

        # Match dimensions and compute element-wise interactions
        self.uid = tf.expand_dims(self.uid, axis=1)
        self.pos_r = tf.reduce_sum(tf.multiply(self.uid, self.pos_item), axis=2)

    def _pre(self):
        u_e = tf.nn.embedding_lookup(self.uidW, self.users)
        pos_i_e = tf.nn.embedding_lookup(self.iidW, self.pos_items)
        self.batch_ratings = tf.matmul(u_e, pos_i_e, transpose_a=False, transpose_b=True)

    def _create_loss(self):
        # Reduce embeddings for loss calculation
        reduced_pos_item = tf.reduce_sum(self.pos_item[:, :, :self.embedding_size], axis=1)
        reduced_uid = self.uid[:, :, :self.embedding_size]

        # Interaction loss
        self.loss1 = self.weight1 * tf.reduce_sum(
            tf.einsum('abc,abd->acd', reduced_uid, reduced_uid) *
            tf.einsum('ab,ac->bc', reduced_pos_item, reduced_pos_item)
        )
        self.loss1 += tf.reduce_sum((1.0 - self.weight1) * tf.square(self.pos_r) - 2.0 * self.pos_r)

        # Regularization
        self.l2_loss = tf.nn.l2_loss(self.uidW) + tf.nn.l2_loss(self.iidW) + \
                       tf.nn.l2_loss(self.user_contentW) + tf.nn.l2_loss(self.item_contentW)
        self.loss = self.loss1 + self.lambda_bilinear[0] * self.l2_loss

        # Optimizer
        self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr, initial_accumulator_value=1e-8).minimize(self.loss)

    def _build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._pre()


def get_lables(temp_set, k=0.9999):
    max_item = 0
    item_lenth = []
    for i in temp_set:
        item_lenth.append(len(temp_set[i]))
        if len(temp_set[i]) > max_item:
            max_item = len(temp_set[i])
    item_lenth.sort()
    max_item = item_lenth[int(len(item_lenth) * k) - 1]

    print(max_item)
    for i in temp_set:
        if len(temp_set[i]) > max_item:
            temp_set[i] = temp_set[i][0:max_item]
        while len(temp_set[i]) < max_item:
            temp_set[i].append(n_items)
    return max_item, temp_set


def get_train_instances(lable):
    user_train, item = [], []

    for i in lable.keys():
        user_train.append(i)
        item.append(lable[i])
    user_train = np.array(user_train)[:, np.newaxis]
    item = np.array(item)
    return user_train, item


if _name_ == '_main_':
    tf.set_random_seed(2021)
    np.random.seed(2021)

    n_users, n_items = data_generator.n_users, data_generator.n_items
    train_items = copy.deepcopy(data_generator.train_items)
    max_item, lable = get_lables(train_items)

    user_content_dim = len(next(iter(user_content_features.values())))
    item_content_dim = len(next(iter(movie_content_features.values())))

    model = WMF(n_users, n_items, max_item, user_content_dim, item_content_dim)
    model._build_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    user_train1, item1 = get_train_instances(lable)
    batch_size = args.batch_size
    epoch_size = args.epoch

    for epoch in range(epoch_size):
        shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
        user_train1 = user_train1[shuffle_indices]
        item1 = item1[shuffle_indices]

        for start in range(0, len(user_train1), batch_size):
            end = min(start + batch_size, len(user_train1))
            batch_users = user_train1[start:end]
            batch_items = item1[start:end]

            feed_dict = {model.input_u: batch_users,
                         model.input_ur: batch_items,
                         model.dropout_keep_prob: 0.9}

            _, loss = sess.run([model.opt, model.loss], feed_dict=feed_dict)
            print("Epoch: {}, Batch Loss: {}".format(epoch, loss))

    trained_user_embeddings = sess.run(model.uidW)
    trained_item_embeddings = sess.run(model.iidW)

    with open('../data/ml-1m/users_pretrain.pk', 'wb') as f:
        pickle.dump(trained_user_embeddings, f)

    with open('../data/ml-1m/items_pretrain.pk', 'wb') as f:
        pickle.dump(trained_item_embeddings, f)

    print("User and item embeddings saved to 'users_pretrain.pk' and 'items_pretrain.pk'.")