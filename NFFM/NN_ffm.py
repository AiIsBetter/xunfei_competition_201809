#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2018.09.19

import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
import itertools
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

class NNFM(BaseEstimator, TransformerMixin):
    def __init__(self,static_feature_size,static_categorical_col ,dynamic_categorical_col,field_size,dynamic_featrue_size,dynamic_max_len,
                 embedding_size=8, deep_layers=[32, 32],dropout_deep=[1.0, 1.0, 1.0],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True
                 ):
        self.static_feature_size = static_feature_size  # denote as M, size of the feature dictionary
        self.static_categorical_col = static_categorical_col
        self.dynamic_categorical_col = dynamic_categorical_col
        self.dynamic_featrue_size = dynamic_featrue_size
        self.dynamic_max_len = dynamic_max_len

        self.field_size = field_size  # denote as F, size of the feature fields
        self.embedding_size = embedding_size  # denote as K, size of the feature embedding

        # self.dropout_ffm = dropout_ffm
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        # self.use_fm = use_fm
        # self.use_deep = use_deep
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [], []

        self._init_graph()



    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # self.feature_index = tf.placeholder(tf.int16,[None,None],name = 'input')
            self.label = tf.placeholder(tf.int8, shape=[None,1], name='output')
            self.weights = self._initialize_weights()
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")
            # self.dropout_ffm_layer = tf.placeholder(tf.float16,shape = [None],name = 'drouput_ffm_0')
            self.dropout_deep_layer = tf.placeholder(tf.float32, shape=[None], name='drouput_deep_0')

            # lr part
            # static
            self.static_feature_index = {}
            for key in self.static_categorical_col:
                self.static_feature_index[key] = tf.placeholder(tf.int32, shape=[None],
                                                             name=key + "_st_input")  #
            # dynamic
            self.dynamic_feature_index = {}
            self.dynamic_lengths_dict = {}
            self.dynamic_feature_index[self.dynamic_categorical_col] = tf.placeholder(tf.int32, shape=[None, self.dynamic_max_len],
                                                            name=key + "_dy_input")  #
            self.dynamic_lengths_dict[self.dynamic_categorical_col] = tf.placeholder(tf.int32, shape=[None],
                                                            name=key + "_dy_length")  # None
            # embedding
            # static
            self.static_lr_embs = [tf.nn.embedding_lookup( self.weights['static_lr_part_embedding'][key], self.static_feature_index[key]) for key in  self.static_categorical_col]
            self.static_lr_embs = tf.concat(self.static_lr_embs, axis=1)
            # dynamic
            self.dynamic_lr_embs = tf.nn.embedding_lookup(self.weights['dynamic_lr_part_embedding'][self.dynamic_categorical_col],
                                                          self.dynamic_feature_index[self.dynamic_categorical_col])#none * maxlen
            self.dynamic_lr_embs = tf.reshape(self.dynamic_lr_embs,shape = (-1,self.dynamic_max_len))#none * maxlen
            self.dynamic_lengths_dict_reshape = tf.reshape(self.dynamic_lengths_dict[self.dynamic_categorical_col],shape = (-1,1))#none-》#none*1
            self.dynamic_lr_embs = tf.div(self.dynamic_lr_embs, tf.to_float(self.dynamic_lengths_dict_reshape))

            # ffm part
            # static
            self.static_feature_ffm_index = {}
            for key in self.static_categorical_col:
                self.static_feature_ffm_index[key] = tf.placeholder(tf.int32, shape=[None],
                                                                name=key + "_st_ffm_input")  #

            embed_var_raw_dict = {}
            embed_var_dict = {}
            for key in self.static_categorical_col:
                embed_var_raw = tf.nn.embedding_lookup(self.weights['static_ffm_part_embedding'][key],
                                                                  self.static_feature_ffm_index[key])
                embed_var_raw_dict[key] = tf.reshape(embed_var_raw,[-1,self.field_size,self.embedding_size])

            #  dynamic
            self.dynamic_feature_ffm_index = {}
            self.dynamic_feature_ffm_index[self.dynamic_categorical_col] = tf.placeholder(tf.int32, shape=[None, self.dynamic_max_len],
                                                                    name=self.dynamic_categorical_col + "_dy_ffm_input")
            embed_var_raw = tf.nn.embedding_lookup(self.weights['dynamic_ffm_part_embedding'][self.dynamic_categorical_col],
                                                             self.dynamic_feature_ffm_index[self.dynamic_categorical_col])
            # tf.sequence_mask，给一个序列，返回一个对应的True False序列
            # tf.sequence_mask([1, 3, 2], 5)
            #  [[True, False, False, False, False],
            #  [True, True, True, False, False],
            #  [True, True, False, False, False]]
            # tf.sequence_mask([[1, 3], [2, 0]])  # [[[True, False, False],
            #   [True, True, True]],
            #  [[True, True, False],
            #   [False, False, False]]]
            ffm_mask = tf.sequence_mask(self.dynamic_lengths_dict[self.dynamic_categorical_col],
                                        maxlen=self.dynamic_max_len)  # None * max_len

            ffm_mask = tf.expand_dims(ffm_mask, axis=-1)  # None * max_len * 1
            ffm_mask = tf.concat([ffm_mask for i in range(self.embedding_size * self.field_size)],
                                 axis=-1)  # None * max_len * [k * F]
            embed_var_raw = tf.multiply(embed_var_raw, tf.to_float(ffm_mask))  # None * max_len * [k * F]
            embed_var_raw = tf.reduce_sum(embed_var_raw, axis=1)  # None * [k*F]
            padding_lengths = tf.concat([tf.expand_dims(self.dynamic_lengths_dict[self.dynamic_categorical_col], axis=-1)#none * 1
                                         for i in range(self.embedding_size * self.field_size)],
                                        axis=-1)  # None * [k*F]
            embed_var_raw = tf.div(embed_var_raw, tf.to_float(padding_lengths))  # None * [k*F]
            embed_var_raw_dict[self.dynamic_categorical_col] = tf.reshape(embed_var_raw, [-1, self.field_size, self.embedding_size])

            self.static_categorical_col_copy = self.static_categorical_col.copy()
            self.static_categorical_col_copy.append(self.dynamic_categorical_col)
            for (i1, i2) in itertools.combinations(list(range(0, len(self.static_categorical_col_copy))), 2):
                c1, c2 = self.static_categorical_col_copy[i1],self.static_categorical_col_copy[i2]

                embed_var_dict.setdefault(c1, {})[c2] = embed_var_raw_dict[c1][:, i2, :]  # None * k
                embed_var_dict.setdefault(c2, {})[c1] = embed_var_raw_dict[c2][:, i1, :]  # None * k
            x_mat = []
            y_mat = []
            self.input_size = 0
            for (c1, c2) in itertools.combinations(embed_var_dict.keys(), 2):
                self.input_size += 1
                x_mat.append(embed_var_dict[c1][c2])  # input_size * None * k
                y_mat.append(embed_var_dict[c2][c1])  # input_size * None * k
            x_mat = tf.transpose(x_mat, perm=[1, 0, 2])  # None * input_size * k
            y_mat = tf.transpose(y_mat, perm=[1, 0, 2])  # None * input_size * k

            x = tf.multiply(x_mat, y_mat)
            self.flat_vars = tf.reshape(x, [-1, self.input_size * self.embedding_size])  # None * [input_size * k]

            # deep
            self.deep = tf.nn.dropout(self.flat_vars,self.dropout_deep_layer[0])
            for i in range(len(self.deep_layers)):
                self.deep = tf.matmul(self.deep, self.weights["deep_layer_%d" % i])
                self.deep = tf.add(self.deep, self.weights["deep_bias_%d" % i])
                if self.batch_norm:
                    self.deep = self.batch_norm_layer(self.deep, train_phase=self.train_phase,
                                                        scope_bn="bn_%d" % i)  # None * layer[i] * 1
                self.deep = self.deep_layers_activation(self.deep)
                self.deep = tf.nn.dropout(self.deep, self.dropout_deep_layer[i + 1])







            self.out = tf.add(tf.matmul(self.deep, self.weights["concat_projection"]), self.weights["concat_bias"])

            self.out = tf.add(self.out, tf.reshape(tf.reduce_sum(self.static_lr_embs, axis=1), [-1, 1]))
            self.out = tf.add(self.out, tf.reshape(tf.reduce_sum(self.dynamic_lr_embs, axis=1), [-1, 1]))
            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
                # if True:
                #     for i in range(len(self.deep_layers)):
                #         self.loss += tf.contrib.layers.l2_regularizer(
                #             self.l2_reg)(self.weights["layer_%d"%i])

            # optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)
            # init
            self.saver = tf.train.Saver()
            self.sess = self._init_session()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def _initialize_weights(self):
        weights = dict()
        weights['static_lr_part_embedding'] = {}
        # weights['static_lr_part_bias'] = {}
        for key in self.static_categorical_col:
            weights['static_lr_part_embedding'][key] = tf.Variable(tf.random_normal([self.static_feature_size[key],1],0.0,0.01),name = 's_l_p_e')
            # weights['static_lr_part_bias'][key] = tf.Variable(tf.random_normal([1],0.0,0.01), name='s_l_p_b')
        # dynamic
        weights['dynamic_lr_part_embedding'] = {}
        # weights['static_lr_part_bias'] = {}
        a = self.dynamic_categorical_col
        weights['dynamic_lr_part_embedding'][self.dynamic_categorical_col] = tf.Variable(tf.random_normal([self.dynamic_featrue_size,1],0.0,0.01),name = 'd_l_p_e')



        # ffm embedding
        # static
        weights['static_ffm_part_embedding'] = {}
        for key in self.static_categorical_col:
            weights['static_ffm_part_embedding'][key] = tf.Variable(
                tf.random_normal([self.static_feature_size[key], self.embedding_size*self.field_size], 0.0, 0.01), name='s_f_p_e')
            # weights['static_lr_part_bias'][key] = tf.Variable(tf.random_normal([1],0.0,0.01), name='s_l_p_b')
        # dynamic
        weights['dynamic_ffm_part_embedding'] = {}
        weights['dynamic_ffm_part_embedding'][self.dynamic_categorical_col] = tf.Variable(
                tf.random_normal([self.dynamic_featrue_size, self.embedding_size * self.field_size], 0.0, 0.01),
                name='d_f_p_e')



        # deep layer
        input_size = 0
        temp = self.static_categorical_col.copy()
        temp.append(self.dynamic_categorical_col)
        for (c1, c2) in itertools.combinations(temp, 2):
            input_size += 1
        input_size = input_size*self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights["deep_layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])),
            dtype=np.float32)  # F*K*deep_layers[0]

        weights["deep_bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1,len(self.deep_layers)):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights["deep_layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1] , self.deep_layers[i])),
                dtype=np.float32)  # F*K*deep_layers[0]
            weights["deep_bias_%d" % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1,self.deep_layers[i-1])),
            dtype=np.float32)


        # out
        glorot = np.sqrt(2.0 / (self.deep_layers[len(self.deep_layers)-1] + 1))
        weights["concat_projection"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[len(self.deep_layers)-1], 1)),
            dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights
    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z
    def get_batch(self,x,x_dynamic,x_dynamic_length,y,index,batch_size):

        begin_index = index*batch_size
        end_index =(index+1)*batch_size
        if end_index >x.shape[0]:
            end_index = x.shape[0]
        x_batch = x.iloc[begin_index:end_index]
        x_dynamic_batch = x_dynamic[begin_index:end_index]
        x_dynamic_length_batch = x_dynamic_length[begin_index:end_index]
        y_batch = y.iloc[begin_index:end_index]
        return x_batch,x_dynamic_batch,x_dynamic_length_batch,y_batch

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b,c,d):
        rng_state = np.random.get_state()
        for key in a.columns:
            np.random.set_state(rng_state)
            np.random.shuffle(a[key].values)
        np.random.set_state(rng_state)
        np.random.shuffle(b.values)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)
    def fit_on_batch(self,x_batch,Xi_dynamic_batch,Xi_dynamic_length_batch,y_batch):
        feed_dict = {self.label:y_batch.values.reshape(-1,1),self.dropout_deep_layer:self.dropout_deep ,self.train_phase: True
                     ,self.dynamic_feature_index[self.dynamic_categorical_col]:Xi_dynamic_batch
                     ,self.dynamic_lengths_dict[self.dynamic_categorical_col] : Xi_dynamic_length_batch
                     ,self.dynamic_feature_ffm_index[self.dynamic_categorical_col]:Xi_dynamic_batch}
        for key in x_batch.columns:
            feed_dict[self.static_feature_index[key]] = x_batch[key].values
        for key in self.static_categorical_col:
            feed_dict[self.static_feature_ffm_index[key]] = x_batch[key].values
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss
    def predict(self,x,x_dynamic,x_dynamic_length):
        total_batch = int(len(x) / self.batch_size)
        if total_batch <=0:
            total_batch = 1
        y_pred  = np.zeros(shape = (x.shape[0],1))
        y = x.iloc[:,1]
        for i in range(total_batch):
            x_train_batch, x_dynamic_batch,x_dynamic_length_batch,y_train_batch = self.get_batch(x,x_dynamic,x_dynamic_length, y, i, self.batch_size)
            feed_dict = {self.label: y_train_batch.values.reshape(-1, 1), self.dropout_deep_layer:[1.0] * len(self.dropout_deep)
                ,self.train_phase: True,self.dynamic_feature_index[self.dynamic_categorical_col]:x_dynamic_batch
                     ,self.dynamic_lengths_dict[self.dynamic_categorical_col] : x_dynamic_length_batch
                     ,self.dynamic_feature_ffm_index[self.dynamic_categorical_col]:x_dynamic_batch}

            for key in self.static_categorical_col:
                feed_dict[self.static_feature_index[key]] = x_train_batch[key].values
            for key in self.static_categorical_col:
                feed_dict[self.static_feature_ffm_index[key]] = x_train_batch[key].values
            y_pred_temp, opt = self.sess.run((self.out,self.loss ), feed_dict=feed_dict)
            begin_index = i* self.batch_size
            end_index = (i + 1) * self.batch_size
            if end_index > x.shape[0]:
                end_index = x.shape[0]

            y_pred[begin_index:end_index] = y_pred_temp

        return y_pred
    def evaluate(self, x, x_dynamic_valid,x_dynamic_length , y):
        y_pred = self.predict(x, x_dynamic_valid,x_dynamic_length )
        return self.eval_metric(y, y_pred)

    def termination(self,valid_result,greater_is_better):
        X = False
        if greater_is_better :
            if valid_result[-2]>valid_result[-1] :

                X =True
                return X
        else:
            if valid_result[-2]<valid_result[-1] :
                X = True
                return X

        return X


    def fit(self,Xi_train_,  y_train_, Xi_valid_, y_valid_,Xi_dynamic_,Xi_dynamic_valid_,
                 Xi_dynamic_length_, Xi_dynamic_length_valid_,early_stopping=False):
        self.shuffle_in_unison_scary(Xi_train_,y_train_,Xi_dynamic_,Xi_dynamic_length_)
        total_batch = int(len(y_train_) / self.batch_size)
        has_valid = Xi_valid_ is not None
        for epoch in range(self.epoch):
            t1 = time()
            for i in range(total_batch):
                Xi_train_batch ,Xi_dynamic_batch,Xi_dynamic_length_batch ,y_train_batch=  self.get_batch(Xi_train_,Xi_dynamic_,Xi_dynamic_length_,y_train_,i,self.batch_size)
                loss = self.fit_on_batch(Xi_train_batch,Xi_dynamic_batch,Xi_dynamic_length_batch,y_train_batch)
                # print(i)
            print('epoch:',epoch)
            train_result = self.evaluate(Xi_train_,Xi_dynamic_,Xi_dynamic_length_,y_train_)
            self.train_result.append(train_result)
            if has_valid:
                valid_result = self.evaluate(Xi_valid_, Xi_dynamic_valid_,Xi_dynamic_length_valid_,y_valid_)
                self.valid_result.append(valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result, valid_result, time() - t1))
                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result, time() - t1))

            if early_stopping and has_valid and len(self.valid_result)>1:
                if self.termination(self.valid_result,self.greater_is_better):
                    break



