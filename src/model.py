from layers import *
from parser import parse_args
import numpy as np
import time
args = parse_args()

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        # name = kwargs.get('name')
        name = args.name
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False) 
        self.logging = logging
        self.vars = {}
        self.placeholders = {}
        self.layers = []
        self.activations = []
        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.compat.v1.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
        # Build metrics
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


# utilize STAM for GNN-based recommendation -- stacking
class STAM4Rec(Model):
    def __init__(self, placeholders, data_config, **kwargs):
        super(STAM4Rec, self).__init__(**kwargs)
        self.num_layers = args.num_layers
        self.input_dim = args.input_dim
        self.input_length = args.maxlen
        self.n_heads = args.n_heads
        self.hidden_dim = args.hidden_dim
        self.decay = args.decay
        self.dim = args.dim 
        self.n_fold = 10
        self.batch_size = args.batch_size
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']  
        self.norm_adj = data_config['norm_adj']
        self.mask = data_config['temporal_index']
        self.temporal_table = tf.Variable(data_config['temporal_table'], trainable=False, name="temporal_table") # [M+N, S]
        self.placeholders = placeholders 
        self.indices = np.squeeze(np.stack((np.repeat(np.arange(0, self.n_users+self.n_items), self.input_length).reshape(-1,1), data_config['temporal_table'].reshape(-1,1)), 1))
        print("self.norm_adj", self.norm_adj)

        self.users = self.placeholders['users'] # [1, batch_size] idx: 0 ~ (M-1)
        self.pos_items = self.placeholders['pos_items'] # [1, batch_size] idx: M ~ (M+N-1)
        self.neg_items = self.placeholders['neg_items'] # [1, batch_size] idx: M ~ (M+N-1)

        # user embedding matrix and item embedding matrix
        initializer = tf.random_normal_initializer(stddev=0.01)
        self.user_embeddings = tf.Variable(initializer([self.n_users, self.input_dim]), name='user_embeddings') # [M, d]
        self.item_embeddings = tf.Variable(initializer([self.n_items, self.input_dim]), name='item_embeddings') # [N, d]
        
        self.stam_layer = STAM(input_dim=self.input_dim, n_heads=self.n_heads,
                              input_length=self.input_length, hidden_dim=self.hidden_dim)

        self.W_Q = self.stam_layer.vars['Q_embedding']

        self.init_embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0) # [M+N, d] 

        self.user_embeddings_pre = tf.nn.embedding_lookup(self.user_embeddings, self.users)
        self.pos_item_embeddings_pre = tf.nn.embedding_lookup(self.init_embeddings, self.pos_items) # M~(M+N-1)
        self.neg_item_embeddings_pre = tf.nn.embedding_lookup(self.init_embeddings, self.neg_items) # M~(M+N-1)

        self.A_new = self.get_stam_weights()

        self._build()
    

    def _build(self):
        self.embeds = self.build_net() # [M+N, d]
        
        # Establish the final representations for user-item pairs in batch.
        self.user_emb = tf.nn.embedding_lookup(self.embeds, self.users)
        self.pos_item_embed = tf.nn.embedding_lookup(self.embeds, self.pos_items)
        self.neg_item_embed = tf.nn.embedding_lookup(self.embeds, self.neg_items)
        
        self._loss()
        self.init_optimizer()
    

    def get_temporal_neighbors(self, input_idx):
        neigh_idx = tf.nn.embedding_lookup(self.temporal_table, input_idx) # [len(input_idx), S]
        stam_inputs = tf.nn.embedding_lookup(self.init_embeddings, neigh_idx)# [len(input_idx), S, d]
        return stam_inputs

    def _convert_sp_mat_to_sp_tensor(self, X):  
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose() 
        return tf.SparseTensor(indices, coo.data, coo.shape) 


    # concat + L2 + mean-pooling for neighbor aggregation
    def get_stam_weights(self):
        # obtain spatiotemporal attention weights [M+N, M+N]
        length = self.n_users+self.n_items
        stam_batch_size = self.batch_size // 8
        batch_num = int(length // stam_batch_size)
        for i in range(batch_num):
            start = i * stam_batch_size
            end = (i+1) * stam_batch_size
            input_idx_batch = tf.range(start, end)
            stam_inputs = self.get_temporal_neighbors(input_idx_batch) # [batch_size, S, d] 
            # STAM forward  
            stam_outs_batch = self.stam_layer(stam_inputs) # spatiotemporal neighbor embeddings [batch_size, S, d]
            W_1 = tf.expand_dims(tf.nn.embedding_lookup(self.init_embeddings, input_idx_batch), 1) #[batch_size, 1, d]
            stam_weights_batch = tf.nn.softmax(tf.squeeze(tf.matmul(stam_outs_batch, tf.transpose(W_1, [0, 2, 1]))), 1) # [batch_size, S]
            if i == 0:
                stam_weights = stam_weights_batch
            else:
                stam_weights = tf.concat([stam_weights, stam_weights_batch], 0)
        input_idx_last = tf.range(batch_num * stam_batch_size, self.n_users+self.n_items)
        stam_inputs = self.get_temporal_neighbors(input_idx_last) # [batch_size, S, d] 
        stam_outs_batch = self.stam_layer(stam_inputs)
        W_1 = tf.expand_dims(tf.nn.embedding_lookup(self.init_embeddings, input_idx_last), 1)
        stam_weights_batch = tf.nn.softmax(tf.squeeze(tf.matmul(stam_outs_batch, tf.transpose(W_1, [0, 2, 1]))), 1)
        stam_weights = tf.concat([stam_weights, stam_weights_batch], 0)

        # stam_weights -> adj 
        adj_mask = tf.multiply(stam_weights, self.mask) 
        adj_sum = tf.div_no_nan(adj_mask, tf.reduce_sum(adj_mask, 1, keep_dims=True))
        adj = tf.boolean_mask(adj_sum, self.mask)
        adj_indices = tf.boolean_mask(self.indices, self.mask.reshape(1,-1)[0])
        
        self.stam_weights = tf.SparseTensor(indices=adj_indices,
                                            values=adj,
                                            dense_shape=[self.n_users+self.n_items, self.n_users+self.n_items])
        return self.stam_weights

    def build_net(self): 
        # stacking L layers     
        ego_embeddings = self.init_embeddings 
        all_embeddings = [ego_embeddings] 
        for k in range(0, self.num_layers): 
            side_embeddings = tf.sparse_tensor_dense_matmul(self.A_new, ego_embeddings)
            ego_embeddings = side_embeddings 
            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)  
        return all_embeddings

    def affinity_score(self, user_emb, item_emb):
        return tf.reduce_sum(user_emb * item_emb, axis=1)


    def _loss(self):
        # bpr loss for GNN-based recommendation
        pos_score = self.affinity_score(self.user_emb, self.pos_item_embed)
        neg_score = self.affinity_score(self.user_emb, self.neg_item_embed) 

        regularizer = tf.nn.l2_loss(self.user_embeddings_pre) + tf.nn.l2_loss(self.pos_item_embeddings_pre) + tf.nn.l2_loss(self.neg_item_embeddings_pre)
        regularizer = regularizer / self.batch_size

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_score - neg_score)))

        emb_loss = self.decay * regularizer

        self.bpr_loss = mf_loss + emb_loss

    def init_optimizer(self):
        trainable_params = tf.trainable_variables()
        print("trainable_params", trainable_params)
        gradients = tf.gradients(self.bpr_loss, trainable_params)
        clipped_grads_and_vars, _ = tf.clip_by_global_norm(gradients, args.max_gradient_norm)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        self.opt_op = self.optimizer.apply_gradients(zip(clipped_grads_and_vars, trainable_params))

