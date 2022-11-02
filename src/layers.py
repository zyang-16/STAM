import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from parser import parse_args

args = parse_args()

class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        # name = kwargs.get('name')
        name = args.name
        if not name:
            layer = self.__class__.__name__.lower()
        self.name = name
        self.vars = {} 
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])



class STAM(Layer):
    def __init__(self, input_dim, n_heads, input_length, hidden_dim, **kwargs):
        super(STAM, self).__init__(**kwargs)
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.input_length = input_length
        self.hidden_dim = hidden_dim # D'
        self.attn_drop = args.attn_drop
        self.attention_head_size = int(hidden_dim / n_heads) 
        
        xavier_init = tf.random_normal_initializer(stddev=0.01)
        with tf.compat.v1.variable_scope(self.name + '_vars', reuse=tf.compat.v1.AUTO_REUSE):
            # define positional embedding
            self.vars["position_embedding"] = tf.compat.v1.get_variable('position_embedding', 
                                                        dtype=tf.float32,
                                                        shape=[self.input_length, self.input_dim],
                                                        initializer=xavier_init) # [S, d]

            # define W_Q, W_K, W_V
            self.vars["Q_embedding"] = tf.compat.v1.get_variable('Q_embedding', 
                                                        dtype=tf.float32,
                                                        shape=[self.input_dim, self.hidden_dim],
                                                        initializer=xavier_init) # [d, D']
            
            self.vars["K_embedding"] = tf.compat.v1.get_variable('K_embedding', 
                                                        dtype=tf.float32,
                                                        shape=[self.input_dim, self.hidden_dim],
                                                        initializer=xavier_init) # [d, D']
            
            self.vars["V_embedding"] = tf.compat.v1.get_variable('V_embedding', 
                                                        dtype=tf.float32,
                                                        shape=[self.input_dim, self.hidden_dim],
                                                        initializer=xavier_init) # [d, d]

    def __call__(self, inputs):
        # inputs: temporal-order embedding [batch_size, S, d]
        # 1. temporal input embeddings = temporal-position + temporal-order [batch_size, S, d]
        position_inputs = tf.tile(tf.expand_dims(tf.range(self.input_length), 0), [tf.shape(inputs)[0], 1]) # [batch_size, 1, S]
        temporal_inputs = inputs + tf.nn.embedding_lookup(self.vars['position_embedding'],
                                                          position_inputs) 
        
        # 2. Q, K, V
        q = tf.tensordot(temporal_inputs, self.vars['Q_embedding'], axes=(2, 0)) # [batch_size, S, D']
        k = tf.tensordot(temporal_inputs, self.vars['K_embedding'], axes=(2, 0)) # [batch_size, S, D']
        v = tf.tensordot(temporal_inputs, self.vars['V_embedding'], axes=(2, 0)) # [batch_size, S, D']


        batch_size = tf.shape(inputs)[0]
        q = tf.transpose(tf.reshape(q,[batch_size, self.input_length, self.n_heads, self.attention_head_size]), [0, 2, 1, 3]) # [batch_size, S, k, D'/k]
        k = tf.transpose(tf.reshape(k,[batch_size, self.input_length, self.n_heads, self.attention_head_size]), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v,[batch_size, self.input_length, self.n_heads, self.attention_head_size]), [0, 2, 1, 3])

        # scaled dot-product attention
        outputs = tf.matmul(q, k, transpose_b=True) 
        outputs = outputs / (self.input_length ** 0.5)

        input_mask = tf.ones(shape=[batch_size, self.input_length], dtype=tf.int32) 
        to_mask = tf.cast(tf.reshape(input_mask, [batch_size, 1, self.input_length]), tf.float32)
        broadcast_ones = tf.ones(shape=[batch_size, self.input_length, 1], dtype=tf.float32)
        mask = broadcast_ones * to_mask
        attention_mask = tf.expand_dims(mask, axis=[1])
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
        outputs += adder

        att_weights = tf.nn.softmax(outputs) 
        att_weights = tf.layers.dropout(att_weights, rate=self.attn_drop)

        h = tf.matmul(att_weights, v) # [batch_size, k, S, D'/k]
        h = tf.transpose(h, [0, 2, 1, 3]) # [batch_size, S, k, D'/k]
        h = tf.reshape(h, [batch_size, self.input_length, self.hidden_dim]) 

        ST_emb = self.feedforward(h)
        ST_emb += temporal_inputs

        return tf.nn.l2_normalize(ST_emb, 2) 

    def feedforward(self, inputs, reuse=None):
        with tf.variable_scope(self.name + '_vars', reuse=tf.AUTO_REUSE):
            inputs = tf.reshape(inputs, [-1, self.input_length, self.input_dim])
            params = {"inputs": inputs, "filters": self.input_dim, "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            outputs += inputs
        return outputs






