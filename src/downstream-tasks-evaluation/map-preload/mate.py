import tensorflow as tf


def label_smoothing(inputs, epsilon=0.1):
    inputs = tf.cast(inputs, tf.float32)
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)


def cross_entropy(logits, labels, class_num, isSmoothing = False):
    '''compute loss of cross_entropy'''
    if isSmoothing: # label smoothing
        one_hot_label = tf.one_hot(tf.cast(labels, tf.int32), depth=class_num)
        smoothed_label = label_smoothing(one_hot_label)
        loss_entropy_event = tf.nn.softmax_cross_entropy_with_logits_v2(labels=smoothed_label, logits=logits)
    else:
        loss_entropy_event = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.int32))

    # # wighted_loss
    # weight = tf.cast(tf.gather(self.weight_per_class, tf.cast(batch_target, tf.int32)), loss_entropy_event.dtype)
    # loss_entropy_event = tf.multiply(loss_entropy_event, weight)
    cost = tf.reduce_mean(loss_entropy_event)
    return cost


def feedforward_sparse(inputs, num_units,
                       scope="multihead_attention_feedforward", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.elu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
    return outputs    #(batch_size, maxlen, hidden_units)


def normalize(inputs, epsilon=1e-8, scope="layer_normalization", reuse=None):
    #layer normalize
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
        return outputs


def embedding(inputs, vocab_size, num_units, zero_pad=False, scale=True, scope="embedding", reuse=None,
              fine_tuning=False, pretrain_emb=None):
    with tf.variable_scope(scope, reuse=reuse):
        if fine_tuning:
            pretrain_emb = tf.concat([pretrain_emb, tf.zeros([1, num_units], dtype=tf.float32)], axis=0)
            lookup_table = tf.get_variable('lookup_table',
                                           dtype=tf.float32,
                                           # shape=[vocab_size + 1, num_units],
                                           initializer=pretrain_emb,)
        else:
            lookup_table = tf.get_variable('lookup_table',
                                           dtype=tf.float32,
                                           shape=[vocab_size + 1, num_units],
                                           initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    return outputs


def get_last_emb(inputs, time, lastlen, maxlen, scope='get_last_emb', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.cast(inputs, tf.float32)  # [N, T, C]
        T = maxlen
        _, last_inputs = tf.split(inputs, [T-lastlen, lastlen], axis=1)
        _, last_time_inputs = tf.split(time, [T-lastlen, lastlen], axis=1)
    return last_inputs, last_time_inputs


def temporal_log_time_positional_encoding(inputs, num_units, time_stamp, scale=True,
                                          scope="temporal_log_positional_encoding", reuse=None):
    inputs = tf.cast(inputs, tf.float32)
    N = tf.shape(inputs)[0]
    T = inputs.get_shape().as_list()[1]
    st = tf.tile(tf.expand_dims(time_stamp[:, 0], 1), [1, T])  # [batch_size, max_len]
    ti = time_stamp - st  # [batch_size, max_len]
    ti = tf.log(ti+1)  # natural logarithm to deal with the skewed dist.
    ti = tf.tile(tf.expand_dims(ti, 2), [1, 1, num_units]) # [batch_size, max_len, num_units]
    ti = tf.cast(ti, tf.float32)

    with tf.variable_scope(scope, reuse=reuse):
        # First part of the PE function: sin and cos argument
        range_tensor = tf.range(num_units)
        mod_tensor = tf.mod(range_tensor, 2*tf.ones_like(range_tensor))
        expnt = range_tensor - mod_tensor
        expnt = tf.cast(expnt/num_units, tf.float32)
        base = tf.pow(20.0 * tf.ones_like(expnt, dtype=tf.float32), expnt)
        base = tf.cast(base, tf.float32)
        base = tf.expand_dims(tf.expand_dims(base, 0), 0)
        base = tf.tile(base, [N, T, 1])
        position_enc = ti / base

        # # Second part, apply the cosine to even columns and sin to odds.
        pos_sin = tf.sin(position_enc)
        pos_cos = tf.cos(position_enc)
        pos_ind = mod_tensor
        pos_ind = tf.tile(tf.expand_dims(pos_ind, 0), [T, 1])
        pos_ind = tf.tile(tf.expand_dims(pos_ind, 0), [N, 1, 1])
        pos_sin_ind = tf.cast(1-pos_ind, dtype=tf.float32)
        pos_cos_ind = tf.cast(pos_ind, dtype=tf.float32)
        position_enc = tf.multiply(pos_sin, pos_sin_ind) + tf.multiply(pos_cos, pos_cos_ind)

        outputs = position_enc

        if scale:
            outputs = outputs * num_units ** 0.5

        return outputs


def relative_multihead_attention(queries, keys, pos_enc=None, u_vec=None, v_vec=None,
                                 num_units=None, num_heads=4, causality=False,
                                 scope="relative_multihead_attention", reuse=None):
    """
    Compute the relative attention to fuse the temporal position information
    Inspired by relative position encoding in Transformer-XL
    :param queries: the input embedding of the token seq
    :param keys: the input embedding of the token seq
    :param u_vec:
    :param v_vec:
    :param num_units:
    :param num_heads:
    :param causality:
    :param scope:
    :param reuse:
    :param pos_enc: the (temporal) position encoding
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        queries = tf.cast(queries, tf.float32)
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        N = tf.shape(queries)[0]
        T = tf.shape(queries)[-2]
        C = num_units

        u_vec = tf.tile(u_vec, [T, 1])  # [max_len, num_units]
        v_vec = tf.tile(v_vec, [T, 1])  # [max_len, num_nuits]
        u_vec = tf.tile(tf.expand_dims(u_vec, 0), [N, 1, 1])  # [batch_size, max_len, num_units]
        v_vec = tf.tile(tf.expand_dims(v_vec, 0), [N, 1, 1])  # [batch_size, max_len, num_units]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        PE = tf.layers.dense(pos_enc,
                             num_units,
                             activation=tf.nn.relu,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
                             # [batch_size, max_len, num_units]

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        PE_ = tf.concat(tf.split(PE, num_heads, axis=2), axis=0)  # [num_heads*batch_size, max_len, num_units/num_heads]
        u_vec_ = tf.concat(tf.split(u_vec, num_heads, axis=2), axis=0)  # [num_heads*batch_size, max_len, num_units/num_heads]
        v_vec_ = tf.concat(tf.split(v_vec, num_heads, axis=2), axis=0)  # [num_heads*batch_size, max_len, num_units/num_heads]

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Relative temporal attention
        outputs += tf.matmul(Q_, tf.transpose(PE_, [0, 2, 1]))
        outputs += tf.matmul(u_vec_, tf.transpose(K_, [0, 2, 1]))
        outputs += tf.matmul(v_vec_, tf.transpose(PE_, [0, 2, 1]))

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys + pos_enc, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            # Causality=True #########
            # tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries+pos_enc, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        outputs2 = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs2 = tf.concat(tf.split(outputs2, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # post attention weighting
        outputs2 = tf.layers.dense(outputs2, units=num_units, use_bias=False)

    return outputs2


class PeriodicSelfAttention:
    def __init__(self, hp):
        self.graph = tf.Graph()
        self.vocab_size = hp.vocab_size
        self.num_unit = hp.hidden_units
        self.maxlen = hp.maxlen
        self.daily_periodic_len = hp.daily_period_len
        self.weekly_periodic_len = hp.weekly_period_len
        self.portrait_vec_len = hp.portrait_vec_len
        self.class_num = hp.output_unit
        self.dropout_rate = hp.dropout_rate
        self.num_block = hp.num_blocks
        self.num_head = hp.num_heads

        self.hp = hp

    def forward(self, x_input, time_input, portrait_input, role_id_seq,
                is_training=True, scope='forward'):
        with tf.variable_scope(scope, reuse=None):
            N = tf.shape(x_input)[0]

            # recent embedding
            x_embed = embedding(x_input, vocab_size=self.vocab_size, num_units=self.num_unit, scale=False,
                                scope='input_embed')

            # get lastlen embedding, time
            last_emb, last_time = get_last_emb(x_embed, time_input, lastlen=self.hp.lastlen, maxlen=self.maxlen)

            # behavior decoder
            with tf.variable_scope("behavior_decoder"):
                temporal_log_pe = temporal_log_time_positional_encoding(last_emb, self.hp.hidden_units, last_time)
                enc = last_emb
                # Initialization for decoupled multihead attention module
                u_vec = tf.get_variable('u_vec',
                                        dtype=tf.float32,
                                        shape=[1, self.num_unit],
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        )
                v_vec = tf.get_variable('v_vec',
                                        dtype=tf.float32,
                                        shape=[1, self.num_unit],
                                        initializer=tf.contrib.layers.xavier_initializer(),
                                        )
                for i in range(self.hp.num_decoder_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        enc = self.residual_de_attention_block(queries=enc, keys=enc, time_encoding=temporal_log_pe,
                                                               u_vec=u_vec, v_vec=v_vec, is_training=is_training)
                dec_behavior = enc

            dec = dec_behavior
            # meta_data fusion with residual mlp
            # todo: let inputs of portrait_input placeholder be zeros
            # portrait_input = tf.zeros_like(portrait_input, tf.float32)
            dec_mlp = self.meta_fusion(dec, portrait_input, is_training=is_training)

            # output
            logits = tf.layers.dense(dec_mlp, units=self.class_num, use_bias=False)
        return logits

    def residual_de_attention_block(self, queries, keys, time_encoding, u_vec, v_vec, scope='decoupled_residual_block',
                                    reuse=None, is_training=False):
        with tf.variable_scope(scope, reuse=reuse):
            enc_q = queries
            enc_k = keys
            enc_a = relative_multihead_attention(queries=normalize(enc_q),
                                                 keys=normalize(enc_k),
                                                 num_units=self.hp.hidden_units,
                                                 num_heads=self.num_head,
                                                 pos_enc=time_encoding,
                                                 u_vec=u_vec,
                                                 v_vec=v_vec,
                                                 causality=True,
                                                 reuse=tf.AUTO_REUSE)
            enc_a = tf.layers.dropout(enc_a, rate=self.hp.dropout_rate,
                                      training=tf.convert_to_tensor(is_training))
            enc_b = normalize(enc_q + enc_a)
            enc_b = feedforward_sparse(enc_b, num_units=[4 * self.hp.hidden_units, self.hp.hidden_units])
            enc_b = tf.layers.dropout(enc_b, rate=self.hp.dropout_rate,
                                      training=tf.convert_to_tensor(is_training))
            enc_q += enc_a + enc_b
            outputs = enc_q
        return outputs

    def meta_fusion(self, inputs, meta_data, is_training=False, scope='meta_fusion', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            # flatten
            inputs = tf.reshape(inputs, [-1, self.hp.lastlen * self.num_unit])
            input_flat = normalize(inputs)

            # concat meta_data
            input_concat = tf.concat([input_flat, meta_data], axis=1)

            # feedforward
            input_mlp = tf.layers.dense(input_concat, units=1024, activation=tf.nn.elu)
            input_mlp = tf.layers.dropout(input_mlp, rate=self.hp.dropout_rate, training=tf.convert_to_tensor(is_training))
            input_mlp = tf.layers.dense(input_mlp, units=self.hp.lastlen * self.hp.hidden_units, activation=None)
            input_mlp = tf.layers.dropout(input_mlp, rate=self.hp.dropout_rate, training=tf.convert_to_tensor(is_training))

            # Residual connection and Normalize
            input_mlp += inputs
            input_mlp = normalize(input_mlp)
            return input_mlp

    def get_loss(self, x_input, time_input, portrait_input, role_id_seq,
                 y_output, is_training):
        with self.graph.as_default():
            logits = self.forward(x_input=x_input,
                                  time_input=time_input,
                                  portrait_input=portrait_input,
                                  role_id_seq=role_id_seq,
                                  is_training=is_training,
                                  )
            cost = cross_entropy(logits=logits, labels=y_output, class_num=self.class_num, isSmoothing=True)
        return logits, cost
