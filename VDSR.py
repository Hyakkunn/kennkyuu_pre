def VDSR(x, hidden_num, repeat_num, data_format, use_norm, name='VDSR',
         k=3, train=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as vs:
        for i in range(repeat_num-1):
            x = conv2d(x, hidden_num, data_format, k=k, s=1, act=tf.nn.relu)
            if use_norm:
                x = batch_norm(x, train, data_format, act=tf.nn.relu)
        x = conv2d(x, 1, data_format, k=k, s=1)
        if use_norm:
            x = batch_norm(x, train, data_format)
        out = tf.nn.relu(x)
    variables = tf.contrib.framework.get_variables(vs)
    return out, variables