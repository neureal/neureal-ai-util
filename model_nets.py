from collections import OrderedDict
import numpy as np
import tensorflow as tf
import model_util as util


class ArchFull(tf.keras.Model):
    def __init__(self, name, inputs, opt_spec, stats_spec, spec_in, spec_out, latent_spec, obs_latent=False, net_blocks=1, net_lstm=False, net_attn=None, num_heads=1, memory_size=None, aug_data_pos=False):
        super(ArchFull, self).__init__(name=name)
        self.inp = In(latent_spec, spec_in, obs_latent=obs_latent, net_attn_io=net_attn['io'], num_heads=num_heads, aug_data_pos=aug_data_pos)
        self.net = Net(latent_spec, obs_latent=obs_latent, net_blocks=net_blocks, net_lstm=net_lstm, net_attn=net_attn['net'], net_attn_io=net_attn['io'], net_attn_ar=net_attn['ar'], num_heads=num_heads, memory_size=memory_size)
        self.out = Out(latent_spec, spec_out, net_attn_io=net_attn['out'], num_heads=num_heads)
        self.dist = self.out.dist
        self.inp_subs = [sm for sm in self.inp.submodules if sm.name.startswith(('dense_','mlp_','attn_','lstm_'))]
        self.net_subs = [sm for sm in self.net.submodules if sm.name.startswith(('dense_','mlp_','attn_','lstm_'))]
        self.out_subs = [sm for sm in self.out.submodules if sm.name.startswith(('dense_','mlp_','attn_','lstm_'))]

        self.optimizer = OrderedDict()
        for spec in opt_spec: self.optimizer[spec['name']] = util.optimizer(name, spec)
        stats = OrderedDict()
        for spec in stats_spec: stats[spec['name']] = {'b1':tf.constant(spec['b1'],spec['dtype']), 'b1_n':tf.constant(1-spec['b1'],spec['dtype']), 'b2':tf.constant(spec['b2'],spec['dtype']), 'b2_n':tf.constant(1-spec['b2'],spec['dtype']), 'dtype':spec['dtype'],
            'ma':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/ma'.format(name,spec['name'])), 'ema':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/ema'.format(name,spec['name'])),
            'iter':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/iter'.format(name,spec['name'])), 'sum':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/sum'.format(name,spec['name'])),}
        self.stats = stats

        self(inputs); self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        arch_in = "{}D{}".format(('Aio+' if net_attn['io'] else ''), latent_spec['inp'])
        arch_net = "{:02d}{}{}D{}".format(net_blocks, ('AT+' if net_attn['net'] else ''), ('LS+' if net_lstm else ''), latent_spec['midp'])
        arch_out = "{}D{}".format(('Aio+' if net_attn['out'] else ''), latent_spec['outp'])
        self.arch_desc_file = "{}[in{}_net{}_out{}_{}]".format(name, arch_in, arch_net, arch_out, self.inp.arch_lat)
        self.arch_desc = "{}[in{}_net{}_out{}_{}]".format(name, arch_in+'-'+self.inp.arch_in, arch_net, arch_out+'-'+self.out.arch_out, self.inp.arch_lat)

    def reset_states(self, use_img=False):
        if self.inp.net_attn_io2: self.inp.layer_attn_io2._mem_score.assign(self.inp.layer_attn_io2._mem_score_zero)
        for layer in self.net.layer_attn: layer.reset_states(use_img=use_img)
        for layer in self.net.layer_lstm: layer.reset_states()
    def call(self, inputs, store_memory=True, use_img=False, store_real=False, training=None):
        out = self.inp(inputs, training=training)
        out = self.net(out, store_memory=store_memory, use_img=use_img, store_real=store_real, training=training)
        dist = self.net.dist(out); out = dist.sample()
        out = self.out(out, training=training)

        for out_n in out:
            isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out_n), tf.math.is_inf(out_n)))
            if isinfnan > 0: tf.print(self.name, 'net out:', out)
        return out

class ArchTrans(tf.keras.Model):
    def __init__(self, name, inputs, opt_spec, stats_spec, spec_in, latent_spec, obs_latent=False, net_blocks=1, net_lstm=False, net_attn=None, num_heads=1, memory_size=None, aug_data_pos=False):
        super(ArchTrans, self).__init__(name=name)
        self.inp = In(latent_spec, spec_in, obs_latent=obs_latent, net_attn_io=net_attn['io'], num_heads=num_heads, aug_data_pos=aug_data_pos)
        self.net = Net(latent_spec, obs_latent=obs_latent, net_blocks=net_blocks, net_lstm=net_lstm, net_attn=net_attn['net'], net_attn_io=net_attn['io'], net_attn_ar=net_attn['ar'], num_heads=num_heads, memory_size=memory_size)
        self.dist = self.net.dist
        self.inp_subs = [sm for sm in self.inp.submodules if sm.name.startswith(('dense_','mlp_','attn_','lstm_'))]
        self.net_subs = [sm for sm in self.net.submodules if sm.name.startswith(('dense_','mlp_','attn_','lstm_'))]

        self.optimizer = OrderedDict()
        for spec in opt_spec: self.optimizer[spec['name']] = util.optimizer(name, spec)
        stats = OrderedDict()
        for spec in stats_spec: stats[spec['name']] = {'b1':tf.constant(spec['b1'],spec['dtype']), 'b1_n':tf.constant(1-spec['b1'],spec['dtype']), 'b2':tf.constant(spec['b2'],spec['dtype']), 'b2_n':tf.constant(1-spec['b2'],spec['dtype']), 'dtype':spec['dtype'],
            'ma':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/ma'.format(name,spec['name'])), 'ema':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/ema'.format(name,spec['name'])),
            'iter':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/iter'.format(name,spec['name'])), 'sum':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/sum'.format(name,spec['name'])),}
        self.stats = stats

        self(inputs); self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        arch_in = "{}D{}".format(('Aio+' if net_attn['io'] else ''), latent_spec['inp'])
        arch_net = "{:02d}{}{}D{}".format(net_blocks, ('AT+' if net_attn['net'] else ''), ('LS+' if net_lstm else ''), latent_spec['midp'])
        self.arch_desc_file = "{}[in{}_net{}_{}]".format(name, arch_in, arch_net, self.net.arch_lat)
        self.arch_desc = "{}[in{}_net{}_{}]".format(name, arch_in+'-'+self.inp.arch_in, arch_net, self.net.arch_lat)

    def reset_states(self, use_img=False):
        if self.inp.net_attn_io2: self.inp.layer_attn_io2._mem_score.assign(self.inp.layer_attn_io2._mem_score_zero)
        for layer in self.net.layer_attn: layer.reset_states(use_img=use_img)
        for layer in self.net.layer_lstm: layer.reset_states()
    def call(self, inputs, store_memory=True, use_img=False, store_real=False, training=None):
        out = self.inp(inputs, training=training)
        out = self.net(out, store_memory=store_memory, use_img=use_img, store_real=store_real, training=training)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print(self.name, 'net out:', out)
        return out

class ArchRep(tf.keras.Model):
    def __init__(self, name, inputs, opt_spec, stats_spec, spec_in, latent_spec, net_attn=None, num_heads=1, aug_data_pos=False):
        super(ArchRep, self).__init__(name=name)
        self.inp = In(latent_spec, spec_in, net_attn_io=net_attn['io'], num_heads=num_heads, aug_data_pos=aug_data_pos)
        self.inp_subs = [sm for sm in self.inp.submodules if sm.name.startswith(('dense_','mlp_','attn_','lstm_'))]

        self.optimizer = OrderedDict()
        for spec in opt_spec: self.optimizer[spec['name']] = util.optimizer(name, spec)
        stats = OrderedDict()
        for spec in stats_spec: stats[spec['name']] = {'b1':tf.constant(spec['b1'],spec['dtype']), 'b1_n':tf.constant(1-spec['b1'],spec['dtype']), 'b2':tf.constant(spec['b2'],spec['dtype']), 'b2_n':tf.constant(1-spec['b2'],spec['dtype']), 'dtype':spec['dtype'],
            'ma':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/ma'.format(name,spec['name'])), 'ema':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/ema'.format(name,spec['name'])),
            'iter':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/iter'.format(name,spec['name'])), 'sum':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/sum'.format(name,spec['name'])),}
        self.stats = stats

        self(inputs); self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        arch_in = "{}D{}".format(('Aio+' if net_attn['io'] else ''), latent_spec['inp'])
        self.arch_desc_file = "{}[in{}_{}]".format(name, arch_in, self.inp.arch_lat)
        self.arch_desc = "{}[in{}_{}]".format(name, arch_in+'-'+self.inp.arch_in, self.inp.arch_lat)

    def reset_states(self, use_img=False):
        if self.inp.net_attn_io2: self.inp.layer_attn_io2._mem_score.assign(self.inp.layer_attn_io2._mem_score_zero)
    def call(self, inputs, training=None):
        out = self.inp(inputs, training=training)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print(self.name, 'net out:', out)
        return out

class ArchGen(tf.keras.Model):
    def __init__(self, name, inputs, opt_spec, stats_spec, spec_out, latent_spec, net_blocks=1, net_lstm=False, net_attn=None, num_heads=1, memory_size=None):
        super(ArchGen, self).__init__(name=name)
        # num_latents, latent_size = latent_spec['num_latents'], latent_spec['latent_size']
        # # self.layer_attn_in = util.MultiHeadAttention(latent_size=latent_size, num_heads=num_heads, norm=False, residual=False, cross_type=1, num_latents=num_latents, channels=latent_size, memory_size=memory_size, name='attn_in')
        # self.layer_attn_in = util.MultiHeadAttention(latent_size=latent_size, num_heads=num_heads, memory_size=memory_size, residual=True, name='attn_in')
        # self.layer_mlp_in = util.MLPBlock(hidden_size=256, latent_size=latent_size, evo=None, residual=True, name='mlp_in')

        self.net = Net(latent_spec, obs_latent=False, net_blocks=net_blocks, net_lstm=net_lstm, net_attn=net_attn['net'], net_attn_io=net_attn['io'], net_attn_ar=net_attn['ar'], num_heads=num_heads, memory_size=memory_size)
        self.out = Out(latent_spec, spec_out, net_attn_io=net_attn['out'], num_heads=num_heads)
        self.dist = self.out.dist
        self.net_subs = [sm for sm in self.net.submodules if sm.name.startswith(('dense_','mlp_','attn_','lstm_'))]
        self.out_subs = [sm for sm in self.out.submodules if sm.name.startswith(('dense_','mlp_','attn_','lstm_'))]

        self.optimizer = OrderedDict()
        for spec in opt_spec: self.optimizer[spec['name']] = util.optimizer(name, spec)
        stats = OrderedDict()
        for spec in stats_spec: stats[spec['name']] = {'b1':tf.constant(spec['b1'],spec['dtype']), 'b1_n':tf.constant(1-spec['b1'],spec['dtype']), 'b2':tf.constant(spec['b2'],spec['dtype']), 'b2_n':tf.constant(1-spec['b2'],spec['dtype']), 'dtype':spec['dtype'],
            'ma':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/ma'.format(name,spec['name'])), 'ema':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/ema'.format(name,spec['name'])),
            'iter':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/iter'.format(name,spec['name'])), 'sum':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/sum'.format(name,spec['name'])),}
        self.stats = stats

        self(inputs); self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        arch_net = "{:02d}{}{}D{}".format(net_blocks, ('AT+' if net_attn['net'] else ''), ('LS+' if net_lstm else ''), latent_spec['midp'])
        arch_out = "{}D{}".format(('Aio+' if net_attn['out'] else ''), latent_spec['outp'])
        self.arch_desc_file = "{}[net{}_out{}_{}]".format(name, arch_net, arch_out, self.net.arch_lat)
        self.arch_desc = "{}[net{}_out{}_{}]".format(name, arch_net, arch_out+'-'+self.out.arch_out, self.net.arch_lat)

    def reset_states(self, use_img=False):
        # self.layer_attn_in.reset_states(use_img=use_img)
        for layer in self.net.layer_attn: layer.reset_states(use_img=use_img)
        for layer in self.net.layer_lstm: layer.reset_states()
    def call(self, inputs, store_memory=True, use_img=False, store_real=False, batch_size=None, training=None):
        # # out = self.layer_attn_in(inputs, use_img=True, store_real=True)
        # # out = self.layer_attn_in(inputs)
        # out = tf.squeeze(self.layer_attn_in(tf.expand_dims(inputs, axis=0), use_img=True, store_real=True), axis=0)
        # out = self.layer_mlp_in(out)

        out = self.net(inputs, store_memory=store_memory, use_img=use_img, store_real=store_real, batch_size=batch_size, training=training)
        dist = self.net.dist(out); out = dist.sample()
        out = self.out(out, batch_size=batch_size, training=training)

        for out_n in out:
            isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out_n), tf.math.is_inf(out_n)))
            if isinfnan > 0: tf.print(self.name, 'net out:', out)
        return out

class ArchNet(tf.keras.Model):
    def __init__(self, name, inputs, opt_spec, stats_spec, latent_spec, net_blocks=1, net_lstm=False, net_attn=None, num_heads=1, memory_size=None):
        super(ArchNet, self).__init__(name=name)
        self.net = Net(latent_spec, obs_latent=False, net_blocks=net_blocks, net_lstm=net_lstm, net_attn=net_attn['net'], net_attn_io=net_attn['io'], net_attn_ar=net_attn['ar'], num_heads=num_heads, memory_size=memory_size)
        self.dist = self.net.dist
        self.net_subs = [sm for sm in self.net.submodules if sm.name.startswith(('dense_','mlp_','attn_','lstm_'))]

        self.optimizer = OrderedDict()
        for spec in opt_spec: self.optimizer[spec['name']] = util.optimizer(name, spec)
        stats = OrderedDict()
        for spec in stats_spec: stats[spec['name']] = {'b1':tf.constant(spec['b1'],spec['dtype']), 'b1_n':tf.constant(1-spec['b1'],spec['dtype']), 'b2':tf.constant(spec['b2'],spec['dtype']), 'b2_n':tf.constant(1-spec['b2'],spec['dtype']), 'dtype':spec['dtype'],
            'ma':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/ma'.format(name,spec['name'])), 'ema':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/ema'.format(name,spec['name'])),
            'iter':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/iter'.format(name,spec['name'])), 'sum':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/sum'.format(name,spec['name'])),}
        self.stats = stats

        self(inputs); self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        arch_net = "{:02d}{}{}D{}".format(net_blocks, ('AT+' if net_attn['net'] else ''), ('LS+' if net_lstm else ''), latent_spec['midp'])
        self.arch_desc_file = "{}[net{}_{}]".format(name, arch_net, self.net.arch_lat); self.arch_desc = self.arch_desc_file

    def reset_states(self, use_img=False):
        for layer in self.net.layer_attn: layer.reset_states(use_img=use_img)
        for layer in self.net.layer_lstm: layer.reset_states()
    def call(self, inputs, store_memory=True, use_img=False, store_real=False, batch_size=None, training=None):
        out = self.net(inputs, store_memory=store_memory, use_img=use_img, store_real=store_real, batch_size=batch_size, training=training)

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print(self.name, 'net out:', out)
        return out

class ArchAR(tf.keras.Model):
    def __init__(self, name, inputs, opt_spec, stats_spec, latent_spec, net_blocks=1, net_lstm=False, net_attn=None, num_heads=1, memory_size=None, mem_img_size=0):
        super(ArchAR, self).__init__(name=name)
        num_latents, latent_size = latent_spec['num_latents'], latent_spec['latent_size']
        self.net_attn_io, self.out_shape = net_attn['out'], (-1, num_latents, latent_size)

        self.net = Net(latent_spec, obs_latent=False, net_blocks=net_blocks, net_lstm=net_lstm, net_attn=net_attn['net'], net_attn_io=net_attn['io'], net_attn_ar=net_attn['ar'], num_heads=num_heads, memory_size=memory_size)
        self.dist = self.net.dist
        self.net_subs = [sm for sm in self.net.submodules if sm.name.startswith(('dense_','mlp_','attn_','lstm_'))]
        if net_attn['out']: self.layer_attn_out = util.MultiHeadAttention(latent_size=latent_size, num_heads=num_heads, norm=False, residual=False, cross_type=1, num_latents=mem_img_size*num_latents, channels=latent_size, name='ar_attn_out')
        else: self.layer_dense_out = tf.keras.layers.Dense(mem_img_size*num_latents*latent_size, name='ar_dense_out')
        # self.layer_out_logits = util.MLPBlock(hidden_size=512, latent_size=latent_size, evo=64, residual=False, name='ar_mlp_out_logits') # _trans-logits

        self.optimizer = OrderedDict()
        for spec in opt_spec: self.optimizer[spec['name']] = util.optimizer(name, spec)
        stats = OrderedDict()
        for spec in stats_spec: stats[spec['name']] = {'b1':tf.constant(spec['b1'],spec['dtype']), 'b1_n':tf.constant(1-spec['b1'],spec['dtype']), 'b2':tf.constant(spec['b2'],spec['dtype']), 'b2_n':tf.constant(1-spec['b2'],spec['dtype']), 'dtype':spec['dtype'],
            'ma':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/ma'.format(name,spec['name'])), 'ema':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/ema'.format(name,spec['name'])),
            'iter':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/iter'.format(name,spec['name'])), 'sum':tf.Variable(0, dtype=spec['dtype'], trainable=False, name='{}/stats_{}/sum'.format(name,spec['name'])),}
        self.stats = stats

        self(inputs); self.call = tf.function(self.call, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        arch_net = "{:02d}{}{}D{}".format(net_blocks, ('AT+' if net_attn['net'] else ''), ('LS+' if net_lstm else ''), latent_spec['midp'])
        self.arch_desc_file = "{}[net{}_{}]".format(name, arch_net, self.net.arch_lat); self.arch_desc = self.arch_desc_file

    def reset_states(self, use_img=False):
        for layer in self.net.layer_attn: layer.reset_states(use_img=use_img)
        for layer in self.net.layer_lstm: layer.reset_states()
    def call(self, inputs, store_memory=True, use_img=False, store_real=False, batch_size=None, training=None):
        out = self.net(inputs, store_memory=store_memory, use_img=use_img, store_real=store_real, batch_size=batch_size, training=training)
        if self.net_attn_io: out = self.layer_attn_out(out, batch_size=batch_size)
        else:
            if batch_size is None: batch_size = 1
            out = tf.reshape(out, (batch_size, -1))
            out = self.layer_dense_out(out)
        out = tf.reshape(out, self.out_shape)
        # out = self.layer_out_logits(out) # _trans-logits

        isinfnan = tf.math.count_nonzero(tf.math.logical_or(tf.math.is_nan(out), tf.math.is_inf(out)))
        if isinfnan > 0: tf.print(self.name, 'net out:', out)
        return out



class In(tf.keras.layers.Layer):
    def __init__(self, latent_spec, spec_in, obs_latent=False, net_attn_io=False, num_heads=1, aug_data_pos=False):
        super(In, self).__init__(name='inp')
        inp, evo, latent_size, aio_max_latents = latent_spec['inp'], latent_spec['evo'], latent_spec['latent_size'], latent_spec['max_latents']
        self.latent_size, self.obs_latent, self.net_attn_io, self.num_latents, self.net_attn_io2, self.spec_in = tf.constant(latent_size), obs_latent, net_attn_io, 0, False, spec_in

        self.net_ins = len(spec_in); self.input_names, self.layer_attn_in, self.layer_mlp_in, self.pos_idx_in = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
        for i in range(self.net_ins):
            space_name, input_name, event_size, channels, step_shape, num_latents = spec_in[i]['space_name'], spec_in[i]['name'], spec_in[i]['event_size'], spec_in[i]['channels'], spec_in[i]['step_shape'], spec_in[i]['num_latents']; self.num_latents += num_latents
            if space_name not in self.input_names: self.input_names[space_name], self.layer_attn_in[space_name], self.layer_mlp_in[space_name], self.pos_idx_in[space_name] = 0, [], [], []
            self.input_names[space_name] += 1
            if aug_data_pos and event_size > 1:
                pos_idx = np.indices(step_shape[1:-1].as_list())
                pos_idx = np.moveaxis(pos_idx, 0, -1)
                # pos_idx = pos_idx / (np.max(pos_idx).item() / 2.0) - 1.0
                self.pos_idx_in[space_name] += [tf.constant(pos_idx, dtype=self.compute_dtype)]
                channels += pos_idx.shape[-1]
            else: self.pos_idx_in[space_name] += [None]
            # if aug_data_group and event_size > 1:
            #     # TODO add an MLP at each different scale: (256,2) = (256,2), (64,8), (16,32), (4,128), (512,)   (32,32,5) = (32,32,5), (8,8,80), (2,2,1280), (5120,)
            if net_attn_io and event_size > num_latents:
                self.layer_attn_in[space_name] += [util.MultiHeadAttention(latent_size=latent_size, num_heads=num_heads, norm=False, residual=False, cross_type=1, num_latents=num_latents, channels=latent_size, name='attn_in_{}_{}'.format(space_name, input_name))]
            # if event_size > num_latents: # TODO
            #     if net_attn_io: self.layer_attn_in[space_name] += [util.MultiHeadAttention(latent_size=latent_size, num_heads=num_heads, norm=False, residual=False, cross_type=1, num_latents=num_latents, channels=latent_size, name='attn_in_{}_{}'.format(space_name, input_name))]
            #     else: self.layer_attn_in[space_name] += [tf.keras.layers.Dense(num_latents*latent_size, name='dense_in_{}_{}'.format(space_name, input_name))]
            else: self.layer_attn_in[space_name] += [None]
            self.layer_mlp_in[space_name] += [util.MLPBlock(hidden_size=inp, latent_size=latent_size, evo=evo, residual=False, name='mlp_in_{}_{}'.format(space_name, input_name))]
        if net_attn_io and self.num_latents > aio_max_latents:
            self.net_attn_io2, self.num_latents = True, aio_max_latents
            self.layer_attn_io2 = util.MultiHeadAttention(latent_size=latent_size, num_heads=num_heads, norm=False, residual=False, cross_type=1, num_latents=self.num_latents, channels=latent_size, save_attn_scores=True, name='attn_io2')
        if obs_latent: self.net_ins += 1; self.num_latents += latent_spec['num_latents']
        else: latent_spec.update({'num_latents':self.num_latents})

        self.arch_in, self.arch_lat = "Ï{}{}".format(len(spec_in), ('io2' if self.net_attn_io2 else '')), "L{}{}x{}".format(latent_spec['dist_type'], self.num_latents, latent_size)

    # def call(self, inputs, training=None):
    #     out_accu = tf.zeros((0, self.latent_size), self.compute_dtype)
    #     for input_name in self.input_names.keys():
    #         for i in range(self.input_names[input_name]):
    #             out = tf.cast(inputs[input_name][i], self.compute_dtype)
    #             if tf.math.count_nonzero(out) > 0:
    #                 if self.pos_idx_in[input_name][i] is not None:
    #                     shape = tf.concat([tf.shape(out)[0:1], self.pos_idx_in[input_name][i].shape], axis=0)
    #                     pos_idx = tf.broadcast_to(self.pos_idx_in[input_name][i], shape)
    #                     out = tf.concat([out, pos_idx], axis=-1)
    #                 out = self.layer_mlp_in[input_name][i](out)
    #                 if self.layer_attn_in[input_name][i] is not None:
    #                     out = self.layer_attn_in[input_name][i](out)
    #                     out_accu = tf.concat([out_accu, out], axis=0)
    #                 else:
    #                     out = tf.reshape(out, (-1, self.latent_size))
    #                     out_accu = tf.concat([out_accu, out], axis=0)
    #     if self.obs_latent:
    #         out = tf.cast(inputs['obs'], self.compute_dtype)
    #         out_accu = tf.concat([out_accu, out], axis=0)
    #     if self.net_attn_io2: out_accu = self.layer_attn_io2(out_accu)
    #     return out_accu

    def call(self, inputs, training=None):
        out_accu, out_accu_i = [None]*self.net_ins, 0
        for input_name in self.input_names.keys():
            for i in range(self.input_names[input_name]):
                out = tf.cast(inputs[input_name][i], self.compute_dtype)
                if self.pos_idx_in[input_name][i] is not None:
                    shape = tf.concat([tf.shape(out)[0:1], self.pos_idx_in[input_name][i].shape], axis=0)
                    pos_idx = tf.broadcast_to(self.pos_idx_in[input_name][i], shape)
                    out = tf.concat([out, pos_idx], axis=-1)
                out = self.layer_mlp_in[input_name][i](out)
                if self.layer_attn_in[input_name][i] is not None: out_accu[out_accu_i] = self.layer_attn_in[input_name][i](out)
                else: out_accu[out_accu_i] = tf.reshape(out, (-1, self.latent_size))
                # if not self.net_attn_io: out = tf.reshape(out, (1, -1)) # TODO
                # if self.layer_attn_in[input_name][i] is not None: out = self.layer_attn_in[input_name][i](out)
                # out_accu[out_accu_i] = tf.reshape(out, (-1, self.latent_size))
                out_accu_i += 1
        if self.obs_latent: out_accu[-1] = tf.cast(inputs['obs'], self.compute_dtype)
        # out = tf.math.add_n(out_accu) # out = tf.math.accumulate_n(out_accu)
        out = tf.concat(out_accu, axis=0)
        if self.net_attn_io2: out = self.layer_attn_io2(out)
        return out

class Net(tf.keras.layers.Layer):
    def __init__(self, latent_spec, obs_latent=False, net_blocks=1, net_lstm=False, net_attn=False, net_attn_io=False, net_attn_ar=False, num_heads=1, memory_size=None):
        super(Net, self).__init__(name='net')
        outp, midp, evo, latent_size = latent_spec['outp'], latent_spec['midp'], latent_spec['evo'], latent_spec['latent_size']
        self.obs_latent, self.net_blocks, self.net_attn, self.net_lstm, self.net_attn_io = obs_latent, net_blocks, net_attn, net_lstm, net_attn_io

        self.layer_attn, self.layer_lstm, self.layer_dense, self._residual_amt, self.layer_mlp = [], [], [], [], []
        for i in range(net_blocks):
            if net_attn:
                mem_active = None if net_attn_ar and i > 0 else memory_size
                self.layer_attn += [util.MultiHeadAttention(latent_size=latent_size, num_heads=num_heads, memory_size=mem_active, residual=True, name='attn_{:02d}'.format(i))]
            if net_lstm:
                # self.layer_lstm += [tf.keras.layers.LSTM(latent_size, activation=util.EvoNormS0(evo), use_bias=False, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i))]
                self.layer_lstm += [tf.keras.layers.LSTM(latent_size, return_sequences=True, stateful=True, name='lstm_{:02d}'.format(i))] # CUDA
                self.layer_dense += [tf.keras.layers.Dense(latent_size, name='dense_{:02d}'.format(i))]
                self._residual_amt += [tf.Variable(0.0, dtype=self.compute_dtype, trainable=True, name='residual_{:02d}'.format(i))] # ReZero
            self.layer_mlp += [util.MLPBlock(hidden_size=midp, latent_size=latent_size, evo=evo, residual=True, name='mlp_{:02d}'.format(i))]

        self.num_latents, self.dist_out = latent_spec['num_latents'], latent_spec['dist_type'] != 'd'
        params_size, self.dist = util.distribution(latent_spec)
        if obs_latent:
            if net_attn_io: self.layer_attn_out = util.MultiHeadAttention(latent_size=latent_size, num_heads=num_heads, norm=False, residual=False, cross_type=1, num_latents=self.num_latents, channels=latent_size, name='attn_out')
            else: self.layer_dense_out = tf.keras.layers.Dense(self.num_latents*latent_size, name='dense_out')
        if self.dist_out: self.layer_out_logits = util.MLPBlock(hidden_size=outp, latent_size=params_size, evo=evo, residual=False, name='mlp_out_logits')

        self.arch_lat = "L{}{}x{}".format(latent_spec['dist_type'], self.num_latents, latent_size)

    def call(self, inputs, store_memory=True, use_img=False, store_real=False, batch_size=None, training=None):
        out = tf.cast(inputs, self.compute_dtype)
        if batch_size is None: out = tf.expand_dims(out, axis=0)
        for i in range(self.net_blocks):
            if self.net_attn: out = self.layer_attn[i](out, auto_mask=training, store_memory=store_memory, use_img=use_img, store_real=store_real)
            if self.net_lstm:
                outL = self.layer_lstm[i](out, training=training)
                outL = self.layer_dense[i](outL)
                out = out + outL * self._residual_amt[i] # Rezero
            out = self.layer_mlp[i](out)
        if batch_size is None: out = tf.squeeze(out, axis=0)
        if self.obs_latent:
            if self.net_attn_io: out = self.layer_attn_out(out)
            else: out = tf.reshape(self.layer_dense_out(tf.reshape(out, (1, -1))), (self.num_latents, -1))
        if self.dist_out: out = self.layer_out_logits(out)
        return out

class Out(tf.keras.layers.Layer):
    def __init__(self, latent_spec, spec_out, net_attn_io=False, num_heads=1):
        super(Out, self).__init__(name='out')
        outp, evo, num_latents, latent_size = latent_spec['outp'], latent_spec['evo'], latent_spec['num_latents'], latent_spec['latent_size']
        self.net_attn_io = (net_attn_io if num_latents > 1 else False)

        self.net_outs = len(spec_out); no = self.net_outs
        self.event_size, self.step_shape, params_size, self.dist, self.layer_attn_out, self.layer_dense_out, self.layer_out_logits, self.seq_size_out, self._dense_scale, self.arch_out = [None]*no, [None]*no, [None]*no, [None]*no, [None]*no, [None]*no, [None]*no, [None]*no, [None]*no, "Ö"
        for i in range(self.net_outs):
            space_name, output_name, dist_type, num_components, event_size, step_shape = spec_out[i]['space_name'], spec_out[i]['name'], spec_out[i]['dist_type'], spec_out[i]['num_components'], spec_out[i]['event_size'], spec_out[i]['step_shape']
            seq_size_out = 1 if 'seq_size_out' not in spec_out[i] or spec_out[i]['seq_size_out'] is None else spec_out[i]['seq_size_out']

            params_size[i], self.dist[i] = util.distribution(spec_out[i])
            if self.net_attn_io: self.layer_attn_out[i] = util.MultiHeadAttention(latent_size=latent_size, num_heads=num_heads, memory_size=None, norm=False, residual=False, cross_type=1, num_latents=seq_size_out*event_size, channels=latent_size, name='attn_out_{}_{}'.format(space_name, output_name))
            else: self.layer_dense_out[i] = tf.keras.layers.Dense(seq_size_out*event_size*latent_size, name='dense_out_{}_{}'.format(space_name, output_name))
            self.layer_out_logits[i] = util.MLPBlock(hidden_size=outp, latent_size=params_size[i], evo=evo, residual=False, name='mlp_out_logits_{}_{}'.format(space_name, output_name))

            self.event_size[i], self.step_shape[i], self.seq_size_out[i] = tf.constant(event_size), list(step_shape[1:-1])+[latent_size], seq_size_out
            self._dense_scale[i] = (self.event_size[i]*self.seq_size_out[i]) / num_latents
            self.arch_out += "{}{}".format(dist_type, num_components)

    def call(self, inputs, batch_size=None, training=None):
        if batch_size is None: batch_size = 1
        out = tf.cast(inputs, self.compute_dtype)
        if not self.net_attn_io: out = tf.reshape(out, (batch_size, -1))
        out_logits = [None]*self.net_outs
        for i in range(self.net_outs):
            if self.net_attn_io: out_logits[i] = self.layer_attn_out[i](out, batch_size=batch_size) # out_logits[i] = self.layer_attn_out[i](out, num_latents=batch_size*self.event_size[i])
            else:
                out_logits[i] = self.layer_dense_out[i](out)
                out_logits[i] = out_logits[i] * self._dense_scale[i]
            # out_logits[i] = tf.reshape(out_logits[i], tf.concat([tf.reshape(batch_size,(1,)), self.step_shape[i]], 0))
            out_logits[i] = tf.reshape(out_logits[i], [batch_size*self.seq_size_out[i]] + self.step_shape[i])
            out_logits[i] = self.layer_out_logits[i](out_logits[i])
        return out_logits
