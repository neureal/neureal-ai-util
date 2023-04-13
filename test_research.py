import time, os
curdir = os.path.expanduser("~")
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
np.random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0,1,2,3
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
# tf.config.run_functions_eagerly(True)
# tf.random.set_seed(0)
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import model_util as util, model_nets as nets

physical_devices_gpu = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices_gpu)): tf.config.experimental.set_memory_growth(physical_devices_gpu[i], True)

compute_dtype = tf.keras.backend.floatx() # string
float_min, float_max = tf.constant(tf.dtypes.as_dtype(compute_dtype).min, compute_dtype), tf.constant(tf.dtypes.as_dtype(compute_dtype).max, compute_dtype)
float_maxroot, float_eps = tf.constant(tf.math.sqrt(float_max), compute_dtype), tf.constant(tf.experimental.numpy.finfo(compute_dtype).eps, compute_dtype)
float_min_prob = tf.constant(tf.math.log(float_eps), compute_dtype)
float_eps_max = tf.constant(1.0 / float_eps, compute_dtype)
tf.keras.backend.set_epsilon(float_eps) # 1e-7 default

num_cats, scale, ylim, trainS, testS, obs_key, action_key, train_actions, test_actions, memory_size, is_image, device = 0, 1, 10, 'train', 'test', 'image', 'label', tf.constant([[0]]), tf.constant([[0]]), None, False, 0
@tfds.decode.make_decoder()
def tfds_scale(serialized_image, features, scale):
    return tf.io.decode_jpeg(serialized_image, ratio=scale)
@tfds.decode.make_decoder(output_dtype=tf.int32)
def tfds_unicode(text, features):
    return tf.strings.unicode_decode(text, 'UTF-8')
@tfds.decode.make_decoder(output_dtype=tf.uint8)
def tfds_bytes(text, features):
    return tf.io.decode_raw(text, tf.uint8)


# predict category
def run1(num_steps, obs_data, actions, net, train=True):
    print("tracing run1"); tf.print("running run1")
    metric_loss = tf.TensorArray(compute_dtype, size=1, dynamic_size=True)
    metric_ma_loss = tf.TensorArray(compute_dtype, size=1, dynamic_size=True)
    metric_snr = tf.TensorArray(compute_dtype, size=1, dynamic_size=True)
    metric_std = tf.TensorArray(compute_dtype, size=1, dynamic_size=True)

    for step in tf.range(num_steps):
        inputs = {'obs':[obs_data[step:step+1]]}
        targets = actions[step:step+1]

        with tf.GradientTape() as tape:
            logits = net(inputs); dist = net.dist[0](logits[0])
            loss = util.loss_likelihood(dist, targets)
        if train:
            gradients = tape.gradient(loss, net.trainable_variables)
            net.optimizer['net'].apply_gradients(zip(gradients, net.trainable_variables))
        loss = tf.squeeze(loss)
        metric_loss = metric_loss.write(step, loss)

        util.stats_update(net.stats['loss'], loss)
        _, ma_loss, _, snr_loss, std_loss = util.stats_get(net.stats['loss'])
        metric_ma_loss = metric_ma_loss.write(step, ma_loss); metric_snr = metric_snr.write(step, snr_loss); metric_std = metric_std.write(step, std_loss)

        if train: net.optimizer['net'].learning_rate = learn_rate * snr_loss**np.e # **np.e # _lr-snre

    metric_loss, metric_ma_loss, metric_snr, metric_std = metric_loss.stack(), metric_ma_loss.stack(), metric_snr.stack(), metric_std.stack()
    return metric_loss, metric_ma_loss, metric_snr, metric_std

# reconstruct
def run2(num_steps, obs_data, actions, net, train=True):
    print("tracing run2"); tf.print("running run2")
    metric_loss = tf.TensorArray(compute_dtype, size=1, dynamic_size=True)
    metric_ma_loss = tf.TensorArray(compute_dtype, size=1, dynamic_size=True)
    metric_snr = tf.TensorArray(compute_dtype, size=1, dynamic_size=True)
    metric_std = tf.TensorArray(compute_dtype, size=1, dynamic_size=True)

    num_out = tf.cast(tf.reduce_prod(tf.shape(obs_data[0])[:-1]),tf.keras.backend.floatx())
    for step in tf.range(num_steps):
        inputs = {'obs':[obs_data[step:step+1]]}
        targets = obs_data[step:step+1]
        # targets = tf.reshape(targets, (tf.shape(targets)[0], -1, tf.shape(targets)[-1]))

        with tf.GradientTape() as tape:
            logits = net(inputs)
            dist = net.dist[0](logits[0])
            loss = util.loss_likelihood(dist, targets)
        if train:
            gradients = tape.gradient(loss, net.trainable_variables)
            net.optimizer['net'].apply_gradients(zip(gradients, net.trainable_variables))
        loss = tf.squeeze(loss) / num_out
        metric_loss = metric_loss.write(step, loss)

        util.stats_update(net.stats['loss'], loss)
        _, ma_loss, _, snr_loss, std_loss = util.stats_get(net.stats['loss'])
        metric_ma_loss = metric_ma_loss.write(step, ma_loss); metric_snr = metric_snr.write(step, snr_loss); metric_std = metric_std.write(step, std_loss)

        if train: net.optimizer['net'].learning_rate = learn_rate * snr_loss**np.e # **np.e # _lr-snre

    metric_loss, metric_ma_loss, metric_snr, metric_std = metric_loss.stack(), metric_ma_loss.stack(), metric_snr.stack(), metric_std.stack()
    return metric_loss, metric_ma_loss, metric_snr, metric_std



device_type = 'CPU'
# device_type, device = 'GPU', 0
seed0 = False
net_blocks = 4; net_width = 2048; latent_size = 16; num_heads = 4; mem_size = 16; latent_dist = 'd'
net_lstm = False; net_attn = {'net':True, 'io':True, 'out':True, 'ar':False}; aio_max_latents = 64
opt_type = 'a'; schedule_type = ''; learn_rate = tf.constant(2e-4, compute_dtype)
aug_data_pos = True # _aug-pos
extra_info = '_lr-snre'

# name, num_cats, ylim = 'cifar10', 10, 100 # (50000, 32, 32, 3) (10000 test)
# name, num_cats, scale, ylim, trainS, testS = 'places365_small', 365, 8, 100, 'train[:262144]', 'test[:49152]' # (1803460, 256, 256, 3) (328500 test) (36500 validation)
# name, num_cats = 'svhn_cropped', 10 # (73257, 32, 32, 3) (26032 test) (531131 extra)
# name, num_cats = 'quickdraw_bitmap', 345; test_split = 40426266 # (50426266, 28, 28, 1) (0 test)
# name, num_cats = 'patch_camelyon', 2 # (262144, 96, 96, 3) (32768 test) (32768 validation)
# name, num_cats = 'i_naturalist2017', 5089 # (579184, None, None, 3) (95986 validation)
# name, num_cats = 'food101', 101 # (75750, None, None, 3) (25250 validation)
# name, num_cats = 'dmlab', 6 # (65550, 360, 480, 3) (22735 test) (22628 validation)

# name, num_cats = 'emnist/byclass', 62 # (697932, 28, 28, 1) (116323 test)
# name, num_cats = 'emnist/bymerge', 47 # (697932, 28, 28, 1) (116323 test)
# name, num_cats = 'emnist/balanced', 47 # (112800, 28, 28, 1) (18800 test)
# name, num_cats = 'emnist/letters', 37 # (88800, 28, 28, 1) (14800 test)
# name, num_cats, ylim = 'emnist/digits', 10, 25 # (240000, 28, 28, 1) (40000 test)
# name, num_cats = 'emnist/mnist', 10 # (60000, 28, 28, 1) (10000 test)

name, obs_key, ylim = 'tiny_shakespeare', 'text', 5 # (1003854, 28, 28, 1) (55770 test)


with tf.device("/device:{}:{}".format(device_type,(device if device_type=='GPU' else 0))):
    seed = 0 if seed0 else time.time_ns(); tf.random.set_seed(seed)
    decoders = {'image':tfds_scale(scale),} if scale > 1 else None

    # text
    ds = tfds.load(name, batch_size=-1, split=[trainS, testS], decoders={obs_key:tfds_bytes(),})
    train_obs, test_obs = tf.expand_dims(ds[0][obs_key][0],-1), tf.expand_dims(ds[1][obs_key][0],-1) # all
    # train_obs, test_obs = tf.expand_dims(ds[0][obs_key][0][:524288],-1), tf.expand_dims(ds[1][obs_key][0][:32768],-1); name += '-lrg'
    # train_obs, test_obs = tf.expand_dims(ds[0][obs_key][0][:12288],-1), tf.expand_dims(ds[1][obs_key][0][:2048],-1); name += '-smal'
    # train_obs, test_obs = tf.expand_dims(ds[0][obs_key][0][:16],-1), tf.expand_dims(ds[1][obs_key][0][:16],-1); name += '-tiny'

    # images
    # ds = tfds.load(name, batch_size=-1, split=[trainS, testS], decoders=decoders) # all
    # ds = tfds.load(name, batch_size=-1, split=['train[:12288]','test[:2048]'], decoders=decoders); name += '-lrg'
    # ds = tfds.load(name, batch_size=-1, split=['train[:4096]','test[:1024]'], decoders=decoders); name += '-med'
    # ds = tfds.load(name, batch_size=-1, split=['train[:512]','test[:128]'], decoders=decoders); name += '-smal'
    # ds = tfds.load(name, batch_size=-1, split=['train[:16]','test[:16]'], decoders=decoders); name += '-tiny'
    # train_obs, train_actions, test_obs, test_actions = ds[0][obs_key], tf.expand_dims(ds[0][action_key],-1), ds[1][obs_key], tf.expand_dims(ds[1][action_key],-1); is_image = True


    # ds = tfds.as_numpy(tfds.load(name, batch_size=-1))
    # train_obs, train_actions, test_obs, test_actions = ds['train']['image'], np.expand_dims(ds['train']['label'],-1), ds['test']['image'], np.expand_dims(ds['test']['label'],-1)
    # train_obs, train_actions, test_obs, test_actions = ds['train']['image'], np.expand_dims(ds['train']['label'],-1), ds['validation']['image'], np.expand_dims(ds['validation']['label'],-1) # i_naturalist2017, food101, places365_small, patch_camelyon, dmlab
    # train_obs, train_actions, test_obs, test_actions = ds['train']['image'][:test_split], np.expand_dims(ds['train']['label'][:test_split],-1), ds['train']['image'][test_split:], np.expand_dims(ds['train']['label'][test_split:],-1) # quickdraw_bitmap
    # train_obs, train_actions = np.concatenate([train_obs, ds['extra']['image']], axis=0), np.concatenate([train_actions, np.expand_dims(ds['extra']['label'],-1)], axis=0) # svhn_cropped
    # train_obs, train_actions = np.concatenate([train_obs, ds['test']['image']], axis=0), np.concatenate([train_actions, np.expand_dims(ds['test']['label'],-1)], axis=0) # places365_small, patch_camelyon, dmlab

    # train_obs, train_actions, test_obs, test_actions = train_obs[:262144], train_actions[:262144], test_obs[:49152], test_actions[:49152]
    # train_obs, train_actions, test_obs, test_actions, name = train_obs[:49152], train_actions[:49152], test_obs[:9216], test_actions[:9216], name+'-full'
    # train_obs, train_actions, test_obs, test_actions, name = train_obs[:12288], train_actions[:12288], test_obs[:2048], test_actions[:2048], name+'-lrg'
    # train_obs, train_actions, test_obs, test_actions, name = train_obs[:4096], train_actions[:4096], test_obs[:1024], test_actions[:1024], name+'-med'
    # train_obs, train_actions, test_obs, test_actions, name = train_obs[:512], train_actions[:512], test_obs[:128], test_actions[:128], name+'-smal'
    # train_obs, train_actions, test_obs, test_actions, name = train_obs[:16], train_actions[:16], test_obs[:16], test_actions[:16], name+'-tiny'
    # train_obs, test_obs = tf.image.resize(train_obs, (8,8), method='nearest'), tf.image.resize(test_obs, (8,8), method='nearest') # _scaled-8px
    # train_obs, test_obs = tf.image.resize(train_obs, (16,16), method='nearest'), tf.image.resize(test_obs, (16,16), method='nearest') # _scaled-16px
    # train_obs, test_obs = tf.image.resize(train_obs, (32,32), method='nearest'), tf.image.resize(test_obs, (32,32), method='nearest') # _scaled-32px
    # train_obs, test_obs = train_obs.numpy(), test_obs.numpy()


    num_steps_train, num_steps_test = train_obs.shape[0], test_obs.shape[0]
    obs_space, action_space = train_obs[0], train_actions[0]
    event_shape, event_size, channels, step_shape = obs_space.shape, int(np.prod(obs_space.shape[:-1]).item()), obs_space.shape[-1], tf.TensorShape(train_obs[0:1].shape)
    num_latents = aio_max_latents if event_size > aio_max_latents else event_size


    initializer = tf.keras.initializers.GlorotUniform(seed)
    opt_spec = [{'name':'net', 'type':opt_type, 'schedule_type':schedule_type, 'learn_rate':learn_rate, 'float_eps':float_eps, 'num_steps':num_steps_train, 'lr_min':2e-7}]; stats_spec = [{'name':'loss', 'b1':0.99, 'b2':0.99, 'dtype':compute_dtype}]
    inputs = {'obs':[tf.constant(train_obs[0:1])]}; in_spec = [{'space_name':'obs', 'name':'', 'event_shape':event_shape, 'event_size':event_size, 'channels':channels, 'step_shape':step_shape, 'num_latents':num_latents}]

    # run, out_spec = run1, [{'space_name':'net', 'name':'', 'dtype':tf.int32, 'dtype_out':compute_dtype, 'min':0, 'max':num_cats-1, 'dist_type':'d', 'num_components':0, 'event_shape':(1,), 'event_size':1, 'step_shape':tf.TensorShape((1,1))}]
    # run, ylim, out_spec = run1, 10, [{'space_name':'net', 'name':'', 'dtype':tf.int32, 'dtype_out':compute_dtype, 'min':0, 'max':num_cats-1, 'dist_type':'c', 'num_components':num_cats, 'event_shape':(1,), 'event_size':1, 'step_shape':tf.TensorShape((1,1))}]
    # run, ylim, out_spec = run1, 10, [{'space_name':'net', 'name':'', 'dtype':tf.int32, 'dtype_out':compute_dtype, 'min':0, 'max':num_cats-1, 'dist_type':'mx', 'num_components':4, 'event_shape':(1,), 'event_size':1, 'step_shape':tf.TensorShape((1,1))}]

    run, out_spec = run2, [{'space_name':'net', 'name':'', 'dtype':tf.int32, 'dtype_out':compute_dtype, 'min':0, 'max':255, 'dist_type':'d', 'num_components':0, 'event_shape':(channels,), 'event_size':event_size, 'step_shape':step_shape}]
    # run, ylim, out_spec = run2, 10, [{'space_name':'net', 'name':'', 'dtype':tf.int32, 'dtype_out':compute_dtype, 'min':0, 'max':255, 'dist_type':'c', 'num_components':256, 'event_shape':(channels,), 'event_size':event_size, 'step_shape':step_shape}]
    # run, ylim, out_spec = run2, 10, [{'space_name':'net', 'name':'', 'dtype':tf.int32, 'dtype_out':compute_dtype, 'min':0, 'max':255, 'dist_type':'mx', 'num_components':4, 'event_shape':(channels,), 'event_size':event_size, 'step_shape':step_shape}]

    latent_spec = {'dtype':compute_dtype, 'latent_size':latent_size, 'num_latents':1, 'max_latents':aio_max_latents, 'max_batch_out':1}
    # latent_spec.update({'inp':latent_size*4, 'midp':latent_size*2, 'outp':latent_size*4, 'evo':int(latent_size/2)})
    latent_spec.update({'inp':net_width, 'midp':int(net_width/2), 'outp':net_width, 'evo':64})
    if latent_dist == 'd': latent_spec.update({'dist_type':'d', 'num_components':latent_size, 'event_shape':(latent_size,)}) # deterministic
    if latent_dist == 'c': latent_spec.update({'dist_type':'c', 'num_components':0, 'event_shape':(latent_size, latent_size)}) # categorical
    if latent_dist == 'mx': latent_spec.update({'dist_type':'mx', 'num_components':int(latent_size/16), 'event_shape':(latent_size,)}) # continuous


    info = "data-{}_{}{}{}_O{}{}_net{}-{}{}{}{}{}_lat{}x{}h{}-{}_lr{:.0e}_Ö{}{}_{}".format(name.replace('/', '_'),run.__name__,out_spec[0]['dist_type'],('_seed0' if seed0 else ''),opt_type,('' if schedule_type=='' else '-'+schedule_type),net_blocks,net_width,('-lstm' if net_lstm else ''),('-attn' if net_attn['net'] else ''),('-io' if net_attn['io'] else ''),('-out' if net_attn['out'] else ''),
        num_latents,latent_size,num_heads,latent_dist,learn_rate,num_cats,extra_info,time.strftime("%y-%m-%d-%H-%M-%S"))

    ## test net
    testnet = nets.ArchFull('TEST', inputs, opt_spec, stats_spec, in_spec, out_spec, latent_spec, net_blocks=net_blocks, net_lstm=net_lstm, net_attn=net_attn, num_heads=num_heads, memory_size=memory_size, aug_data_pos=aug_data_pos); outputs = testnet(inputs)
    testnet.optimizer_weights = util.optimizer_build(testnet.optimizer['net'], testnet.trainable_variables)
    util.net_build(testnet, initializer)

    ## run
    run_fn = tf.function(run, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS); print("RUN {}".format(info))
    import matplotlib as mpl; mpl.rcParams['agg.path.chunksize'] = 50000
    plt.figure(num=info, figsize=(34, 18), tight_layout=True); ylim = (0, ylim)
    rows = (6 if run == run2 and is_image else 4); cols = 7

    t1_start = time.perf_counter_ns()
    metric_loss, metric_ma_loss, metric_snr, metric_std = run_fn(num_steps_train, tf.constant(train_obs), tf.constant(train_actions), testnet)
    total_time = (time.perf_counter_ns() - t1_start) / 1e9 # seconds
    title = "TRAIN    [{}-{}]     total time {}     sec/step {}".format(device_type, tf.keras.backend.floatx(), util.print_time(total_time), total_time/num_steps_train); print(title)
    steps = np.arange(num_steps_train) # , ylim=(0, 10)
    plt.subplot2grid((rows, cols), (0, 0), rowspan=4, colspan=6, xlim=(0, num_steps_train-1), ylim=ylim); plt.plot(steps, metric_loss.numpy(), alpha=0.7, color='lightblue', label='loss'); plt.plot(steps, metric_ma_loss.numpy(), label='ma_loss')
    # plt.plot(steps, metric_std.numpy(), label='std'); plt.plot(steps, metric_snr.numpy(), label='snr')
    plt.legend(loc='upper left'); plt.grid(axis='both',alpha=0.3); plt.title(title)

    testnet.optimizer['net'].learning_rate = learn_rate
    t1_start = time.perf_counter_ns()
    metric_loss, metric_ma_loss, metric_snr, metric_std = run_fn(num_steps_test, tf.constant(test_obs), tf.constant(test_actions), testnet, train=False)
    total_time = (time.perf_counter_ns() - t1_start) / 1e9 # seconds
    title = "TEST  {}  {}".format(util.print_time(total_time), total_time/num_steps_test); print(title)
    steps = np.arange(num_steps_test) # , ylim=(0, 10)
    plt.subplot2grid((rows, cols), (0, 6), rowspan=4, xlim=(0, num_steps_test-1), ylim=ylim); plt.plot(steps, metric_loss.numpy(), alpha=0.7, color='lightblue', label='loss'); plt.plot(steps, metric_ma_loss.numpy(), label='ma_loss')
    # plt.plot(steps, metric_std.numpy(), label='std')
    plt.grid(axis='both',alpha=0.3); plt.title(title)

if run == run2 and is_image:
    rnd = np.random.randint(1,len(test_obs)-1,5)
    itms = [train_obs[0:1], test_obs[0:1], test_obs[rnd[0]:rnd[0]+1], test_obs[rnd[1]:rnd[1]+1], test_obs[rnd[2]:rnd[2]+1], test_obs[rnd[3]:rnd[3]+1], test_obs[rnd[4]:rnd[4]+1]]
    for i in range(len(itms)):
        itm = itms[i]; plt.subplot2grid((rows, cols), (4, i)); plt.axis('off'); plt.imshow(itm[0], cmap='gray')
        plt.subplot2grid((rows, cols), (5, i)); plt.axis('off')

        inputs = {'obs':[tf.constant(itm)]}
        logits = testnet(inputs)
        dist = testnet.dist[0](logits[0])
        # img = tf.reshape(tf.cast(dist.sample(),tf.uint8),itm.shape)[0].numpy()
        # img = dist.mode()
        img = dist.sample()
        img = tf.math.round(img)
        img = tf.clip_by_value(img,0,255)
        img = tf.cast(img,tf.uint8)[0].numpy()
        plt.imshow(img, cmap='gray')

plt.show()
