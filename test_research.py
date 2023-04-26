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
import matplotlib as mpl; mpl.rcParams['agg.path.chunksize'] = 50000
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

seq_size, num_cats, scale, trainS, testS, obs_key, action_key, name_size, train_act, test_act, is_image, is_text, decoders, loaded_model = 1, 256, 1, 'train', 'test', 'image', 'label', '', tf.constant([[0]]), tf.constant([[0]]), False, False, None, False
@tfds.decode.make_decoder()
def tfds_scale(serialized_image, features, scale):
    return tf.io.decode_jpeg(serialized_image, ratio=scale)
# @tfds.decode.make_decoder(output_dtype=tf.int32)
# def tfds_unicode(text, features):
#     return tf.strings.unicode_decode(text, 'UTF-8')
# @tfds.decode.make_decoder(output_dtype=tf.uint8)
# def tfds_bytes(text, features):
#     return tf.io.decode_raw(text, tf.uint8)
#     # tf.gather(tf.where(tf.not_equal(a, zero_tensor)))
def decode_to_bytes(text):
    text = tf.strings.reduce_join(text, separator='\f\f')
    text = tf.io.decode_raw(text, tf.uint8)
    # tf.print(tf.math.count_nonzero(tf.math.equal(12,text)))
    text = tf.expand_dims(text,-1)
    return text


def run_graph(num_steps, obs_data, actions, net, seq_size, train=True):
    print("tracing"); tf.print("running")
    metric_loss = tf.TensorArray(compute_dtype, size=num_steps, dynamic_size=False, infer_shape=False, element_shape=())
    metric_ma_loss = tf.TensorArray(compute_dtype, size=num_steps, dynamic_size=False, infer_shape=False, element_shape=())
    metric_snr = tf.TensorArray(compute_dtype, size=num_steps, dynamic_size=False, infer_shape=False, element_shape=())
    metric_std = tf.TensorArray(compute_dtype, size=num_steps, dynamic_size=False, infer_shape=False, element_shape=())
    metric_diff = tf.TensorArray(compute_dtype, size=num_steps, dynamic_size=False, infer_shape=False, element_shape=())
    metric_ma_diff = tf.TensorArray(compute_dtype, size=num_steps, dynamic_size=False, infer_shape=False, element_shape=())

    action_size = tf.reduce_prod(tf.shape(actions[0]))
    for step in tf.range(num_steps, dtype=tf.int32):
        inputs = {'obs':[obs_data[step:step+1]]}
        targets = actions[step:step+seq_size]; out_size = tf.shape(targets)[0]

        with tf.GradientTape() as tape:
            logits = net(inputs); dist = net.dist[0](logits[0][:out_size])
            loss = util.loss_likelihood(dist, targets)
            # if out_spec[0]['dist_type'] == 'c':
            #     logit_scale = tf.reduce_mean(tf.math.abs(logits[0][:out_size])) # _loss-logits
            #     if logit_scale < 15: logit_scale = tf.constant(0,compute_dtype)
            #     loss = loss + logit_scale
        if train:
            gradients = tape.gradient(loss, net.trainable_variables)
            net.optimizer['net'].apply_gradients(zip(gradients, net.trainable_variables))
        num_out = tf.cast(action_size*out_size,compute_dtype)

        loss = tf.squeeze(loss) / num_out
        util.stats_update(net.stats['loss'], loss); _, ma_loss, _, snr_loss, std_loss = util.stats_get(net.stats['loss'])
        metric_loss = metric_loss.write(step, loss); metric_ma_loss = metric_ma_loss.write(step, ma_loss); metric_snr = metric_snr.write(step, snr_loss); metric_std = metric_std.write(step, std_loss)

        if train: net.optimizer['net'].learning_rate = learn_rate * snr_loss**np.e # **np.e # _lr-snre
        # if train: net.optimizer['net'].beta_1 = 0.0 + snr_loss * 0.9999 # _opt-bd

        diff = tf.reduce_sum(util.loss_diff(dist.sample(), targets)) / num_out
        util.stats_update(net.stats['diff'], diff); _, ma_diff, _, _, _ = util.stats_get(net.stats['diff'])
        metric_diff = metric_diff.write(step, diff); metric_ma_diff = metric_ma_diff.write(step, ma_diff)

    metric_loss, metric_ma_loss, metric_snr, metric_std, metric_diff, metric_ma_diff = metric_loss.stack(), metric_ma_loss.stack(), metric_snr.stack(), metric_std.stack(), metric_diff.stack(), metric_ma_diff.stack()
    return metric_loss, metric_ma_loss, metric_snr, metric_std, metric_diff, metric_ma_diff
run_fn = tf.function(run_graph, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)



device_type = 'CPU'
# device_type = 'GPU' # use for images (even run1)
load_model, save_model = False, False
max_episodes = 1; max_steps = 256; mem_img_size = 4; memory_size = 256
seed0 = False
net_blocks = 4; net_width = 2048; latent_size = 16; num_heads = 4; latent_dist = 'd'
net_lstm = False; net_attn = {'net':True, 'io':True, 'out':True, 'ar':True}; aio_max_latents = 16
opt_type = 'a'; schedule_type = ''; learn_rate = tf.constant(2e-4, compute_dtype)
aug_data_pos = True # _aug-pos
machine, device, extra_info = 'dev', 0, '' # _aug-pos _lr-snre


with tf.device("/device:{}:{}".format(device_type,(device if device_type=='GPU' else 0))):
    seed = 0 if seed0 else time.time_ns(); tf.random.set_seed(seed)


    ## set text
    # scientific_papers gpt3 lambada reddit # imdb_reviews yelp_polarity_reviews lm1b reddit_tifu pg19
    # name, obs_key = 'huggingface:pg19', 'text'; trainS, testS = 'train[:5000]', 'test[:100]' # (28602, str) (100 test) # 5000 = 2 bil chars
    # name, obs_key = 'lm1b', 'text'; trainS, testS = 'train[:15000000]', 'test[:250000]' # (30301028, str) (306688 test) # 15000000 = 2 bil chars
    # name, obs_key = 'reddit_tifu/short', 'documents'; trainS, testS = 'train[:90%]', 'train[90%:]' # (71766, str) (7974 test) # 115 mil chars
    name, obs_key = 'tiny_shakespeare', 'text' #; decoders = {obs_key:tfds_bytes(),} # (1, 1003854) (55770 test) # 1 mil chars

    ds = tfds.load(name, batch_size=-1, split=[trainS, testS], decoders=decoders)
    train_obsA, test_obsA = decode_to_bytes(ds[0][obs_key]), decode_to_bytes(ds[1][obs_key]); is_text = True
    for ofs in range(0, max_episodes):
        # train_obs, train_act, test_obs, test_act = train_obsA, tf.roll(train_obsA, shift=-1, axis=0), test_obsA, tf.roll(test_obsA, shift=-1, axis=0) # all
        # train_obs, train_act, test_obs, test_act = train_obsA[ofs*524288:ofs*524288+524288], train_obsA[ofs*524288+1:ofs*524288+1+524288], test_obsA[:32768], test_obsA[1:1+32768]; name_size = '-lrg' # 1 hr
        # train_obs, train_act, test_obs, test_act = train_obsA[ofs*98304:ofs*98304+98304], train_obsA[ofs*98304+1:ofs*98304+1+98304], test_obsA[:8192], test_obsA[1:1+8192]; name_size = '-med' # 10 min
        # train_obs, train_act, test_obs, test_act = train_obsA[ofs*12288:ofs*12288+12288], train_obsA[ofs*12288+1:ofs*12288+1+12288], test_obsA[:2048], test_obsA[1:1+2048]; name_size = '-smal' # 1 min
        train_obs, train_act, test_obs, test_act = train_obsA[ofs*16:ofs*16+16], train_obsA[ofs*16+1:ofs*16+1+16], test_obsA[:16], test_obsA[1:1+16]; name_size = '-tiny'


    # ## set images
    # # name, num_cats = 'cifar10', 10 # (50000, 32, 32, 3) (10000 test)
    # # name, num_cats, scale = 'places365_small', 365, 8; trainS, testS = 'train[:262144]', 'test[:49152]' # (1803460, 256, 256, 3) (328500 test) (36500 validation) # test cats == -1
    # # name, num_cats, scale = 'places365_small', 365, 8; trainS, testS = 'train[:{}]'.format(262144), 'train[{}:{}]'.format(262144, 262144+49152) # (1803460, 256, 256, 3) (328500 test) (36500 validation)
    # # name, num_cats = 'svhn_cropped', 10 # (73257, 32, 32, 3) (26032 test) (531131 extra)
    # # name, num_cats = 'quickdraw_bitmap', 345; test_split = 40426266 # (50426266, 28, 28, 1) (0 test)
    # # name, num_cats = 'patch_camelyon', 2 # (262144, 96, 96, 3) (32768 test) (32768 validation)
    # # name, num_cats = 'i_naturalist2017', 5089 # (579184, None, None, 3) (95986 validation)
    # # name, num_cats = 'food101', 101 # (75750, None, None, 3) (25250 validation)
    # # name, num_cats = 'dmlab', 6 # (65550, 360, 480, 3) (22735 test) (22628 validation)
    # # name, num_cats = 'emnist/byclass', 62 # (697932, 28, 28, 1) (116323 test)
    # # name, num_cats = 'emnist/bymerge', 47 # (697932, 28, 28, 1) (116323 test)
    # # name, num_cats = 'emnist/balanced', 47 # (112800, 28, 28, 1) (18800 test)
    # # name, num_cats = 'emnist/letters', 37 # (88800, 28, 28, 1) (14800 test)
    # # name, num_cats = 'emnist/digits', 10 # (240000, 28, 28, 1) (40000 test)
    # name, num_cats = 'emnist/mnist', 10 # (60000, 28, 28, 1) (10000 test)

    # decoders = {'image':tfds_scale(scale),} if scale > 1 else None
    # for ofs in range(0, max_episodes):
    #     # ds = tfds.load(name, batch_size=-1, split=[trainS, testS], decoders=decoders) # all
    #     # ds = tfds.load(name, batch_size=-1, split=['train[{}:{}]'.format(ofs*122880,ofs*122880+122880),'test[:12288]'], decoders=decoders); name_size = '-lrg' # 1 hr
    #     # ds = tfds.load(name, batch_size=-1, split=['train[{}:{}]'.format(ofs*20480,ofs*20480+20480),'test[:2048]'], decoders=decoders); name_size = '-med' # 10 min
    #     # ds = tfds.load(name, batch_size=-1, split=['train[{}:{}]'.format(ofs*2048,ofs*2048+2048),'test[:256]'], decoders=decoders); name_size = '-smal' # 1 min
    #     ds = tfds.load(name, batch_size=-1, split=['train[{}:{}]'.format(ofs*16,ofs*16+16),'test[:16]'], decoders=decoders); name_size = '-tiny'
    #     train_obs, train_act, test_obs, test_act = ds[0][obs_key], tf.expand_dims(ds[0][action_key],-1), ds[1][obs_key], tf.expand_dims(ds[1][action_key],-1); is_image = True



        if train_act.shape[0] == 0: break
        train_obs = train_obs[:train_act.shape[0]]

        ## set text run type
        # run = 'runT1'; memory_size = None; train_act, test_act = train_obs, test_obs ## reconstruct text
        # run = 'runT2' ## predict next text
        run = 'runT3'; seq_size = mem_img_size; train_act, test_act = train_obs, test_obs ## predict next text trajectory

        ## set image run type
        # run = 'run1'; memory_size = None ## predict image category
        # run = 'run2'; memory_size = None; num_cats = 256; train_act, test_act = train_obs, test_obs ## reconstruct image
        # run = 'run3'; memory_size = None; num_cats = 256; train_obs, train_act = train_act, train_obs; test_obs, test_act = test_act, test_obs ## generate image from category



        num_steps_train, num_steps_test = train_obs.shape[0], test_obs.shape[0]
        obs_space, action_space = train_obs[0], train_act[0]
        obs_event_shape, obs_event_size, obs_channels, obs_step_shape = obs_space.shape, int(np.prod(obs_space.shape[:-1]).item()), obs_space.shape[-1], tf.TensorShape(train_obs[0:1].shape)
        act_event_shape, act_event_size, act_channels, act_step_shape = action_space.shape, int(np.prod(action_space.shape[:-1]).item()), action_space.shape[-1], tf.TensorShape(train_act[0:1].shape)
        num_latents = aio_max_latents if obs_event_size > aio_max_latents else obs_event_size
        if num_latents == 1 and memory_size is not None: net_attn.update({'ar':False}) # attn makes no sense w/1 latent
        if num_latents == 1 and memory_size is None: net_lstm = False; net_attn.update({'net':False, 'out':False, 'ar':False})
        ylimD = min(int(num_cats/2),100)
        out_spec_dtype = tf.uint8 if num_cats <= 256 else tf.int32

        ## set out_spec dist type
        # ylim, out_spec = ylimD, [{'space_name':'net', 'name':'', 'dtype':out_spec_dtype, 'dtype_out':compute_dtype, 'min':0, 'max':num_cats-1, 'dist_type':'d', 'num_components':0, 'event_shape':(act_channels,), 'event_size':act_event_size, 'step_shape':act_step_shape, 'seq_size_out':seq_size}]
        ylim, out_spec = 5, [{'space_name':'net', 'name':'', 'dtype':out_spec_dtype, 'dtype_out':compute_dtype, 'min':0, 'max':num_cats-1, 'dist_type':'c', 'num_components':num_cats, 'event_shape':(act_channels,), 'event_size':act_event_size, 'step_shape':act_step_shape, 'seq_size_out':seq_size}]
        # ylim, out_spec = 5, [{'space_name':'net', 'name':'', 'dtype':out_spec_dtype, 'dtype_out':compute_dtype, 'min':0, 'max':num_cats-1, 'dist_type':'mx', 'num_components':4, 'event_shape':(act_channels,), 'event_size':act_event_size, 'step_shape':act_step_shape, 'seq_size_out':seq_size}]



        initializer = tf.keras.initializers.GlorotUniform(seed)
        opt_spec = [{'name':'net', 'type':opt_type, 'schedule_type':schedule_type, 'learn_rate':learn_rate, 'float_eps':float_eps, 'num_steps':num_steps_train, 'lr_min':2e-7}]; stats_spec = [{'name':'loss', 'b1':0.99, 'b2':0.99, 'dtype':compute_dtype},{'name':'diff', 'b1':0.99, 'b2':0.99, 'dtype':compute_dtype}]
        inputs = {'obs':[tf.constant(train_obs[0:1])]}; in_spec = [{'space_name':'obs', 'name':'', 'event_shape':obs_event_shape, 'event_size':obs_event_size, 'channels':obs_channels, 'step_shape':obs_step_shape, 'num_latents':num_latents}]

        latent_spec = {'dtype':compute_dtype, 'latent_size':latent_size, 'num_latents':1, 'max_latents':aio_max_latents}
        # latent_spec.update({'inp':latent_size*4, 'midp':latent_size*2, 'outp':latent_size*4, 'evo':int(latent_size/2)})
        latent_spec.update({'inp':net_width, 'midp':int(net_width/2), 'outp':net_width, 'evo':64})
        if latent_dist == 'd': latent_spec.update({'dist_type':'d', 'num_components':latent_size, 'event_shape':(latent_size,)}) # deterministic
        if latent_dist == 'c': latent_spec.update({'dist_type':'c', 'num_components':0, 'event_shape':(latent_size, latent_size)}) # categorical
        if latent_dist == 'mx': latent_spec.update({'dist_type':'mx', 'num_components':int(latent_size/16), 'event_shape':(latent_size,)}) # continuous


        info = "data-{}{}_{}-a{}_{}{}{}_O{}{}_net{}-{}{}{}{}{}{}_lat{}x{}h{}-{}_lr{:.0e}_Ö{}{}{}{}_{}".format(name.replace('_','').replace('/','-').replace(':','-'),name_size,machine,device,run,out_spec[0]['dist_type'],('_seed0' if seed0 else ''),opt_type,('' if schedule_type=='' else '-'+schedule_type),net_blocks,net_width,
            ('-lstm' if net_lstm else ''),('-attn' if net_attn['net'] else ''),('-ar' if net_attn['net'] and net_attn['ar'] and memory_size is not None else ''),('-io' if net_attn['io'] else ''),('-out' if net_attn['out'] else ''),
            num_latents,latent_size,num_heads,latent_dist,learn_rate,num_cats,('_mem'+str(memory_size) if net_attn['net'] and memory_size is not None else ''),('-img'+str(mem_img_size) if seq_size > 1 else ''),extra_info,time.strftime("%y-%m-%d-%H-%M-%S"))

        ## test net
        testnet = nets.ArchFull('TEST', inputs, opt_spec, stats_spec, in_spec, out_spec, latent_spec, net_blocks=net_blocks, net_lstm=net_lstm, net_attn=net_attn, num_heads=num_heads, memory_size=memory_size, aug_data_pos=aug_data_pos); outputs = testnet(inputs)
        testnet.optimizer_weights = util.optimizer_build(testnet.optimizer['net'], testnet.trainable_variables)
        util.net_build(testnet, initializer)

        model_file = "{}/tf-data-models-local/{}-{}-a{}.h5".format(curdir, testnet.arch_desc_file, machine, device)
        if load_model and tf.io.gfile.exists(model_file): testnet.load_weights(model_file, by_name=True, skip_mismatch=True); print("LOADED {} weights from {}".format(testnet.name, model_file)); loaded_model = True


        ## run
        print("RUN {:03} {}".format(ofs, info))
        plt.figure(num=info, figsize=(34, 18), tight_layout=True); ylim = (0, ylim); ylimD = (0, ylimD)
        rows, cols = 8, 7
        if is_text and run in ('runT2','runT3'): rows += 2
        if is_image and run in ('run2','run3'): rows += 4

        t1_start = time.perf_counter_ns()
        metric_loss, metric_ma_loss, metric_snr, metric_std, metric_diff, metric_ma_diff = run_fn(num_steps_train, tf.constant(train_obs), tf.constant(train_act), testnet, seq_size=seq_size, train=True)
        total_time = (time.perf_counter_ns() - t1_start) / 1e9 # seconds
        title = "TRAIN    [{}-{}]     total time {}     sec/step {}      {}-{}".format(device_type, tf.keras.backend.floatx(), util.print_time(total_time), total_time/num_steps_train, testnet.arch_desc_file, ('load' if loaded_model else 'new')); print(title)
        steps = np.arange(num_steps_train) # ylim=ylim yscale='log'
        plt.subplot2grid((rows, cols), (0, 0), rowspan=4, colspan=6, xlim=(0, num_steps_train-1), ylim=ylim); plt.plot(steps, metric_loss.numpy(), alpha=0.7, color='lightblue', label='loss'); plt.plot(steps, metric_ma_loss.numpy(), label='ma_loss'); plt.legend(loc='upper left'); plt.grid(axis='both',alpha=0.3); plt.title(title)
        plt.subplot2grid((rows, cols), (4, 0), rowspan=4, colspan=6, xlim=(0, num_steps_train-1), ylim=ylimD); plt.plot(steps, metric_diff.numpy(), alpha=0.7, color='lightblue', label='diff'); plt.plot(steps, metric_ma_diff.numpy(), label='ma_diff'); plt.legend(loc='upper left'); plt.grid(axis='both',alpha=0.3)
        # plt.subplot2grid((rows, cols), (8, 0), rowspan=1, colspan=6, xlim=(0, num_steps_train-1), ylim=(0,1)); plt.plot(steps, metric_snr.numpy(), label='snr'); plt.grid(axis='both',alpha=0.3)

        testnet.optimizer['net'].learning_rate = learn_rate # _lr-snre
        # testnet.optimizer['net'].beta_1 = tf.constant(0.9,compute_dtype) # _opt-bd
        t1_start = time.perf_counter_ns()
        metric_loss, metric_ma_loss, metric_snr, metric_std, metric_diff, metric_ma_diff = run_fn(num_steps_test, tf.constant(test_obs), tf.constant(test_act), testnet, seq_size=seq_size, train=False)
        total_time = (time.perf_counter_ns() - t1_start) / 1e9 # seconds
        title = "TEST  {}  {}".format(util.print_time(total_time), total_time/num_steps_test); print(title)
        steps = np.arange(num_steps_test) # ylim=ylim yscale='log'
        plt.subplot2grid((rows, cols), (0, 6), rowspan=4, xlim=(0, num_steps_test-1), ylim=ylim); plt.plot(steps, metric_loss.numpy(), alpha=0.7, color='lightblue', label='loss'); plt.plot(steps, metric_ma_loss.numpy(), label='ma_loss'); plt.grid(axis='both',alpha=0.3); plt.title(title)
        plt.subplot2grid((rows, cols), (4, 6), rowspan=4, xlim=(0, num_steps_test-1), ylim=ylimD); plt.plot(steps, metric_diff.numpy(), alpha=0.7, color='lightblue', label='diff'); plt.plot(steps, metric_ma_diff.numpy(), label='ma_diff'); plt.grid(axis='both',alpha=0.3)
        # plt.subplot2grid((rows, cols), (8, 6), rowspan=1, xlim=(0, num_steps_test-1), ylim=(0,1)); plt.plot(steps, metric_snr.numpy(), label='snr'); plt.grid(axis='both',alpha=0.3)


        if is_text and run in ('runT2','runT3'):
            # testnet.reset_states()
            out = train_obs[0:1]
            text = b''
            for i in range(max_steps):
                byte = util.discretize(out, out_spec[0])
                inputs = {'obs':[tf.constant(tf.reshape(byte,(1,1)))]}
                logits = testnet(inputs)
                dist = testnet.dist[0](logits[0])
                if run == 'runT2': out = dist.sample()[0:1]
                if run == 'runT3': out = dist.sample()[1:2]
                byte = byte.numpy().tobytes()
                text += byte
            text = text.decode('utf-8', errors='replace')
            print(text)
            ax = plt.subplot2grid((rows, cols), (rows-2, 0), rowspan=2, colspan=cols); plt.axis('off'); ax.text(0.5, 0.5, text, va='center', ha='center', ma='left', clip_on=False, in_layout=False)

        if is_image and run in ('run2','run3'):
            rnd = np.random.randint(1,len(test_obs)-1,5)
            itms = [train_obs[0:1], test_obs[0:1], test_obs[rnd[0]:rnd[0]+1], test_obs[rnd[1]:rnd[1]+1], test_obs[rnd[2]:rnd[2]+1], test_obs[rnd[3]:rnd[3]+1], test_obs[rnd[4]:rnd[4]+1]]
            for i in range(len(itms)):
                itm = itms[i]; ax = plt.subplot2grid((rows, cols), (rows-4, i), rowspan=2); plt.axis('off')
                if run == 'run2': plt.imshow(itm[0], cmap='gray')
                if run == 'run3': ax.text(0.5, 0.5, str(itm[0][0].numpy().item()), va='center', ha='center', size='xx-large', weight='bold')
                plt.subplot2grid((rows, cols), (rows-2, i), rowspan=2); plt.axis('off')

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

        if max_episodes == 1: plt.show()
        else: plt.savefig("output_util/{}.png".format(info))
        plt.close("all")
        if save_model: testnet.save_weights(model_file); print("SAVED {} weights to {}".format(testnet.name, model_file)); load_model = True
