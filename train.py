import tensorflow as tf
import numpy as np
from network import *
from utils import *
from loss import *
from data import *
from tqdm import tqdm


def random_select(x, m):
    n = int(x.get_shape()[-1])
    prob = [1.0/n] * n
    dist = tf.contrib.distributions.Categorical(tf.log(prob))
    samples = dist.sample(m)
    vv = tf.gather(x, samples, axis=int(x.shape.ndims) - 1)
    print(samples.get_shape())
    print('random select {}'.format(vv.get_shape()))
    return vv


def fixed_random_select(x, m):
    samples = np.random.choice(x.get_shape()[-1], m)
    print('fixed random select = {}'.format(samples))
    xsamples = tf.gather(x, samples, axis=int(x.shape.ndims) - 1)
    return xsamples


def allocate_channels(channels, pweight):
    total_left = channels
    alloc = []
    for i in range(len(pweight) - 1):
        cur = int(pweight[i] * channels)
        alloc.append(cur)
        total_left -= cur
    alloc.append(total_left)
    return alloc


def show_params(domain, var_list):
    print('Domain {}:'.format(domain))
    for var in var_list:
        print('{}: {}'.format(var.name, var.shape))


def show_grads(domain, gd_list):
    print('Grad Domain {}:'.format(domain))
    for (grad, var) in gd_list:
        if grad is not None:
            print(var.name)

def create_var_dict(var_list):
    var_dict = {}
    for var in var_list:
        var_dict[var.name] = var
    return var_dict

def find_layer_feature_map(modules, name):
    for key, value in modules.items():
        if name in key:
            print('Find Layer {} Feature Map: shape = {}'.format(name, value.shape))
            return value

def find_weight(var_list, name):
    for weight in var_list:
        if name in weight.name and 'weight' in weight.name:
            print('Hit weight name [{}] in dict: name = {}'.format(name, weight.name))
            return weight


def build_2nn(hiddens, x_dim, y_dim, regularizer_type='l2', regw=1.0, base_lr=1e-3):

    inp_x = tf.placeholder(dtype=tf.float32, 
                           shape=[None] + [x_dim],
                           name='x')
    out_y = tf.placeholder(dtype=tf.float32,
                           shape=[None] + [y_dim],
                           name='y')
    lr_decay = tf.placeholder(dtype=tf.float32, shape=[], name='lr_decay')

    ph = {
        'x': inp_x,
        'y': out_y,
        'lr_decay': lr_decay
    }

    graph = {}
    net_vars = {}
    targets = {}
    for domain in ['net1', 'net2']:
        modules = make_layers_fully_connected(domain, inp_x, hiddens, y_dim, 
            activation_fn=tf.nn.tanh)
        graph[domain] = modules
        net_vars[domain] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                             scope=domain)
        show_params(domain, net_vars[domain])

    for domain in ['net1', 'net2']:
        pred_y = graph[domain]['out']
        regression_loss = tf.reduce_mean(tf.square(pred_y - out_y))
        regularizer = get_regularizer_loss(net_vars[domain], regularizer_type)

        loss = regression_loss + regw * regularizer

        sl_op = tf.train.MomentumOptimizer(base_lr * ph['lr_decay'], 
            0.9, use_nesterov=True)
        #sl_op = tf.train.AdamOptimizer(base_lr * ph['lr_decay'])
        sl_grads = sl_op.compute_gradients(loss=loss, var_list=net_vars[domain])
        sl_train_op = sl_op.apply_gradients(grads_and_vars=sl_grads)

        show_grads(domain, sl_grads)
        train = {
            'train': sl_train_op,
            'overall_loss': loss,
            'l2_loss': regression_loss,
            'reg_loss': regularizer,
        }

        test = {
            'overall_loss': loss,
            'l2_loss': regression_loss,
            'reg_loss': regularizer,
        }
        targets[domain] = {'train': train, 'test': test}

    targets['eval'] = {}

    for lid in range(len(hiddens)):
        feat = {}
        #feature_std = {}
        for domain in ['net1', 'net2']:
            name = 'l{}-fc'.format(lid)
            feat[domain] = tf.transpose(graph[domain][name])
            print('feature map at layer {} in {} shape = {}'.format(lid, 
                domain, feat[domain].get_shape()))

        l2_dist = compute_pairwise_l2_dist(feat['net1'], feat['net2']) / \
            tf.cast(tf.shape(feat['net1'])[1], tf.float32)
        l2_ndist = compute_pairwise_l2_ndist(feat['net1'], feat['net2']) / \
            tf.cast(tf.shape(feat['net1'])[1], tf.float32)
        targets['eval'][lid] = {
            'l2_dist': l2_dist,
            'l2_ndist': l2_ndist
        }

    return ph, graph, net_vars, targets


# set hyperparameters
hiddens = [1000, 1000, 1000]
x_dim = 1
y_dim = 1
regularizer_type = 'l2'
regw = 2e-4
base_lr = 1e-3
batch_size = 100
num_epoches = 20
decay = 1.0
gpu = -1
figlogdir = 'semi_same/'

ph, graph, net_vars, targets = build_2nn(hiddens, x_dim, y_dim, 
                                         regularizer_type, regw, base_lr)

# fetch data
RANGE = 2 * np.pi
reg_func = cosc
reg_func2 = cosc
ndata_train = 100000
ndata_test = 1000
ndata_eval = 100

train_x, train_y = cosc_data(-RANGE, RANGE, 1.0, ndata_train)
train_y2 = reg_func2(train_x)
test_x, test_y = cosc_data(-RANGE, RANGE, 1.0, ndata_test)
test_y2 = reg_func2(test_x)
print(train_x.shape)
print(train_y.shape)

train_y = {'net1': train_y, 'net2': train_y2}
test_y = {'net1': test_y, 'net2': test_y2}

eval_x = np.arange(ndata_eval)
eval_x = np.reshape(eval_x, [ndata_eval, 1])
eval_x = (eval_x - ndata_eval * 0.5) / (ndata_eval * 0.5)
eval_x = eval_x * RANGE

# scaling
mean_x = np.mean(train_x, 0, keepdims=True)
std_x = np.std(train_x, 0, keepdims=True)
train_x = (train_x - mean_x) / std_x
test_x = (test_x - mean_x) / std_x
eval_x = (eval_x - mean_x) / std_x

# set up section
if gpu > -1:
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, 
                                            log_device_placement=True))
else:
    sess = tf.Session()
sess.run(tf.global_variables_initializer())

matching_result = np.zeros((num_epoches, len(hiddens)))
raw_data = np.zeros((num_epoches, len(hiddens), hiddens[0], hiddens[0]))
matched_rdata = np.zeros((num_epoches, len(hiddens), hiddens[0]), dtype=np.int64)

def semi_matching_algo(dist_mat):
    value = np.min(dist_mat, 1)
    ind = np.argmin(dist_mat, 1)
    return value, ind

def greedy_matching_algo(x):
    oc = np.zeros(x.shape[0])
    perm = []
    values = []
    for i in range(x.shape[0]):
        val, ic = 1e9, 0
        for j in range(x.shape[0]):
            if val > x[i, j] and oc[j] < 0.5:
                val = x[i, j]
                ic = j
        oc[ic] = 1
        values.append(val)
        perm.append(ic)
    values = np.array(values)
    perm = np.array(perm, dtype=np.int32)
    return values, perm

def hungarian_matching_algo(dist_mat):
    np.savetxt('mat.in', dist_mat, delimiter=' ')
    os.system("./hungarian")
    perm = np.loadtxt('mat.out', delimiter=' ')
    perm = perm.astype(np.int64)
    values = np.zeros(dist_mat.shape[0])
    for i in range(dist_mat.shape[0]):
        values[i] = dist_mat[i, perm[i]]
    return values, perm


my_matching_algo = semi_matching_algo

for epoch in range(num_epoches):
    # calculate l2 distance
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    plt.figure(figsize=(18, 6))

    for lid in range(len(hiddens)):
        fetch = sess.run(targets['eval'][lid], feed_dict={ph['x']: eval_x})
        l2_ndist = fetch['l2_ndist']
        print(l2_ndist.shape)
        
        # plot
        v, ind = my_matching_algo(l2_ndist)
        raw_data[epoch, lid, :, :] = l2_ndist
        matched_rdata[epoch, lid, :] = ind
        vind = np.argsort(v)
        matching_result[epoch, lid] = np.mean(v)
        # nearest channel in net2 (sorted)
        ax = plt.subplot(1, 3, lid + 1)
        plt.plot(np.arange(hiddens[lid]) + 1, v[vind])
        plt.xlabel('channel index in net1')
        plt.ylabel('matched l2 distance (after normalization)')
        plt.ylim(0, 2.0)

    filename = os.path.join(figlogdir, 'e{}.pdf'.format(epoch))
    plt.savefig(filename)
    plt.close()
    plt.clf()

    print('======== EVAL {} ========'.format(epoch))
    print(matching_result[epoch, :])

    test_info = {'net1': {}, 'net2': {}}
    for t in tqdm(range(ndata_test // batch_size)):
        batch_x = test_x[t * batch_size: (t + 1) * batch_size, :]

        for domain in ['net1', 'net2']:
            batch_y = test_y[domain][t * batch_size: (t + 1) * batch_size]
            fetch = sess.run(targets[domain]['test'], 
                             feed_dict={ph['x']: batch_x, ph['y']: batch_y})
            update_loss(fetch, test_info[domain])

    for domain in ['net1', 'net2']:
        print_log('Test ' + domain, epoch, test_info[domain])

    train_info = {'net1': {}, 'net2': {}}
    cur_idx = np.random.permutation(ndata_train)
    for t in tqdm(range(ndata_train // batch_size)):
        batch_idx = cur_idx[t * batch_size: (t + 1) * batch_size]
        batch_x = train_x[batch_idx, :]
        ep_id = epoch
        for domain in ['net1', 'net2']:
            batch_y = train_y[domain][batch_idx]
            fetch = sess.run(targets[domain]['train'], 
                             feed_dict={ph['x']: batch_x, ph['y']: batch_y, 
                                        ph['lr_decay']: decay**(ep_id)})
            update_loss(fetch, train_info[domain])
    for domain in ['net1', 'net2']:
        print_log('Train ' + domain, epoch, test_info[domain])

np.save('matching_result.npy', matching_result)
np.save('matching_raw_data.npy', raw_data)
np.save('matching_matched_rdata.npy', matched_rdata)

