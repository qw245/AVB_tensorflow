# Set up
import tensorflow as tf
import numpy as np
from scipy.misc import imsave as ims

import os
GPUID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
from tensorflow.contrib import layers
import matplotlib.pyplot as plt

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace

## CIFAR data
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
n_samples = 50000
nr = 32
nc = 32
nch = 3

file = "cifar-10-batches-py/data_batch_1"
images_1 = unpickle(file)
images_t = images_1['data']

file = "cifar-10-batches-py/data_batch_2"
images_2 = unpickle(file)
images_t = np.concatenate((images_t, images_2['data']), axis=0)

file = "cifar-10-batches-py/data_batch_3"
images_3 = unpickle(file)
images_t = np.concatenate((images_t, images_3['data']), axis=0)

file = "cifar-10-batches-py/data_batch_4"
images_4 = unpickle(file)
images_t = np.concatenate((images_t, images_4['data']), axis=0)

file = "cifar-10-batches-py/data_batch_5"
images_5 = unpickle(file)
images_train = np.concatenate((images_t, images_5['data']), axis=0)
images_train.astype(float)
images_t = images_train / 255.0

file = "cifar-10-batches-py/test_batch"
# im_test = tf.stack(unpickle(file)['data'])
im_test = unpickle(file)['data']
im_test.astype(float)
# im_test = tf.float32(im_test)
images_test = im_test / 255.0
#########################################################################################


#Parameters: this is a dictionary
params = {
    'batch_size': 128,
    'latent_dim': 256,  # dimensionality of latent space
    'eps_dim': 3072,  # dimensionality of epsilon, used in inference net, z_phi(x, eps) 3072
    'input_dim': 3072,  # dimensionality of input (also the number of unique datapoints) 3072
    'n_hidden_disc': 256,  # number of hidden units in discriminator 256
    'n_hidden_gen': 64,  # number of hidden units in generator
    'n_hidden_inf': 64,  # number of hidden units in inference model
}

# Network definitions
def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return st.StochasticTensor(
        ds.MultivariateNormalDiag(mu=tf.zeros(shape), diag_stdev=tf.ones(shape)*0.01, **kwargs))


# Note that the output of generative network is two distributions
def generative_network(batch_size, latent_dim, input_dim, eps=1e-6):
    with tf.variable_scope("generative"):
        p_z = standard_normal([batch_size, latent_dim], name="p_z")
        net = tf.expand_dims(p_z.value(), 1)
        net = tf.expand_dims(net, 1) # The dimension of net is batch_size * 1 * 1 * latent_dim
        net = slim.conv2d_transpose(net, 16, [5, 5], stride=1, padding='VALID', activation_fn=tf.nn.relu)
        net = slim.conv2d_transpose(net, 32, [5, 5], stride=2, padding='VALID', activation_fn=tf.nn.relu)
        net = slim.conv2d_transpose(net, 64, [5, 5], stride=2, padding='VALID', activation_fn=tf.nn.relu)
        h = slim.flatten(net)

        # BUG: BernoulliSigmoidP gives NaNs when log_p is large, so we constrain
        # probabilities to be in (eps, 1-eps) and use Bernoulli
        p = eps + (1-2 * eps) * slim.fully_connected(h, input_dim, activation_fn=tf.nn.sigmoid)
        p_x = st.StochasticTensor(ds.Bernoulli(p=p, name="p_x"))
    return [p_x, p_z]


# Note that the output of inference network is a value instead of a distribution
def inference_network(x, latent_dim, eps_dim):
    eps = standard_normal([x.get_shape().as_list()[0], eps_dim], name="eps").value()
    h = tf.stack([x, eps], 1)
    # h = tf.reshape(h, [128, nr, nc, 2])
    # h = reshape(params['batch_size'], nr, nc, 2)
    with tf.variable_scope("inference"):
        # h = slim.max_pool2d(h, [2, 2])
        # z = slim.dropout(z, 0.5, scope='dropout6')
        h = tf.reshape(h, [-1, nr, nc, 2*nch])
        # h = layers.conv2d(h, 32, 3, stride=2, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)

        h = slim.conv2d(h, 64, [5, 5], padding='SAME', stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        h = slim.conv2d(h, 32, [5, 5], padding='SAME', stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        h = slim.conv2d(h, 16, [5, 5], padding='SAME', stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        # h = layers.dropout(h, keep_prob=0.9)
        h = slim.flatten(h)
        z = slim.fully_connected(h, latent_dim, activation_fn=None)
    return z


# the output of data_network is T(x,z), which should be log q(z|x) - log p(z)
def data_network(x, z, n_hidden=params['n_hidden_disc'], activation_fn=None):
    """Approximate log data density."""
    x = tf.reshape(x, [128, nr, nc, nch])
    with tf.variable_scope('discriminator'):
        x = slim.conv2d(x, 64, [5, 5], padding='SAME', stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        x = slim.conv2d(x, 32, [5, 5], padding='SAME', stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        x = slim.conv2d(x, 16, [5, 5], padding='SAME', stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        x = tf.reshape(x, [128, nr/4 * nc/4 * 16])
        z = slim.fully_connected(z, n_hidden, activation_fn=None)
        h = tf.concat([x, z], 1)
        log_d = slim.fully_connected(h, 1, activation_fn=activation_fn)
    return tf.squeeze(log_d, squeeze_dims=[1])


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], size[2]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

# Construct model and training ops
tf.reset_default_graph()

#x = tf.constant(np_data)
# vislz = mnist.train.next_batch(params['batch_size'])[0]
# # vislz_tf = tf.constant(vislz, dtype=tf.float32)
# reshaped_vis = vislz.reshape(params['batch_size'], nr, nc, nch)

# vislz = images_t[0:params['batch_size'], :]
vislz = images_test[0:params['batch_size'], :]
# vislz_tf = tf.constant(vislz, dtype=tf.float32)
reshaped_vis = np.transpose(vislz.reshape(params['batch_size'], nch, nr, nc), (0, 2, 3, 1))

ims("results/base.jpg", np.squeeze(merge(reshaped_vis[:64], [8, 8, nch])))

images = tf.placeholder(tf.float32, [params['batch_size'], nr*nc*nch])

p_x, p_z = generative_network(params['batch_size'], params['latent_dim'], params['input_dim'])
q_z = inference_network(images, params['latent_dim'], params['eps_dim'])

# Discriminator classifies between (x, z_prior) and (x, z_posterior)
# where z_prior ~ p(z), and z_posterior = q(z, eps) with eps ~ N(0, I)

# p_z.value() is a sample of z drew from p_z
log_d_prior = data_network(images, p_z.value(), n_hidden=params['n_hidden_disc'])
log_d_posterior = graph_replace(log_d_prior, {p_z.value(): q_z})
disc_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=log_d_posterior, labels=tf.ones_like(log_d_posterior)) +
    tf.nn.sigmoid_cross_entropy_with_logits(logits=log_d_prior, labels=tf.zeros_like(log_d_prior)))

# Compute log p(x|z) with z ~ p(z), used as a placeholder
recon_likelihood_prior = p_x.distribution.log_prob(images)
# Compute log p(x|z) with z = q(x, eps)
# This is the same as the above expression, but with z replaced by a sample from q instead of p
recon_likelihood = tf.reduce_sum(graph_replace(recon_likelihood_prior, {p_z.value(): q_z}), [1])

# Generator tries to maximize reconstruction log-likelihood while minimizing the discriminator output
gen_loss = tf.reduce_mean(log_d_posterior) - tf.reduce_mean(recon_likelihood)

qvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inference")
pvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generative")
dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
opt = tf.train.AdamOptimizer(2e-4, beta1=0.5)#, epsilon=1e-3)
# opt = tf.train.AdagradOptimizer(2e-4)
# opt = tf.train.GradientDescentOptimizer(1e-4)

train_gen_op = opt.minimize(gen_loss, var_list=qvars + pvars)
train_disc_op = opt.minimize(disc_loss, var_list=dvars)

#from tqdm import tqdm
fs = []
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth=True
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# sess = tf.InteractiveSession()
# tf.ConfigProto(gpu_options=gpu_options)
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(200): #tqdm(xrange(100)):
        for idx in range(int(n_samples / params['batch_size'])):
            # batch = mnist.train.next_batch(params['batch_size'])[0]
            batch = images_t[params['batch_size'] * idx:params['batch_size'] * (idx + 1), :]
            f, _, _ = sess.run([[gen_loss, disc_loss], train_gen_op, train_disc_op], feed_dict={images: batch})
            fs.append(f)
        print "epoch %d: genloss %f discloss %f" % (epoch, f[0], f[1])

        #vislz_test = sess.run(p_x.value(), feed_dict={images: vislz})
        tem = sess.run(q_z, feed_dict={images: vislz})
        vislz_test = sess.run(p_x.distribution.p, feed_dict={p_z.value(): tem})

        # vislz_test = sess.run(p_x.value(), feed_dict={p_z.value(): tem})
        # vislz_test = sess.run(p_x.distribution.log_prob(vislz))
        # vislz_test = np.exp(vislz_test)


        # generated_test = vislz_test.reshape(params['batch_size'], nr, nc, nch)
        generated_test = np.transpose(vislz_test.reshape(params['batch_size'], nch, nr, nc), (0, 2, 3, 1))

        print "test error: %f " % (np.mean(np.square(generated_test - reshaped_vis)))
        ims("results/" + str(epoch) + ".jpg", np.squeeze(merge(generated_test[:64], [8, 8, nch])))