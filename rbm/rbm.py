import numpy as np
import tensorflow as tf
import logging
import os
from pathlib import Path

tf.compat.v1.disable_eager_execution()
log = logging.getLogger(__name__)


class RBM:

    def __init__(
        self,
        possible_ratings,
        visible_units,
        hidden_units=500,
        keep_prob=0.7,
        init_stdv=0.1,
        learning_rate=0.004,
        minibatch_size=100,
        training_epoch=20,
        display_epoch=10,
        sampling_protocol=[50, 70, 80, 90, 100],
        debug=False,
        with_metrics=True,
        seed=42,
    ):
        #Implementation of a multinomial Restricted Boltzmann Machine for collaborative filtering in numpy/pandas/tensorflow
        
        self.n_hidden = hidden_units
        self.keep = keep_prob
        self.stdv = init_stdv
        self.learning_rate = learning_rate
        self.minibatch = minibatch_size
        self.epochs = training_epoch + 1
        self.display_epoch = display_epoch
        self.sampling_protocol = sampling_protocol
        self.debug = debug
        self.with_metrics = with_metrics
        self.seed = seed
        np.random.seed(self.seed)
        tf.compat.v1.set_random_seed(self.seed)
        self.n_visible = visible_units
        tf.compat.v1.reset_default_graph()
        self.possible_ratings = possible_ratings
        self.ratings_lookup_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(list(range(len(self.possible_ratings))), dtype=tf.int32),
                tf.constant(list(self.possible_ratings), dtype=tf.float32),
            ), default_value=0
        )
        self.generate_graph()
        self.init_metrics()
        self.init_gpu()
        init_graph = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.Session(config=self.config_gpu)
        self.sess.run(init_graph)

    def binomial_sampling(self, pr):
        #Binomial sampling of hidden units activations using a rejection method
        
        g = tf.convert_to_tensor(
            value=np.random.uniform(size=pr.shape[1]), dtype=tf.float32
        )
        h_sampled = tf.nn.relu(tf.sign(pr - g))
        return h_sampled

    def multinomial_sampling(self, pr):
        #Multinomial Sampling of ratings
        
        g = np.random.uniform(size=pr.shape[2])
        f = tf.convert_to_tensor(
            value=g / g.sum(), dtype=tf.float32
        )
        samp = tf.nn.relu(tf.sign(pr - f)) 
        v_argmax = tf.cast(
            tf.argmax(input=samp, axis=2), "int32"
        )
        v_samp = tf.cast(
            self.ratings_lookup_table.lookup(v_argmax), "float32"
        )
        return v_samp

    def multinomial_distribution(self, phi):
        #Probability that unit v has value l given phi: P(v=l|phi)
        
        numerator = [
            tf.exp(tf.multiply(tf.constant(k, dtype="float32"), phi))
            for k in self.possible_ratings
        ]
        denominator = tf.reduce_sum(input_tensor=numerator, axis=0)
        prob = tf.compat.v1.div(numerator, denominator)
        return tf.transpose(a=prob, perm=[1, 2, 0])

    def free_energy(self, x):
        #Free energy of the visible units given the hidden units. Since the sum is over the hidden units' states, the functional form of the visible units Free energy is the same as the one for the binary model
        
        bias = -tf.reduce_sum(input_tensor=tf.matmul(x, tf.transpose(a=self.bv)))
        phi_x = tf.matmul(x, self.w) + self.bh
        f = -tf.reduce_sum(input_tensor=tf.nn.softplus(phi_x))
        F = bias + f
        return F

    def placeholder(self):
        #Initialize the placeholders for the visible units
        
        self.vu = tf.compat.v1.placeholder(shape=[None, self.n_visible], dtype="float32")

    def init_parameters(self):
        #Initialize the parameters of the model. This is a single layer model with two biases. So we have a rectangular matrix w_{ij} and two bias vectors to initialize
        
        with tf.compat.v1.variable_scope("Network_parameters"):
            self.w = tf.compat.v1.get_variable(
                "weight",
                [self.n_visible, self.n_hidden],
                initializer=tf.compat.v1.random_normal_initializer(
                    stddev=self.stdv, seed=self.seed
                ),
                dtype="float32",
            )
            self.bv = tf.compat.v1.get_variable(
                "v_bias",
                [1, self.n_visible],
                initializer=tf.compat.v1.zeros_initializer(),
                dtype="float32",
            )
            self.bh = tf.compat.v1.get_variable(
                "h_bias",
                [1, self.n_hidden],
                initializer=tf.compat.v1.zeros_initializer(),
                dtype="float32",
            )

    def sample_hidden_units(self, vv):
        #In RBM we use Contrastive divergence to sample the parameter space. In order to do that we need to initialize the two conditional probabilities:
        #1. P(h|phi_v) --> returns the probability that the i-th hidden unit is active
        #2. P(v|phi_h) --> returns the probability that the  i-th visible unit is active
        
        with tf.compat.v1.name_scope("sample_hidden_units"):
            phi_v = tf.matmul(vv, self.w) + self.bh
            phv = tf.nn.sigmoid(phi_v) 
            phv_reg = tf.nn.dropout(phv, 1 - (self.keep))
            h_ = self.binomial_sampling(
                phv_reg
            )
        return phv, h_

    def sample_visible_units(self, h):
        #Sample the visible units given the hiddens. This can be thought of as a Backward pass in a FFN (negative phase). Each visible unit can take values in [1,rating], while the zero is reserved for missing data; as such the value of the hidden unit is sampled from a multinomial distribution

        with tf.compat.v1.name_scope("sample_visible_units"):
            phi_h = tf.matmul(h, tf.transpose(a=self.w)) + self.bv  # linear combination
            pvh = self.multinomial_distribution(
                phi_h
            )
            v_tmp = self.multinomial_sampling(
                pvh
            )
            mask = tf.equal(self.v, 0)
            v_ = tf.compat.v1.where(
                mask, x=self.v, y=v_tmp
            ) 
        return pvh, v_

    def gibbs_sampling(self):
        #Determines an estimate of the model configuration via sampling. In the binary RBM we need to impose that unseen movies stay as such, i.e. the sampling phase should not modify the elements where v=0

        with tf.compat.v1.name_scope("gibbs_sampling"):
            self.v_k = (
                self.v
            )
            if self.debug:
                print("CD step", self.k)
            for i in range(self.k):  # k_sampling
                _, h_k = self.sample_hidden_units(self.v_k)
                _, self.v_k = self.sample_visible_units(h_k)

    def losses(self, vv):
        #Calculate contrastive divergence, which is the difference between the free energy clamped on the data (v) and the model Free energy (v_k)

        with tf.compat.v1.variable_scope("losses"):
            obj = self.free_energy(vv) - self.free_energy(self.v_k)
        return obj

    def gibbs_protocol(self, i):
        #Gibbs protocol

        with tf.compat.v1.name_scope("gibbs_protocol"):
            epoch_percentage = (
                i / self.epochs
            ) * 100 
            if epoch_percentage != 0:
                if (
                    epoch_percentage >= self.sampling_protocol[self.l]
                    and epoch_percentage <= self.sampling_protocol[self.l + 1]
                ):
                    self.k += 1
                    self.l += 1
                    self.gibbs_sampling()
            if self.debug:
                log.info("percentage of epochs covered so far %f2" % (epoch_percentage))
                print("Percentage of epochs covered so far %f2" % (epoch_percentage))

    def data_pipeline(self):
        #Define the data pipeline
        
        self.batch_size = tf.compat.v1.placeholder(tf.int64)
        self.dataset = tf.data.Dataset.from_tensor_slices(self.vu)
        self.dataset = self.dataset.shuffle(
            buffer_size=50, reshuffle_each_iteration=True, seed=self.seed
        ) 
        self.dataset = self.dataset.batch(batch_size=self.batch_size).repeat()
        self.iter = tf.compat.v1.data.make_initializable_iterator(self.dataset)
        self.v = self.iter.get_next()

    def init_metrics(self):
        #Initialize metrics

        if self.with_metrics:
            self.rmse = tf.sqrt(
                tf.compat.v1.losses.mean_squared_error(self.v, self.v_k, weights=tf.where(self.v > 0, 1, 0))
            )

    def generate_graph(self):
        #Call the different RBM modules to generate the computational graph
        
        log.info("Creating the computational graph")
        print("Creating the computational graph")
        self.placeholder()
        self.data_pipeline() 
        self.init_parameters()
        log.info("Initialize Gibbs protocol")
        print("Initialize Gibbs protocol")
        self.k = 1
        self.l = 0 
        self.gibbs_sampling()
        obj = self.losses(self.v)
        rate = (
            self.learning_rate / self.minibatch
        )
        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=rate).minimize(
            loss=obj
        )

    def init_gpu(self):
        #Config GPU memory
        
        self.config_gpu = tf.compat.v1.ConfigProto(
            log_device_placement=False, allow_soft_placement=True
        )
        self.config_gpu.gpu_options.allow_growth = True 

    def init_training_session(self, xtr):
        #Initialize the TF session on training data
        
        self.sess.run(
            self.iter.initializer,
            feed_dict={self.vu: xtr, self.batch_size: self.minibatch},
        )
        self.sess.run(tf.compat.v1.tables_initializer())

    def batch_training(self, num_minibatches):
        #Perform training over input minibatches. If `self.with_metrics` is False, no online metrics are evaluated
        epoch_tr_err = 0 
        for _ in range(num_minibatches):
            if self.with_metrics:
                _, batch_err = self.sess.run([self.opt, self.rmse])
                epoch_tr_err += batch_err / num_minibatches
            else:
                _ = self.sess.run(self.opt)
        return epoch_tr_err

    def fit(self, xtr): 
        #Main component of the algo; once instantiated, it generates the computational graph and performs model training
        
        self.seen_mask = np.not_equal(xtr, 0)
        n_users = xtr.shape[0]
        num_minibatches = int(n_users / self.minibatch) 
        self.init_training_session(xtr)
        rmse_train = []
        for i in range(self.epochs):
            self.gibbs_protocol(i)
            epoch_tr_err = self.batch_training(num_minibatches) 
            if self.with_metrics and i % self.display_epoch == 0:
                log.info("training epoch %i rmse %f" % (i, epoch_tr_err))
                print("training epoch %i rmse %f" % (i, epoch_tr_err))
            rmse_train.append(epoch_tr_err) 
        self.rmse_train = rmse_train

    def eval_out(self):
        #Implement multinomial sampling from a trained model
        
        _, h = self.sample_hidden_units(self.vu) 
        phi_h = (
            tf.transpose(a=tf.matmul(self.w, tf.transpose(a=h))) + self.bv
        )
        pvh = self.multinomial_distribution(
            phi_h
        ) 
        v = self.multinomial_sampling(pvh)
        return v, pvh

    def recommend_k_items(self, x, top_k=10, remove_seen=True):
        #Returns the top-k items ordered by a relevancy score
        
        v_, pvh_ = self.eval_out()
        vp, pvh = self.sess.run([v_, pvh_], feed_dict={self.vu: x})
        pv = np.max(pvh, axis=2)
        score = np.multiply(vp, pv)
        log.info("Extracting top %i elements" % top_k)
        print("Extracting top %i elements" % top_k)
        if remove_seen:
            vp[self.seen_mask] = 0
            pv[self.seen_mask] = 0
            score[self.seen_mask] = 0
        top_items = np.argpartition(-score, range(top_k), axis=1)[
            :, :top_k
        ] 
        score_c = score.copy() 
        score_c[
            np.arange(score_c.shape[0])[:, None], top_items
        ] = 0
        top_scores = score - score_c
        return top_scores

    def predict(self, x):
        #Returns the inferred ratings. This method is similar to recommend_k_items() with the exceptions that it returns all the inferred ratings
        
        v_, _ = self.eval_out()
        vp = self.sess.run(v_, feed_dict={self.vu: x})
        return vp

    def save(self, file_path='./rbm_model.ckpt'):
        #Save model parameters to `file_path`
        
        f_path = Path(file_path)
        dir_name, file_name = f_path.parent, f_path.name
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, os.path.join(dir_name, file_name))

    def load(self, file_path='./rbm_model.ckpt'):
        #Load model parameters for further use
        
        f_path = Path(file_path)
        dir_name, file_name = f_path.parent, f_path.name
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, os.path.join(dir_name, file_name))
