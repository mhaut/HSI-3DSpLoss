from time import time
import numpy as np
import keras.backend as K
# from keras.engine.topology import Layer, InputSpec
from tensorflow.keras.layers import Layer, InputSpec
from keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
import metrics
from ConvAE import CAE, CAE3D
import tensorflow as tf
from keras.layers import Lambda
import keras
import matplotlib.pyplot as plt
import os
from datasets import load_hyper
from keras.callbacks import CSVLogger


class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class DCEC(object):
    def __init__(self,
                 input_shape,
                 filters=[32, 64, 128, 10],
                 n_clusters=10,
                 alpha=1.0):
        super(DCEC, self).__init__()
        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.alpha = alpha
        self.pretrained = False
        self.y_pred = []
        self.cae = CAE3D(input_shape, filters)
        hidden = self.cae.get_layer(name='embedding').output
        self.encoder = Model(inputs=self.cae.input, outputs=hidden)
        # Define DCEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
        self.model = Model(inputs=self.cae.input,
                           outputs=[clustering_layer, self.cae.output, clustering_layer])


    def pretrain(self, x, batch_size=256, epochs=20, optimizer='adam', save_dir='results/temp'):
        self.cae.compile(optimizer=optimizer, loss='mse')
        csv_logger = CSVLogger(args.save_dir + '/pretrain_log.csv')
        # begin training
        t0 = time()
        self.cae.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger], verbose=2)
        self.cae.save(save_dir + '/pretrain_cae_model.h5')
        self.pretrained = True


    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)


    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)


    def predict(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)


    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T


    def calculate_spectral_loss(self, image, nn_output, eps=1e-6):
        _sm = K.softmax(nn_output, axis=1)
        _sm   = K.reshape(_sm, [_sm.shape[0], _sm.shape[1], 1])
        image = image[:,image.shape[1]//2, image.shape[2]//2,:,0]
        image = K.reshape(image, [image.shape[0], 1, image.shape[1]])
        _ws = K.batch_dot(_sm, image)
        _r = K.batch_dot(_sm, K.ones_like(image)) + eps
        # Divide the weighted signatures by the sum of weights to get the centroids
        _c = Lambda(lambda inputs: inputs[0] / inputs[1])([_ws, _r])
        _sm = K.reshape(_sm, [_sm.shape[0], 1, _sm.shape[1]])
        reconstructed_image = K.batch_dot(_sm, _c)
        _l = K.sum(K.abs(reconstructed_image - image))
        return _l


    def compile(self, loss=['kld', 'mse', 'sp_loss'], loss_weights=[1, 1, 1], optimizer='adam'):
        loss_weights = [0.25, 0.25, 0.5]
        loss = ['kld', 'mse', self.calculate_spectral_loss]
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)


    def fit(self, x, y=None, labels=None, batch_size=256, maxiter=2e3, tol=1e-3,
            update_interval=140, cae_weights=None, save_dir='./results/temp'):
        save_interval = x.shape[0] / batch_size * 5
        # Step 1: pretrain if necessary
        t0 = time()
        if not self.pretrained and cae_weights is None:
            self.pretrain(x, batch_size, save_dir=save_dir)
            self.pretrained = True
        elif cae_weights is not None:
            self.cae.load_weights(cae_weights)
        # Step 2: initialize cluster centers using k-means
        t1 = time()
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=40)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        # Step 3: deep clustering
        t2 = time()
        loss = [0, 0, 0, 0]
        index = 0
        cont = 0; maxnmi = -1; maxari = -1; maxboth = -1
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _, _ = self.model.predict(x.astype("float32"), verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                if y is not None:
                    yaux      = y[or_labels!=0]#-1
                    y_predaux = self.y_pred[or_labels!=0]#-1
                    acc = np.round(metrics.acc(yaux, y_predaux), 5)
                    nmi = np.round(metrics.nmi(yaux, y_predaux), 5)
                    ari = np.round(metrics.ari(yaux, y_predaux), 5)
                    loss = np.round(loss, 5)
                    if ite > 0:
                        print('Iter', ite, 'loss=', loss)

                    if maxboth <= (nmi+ari)/2:
                        maxboth = (nmi+ari)/2
                        maxnmi  = nmi
                        maxari  = ari
                        cont = 0
                    else:
                        cont += 1
                        if cont == 10: break

            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * batch_size::],
                                                 y=[p[index * batch_size::], x[index * batch_size::]])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                                 y=[p[index * batch_size:(index + 1) * batch_size],
                                                    x[index * batch_size:(index + 1) * batch_size],
                                                    x[index * batch_size:(index + 1) * batch_size]])
                index += 1
            # save intermediate model
            if ite % save_interval == 0:
                self.model.save_weights(save_dir + '/dcec_model_spectral_loss_' + str(ite) + '.h5')
            ite += 1
        # save the trained model
        self.model.save_weights(save_dir + '/dcec_model_final_spectral_loss.h5')
        t3 = time()
        pretrain_time   = np.round(t1 - t0, 2)
        clustering_time = np.round(t3 - t1, 2)
        total_time      = np.round(t3 - t0, 2)
        redstring = "None" if args.redm is None else args.redm
        cadena = args.dataset + " " + str(args.idtest) + " " + redstring + " " + str(pretrain_time) + " " + str(clustering_time) + " " + str(total_time) + " " + str(maxnmi) + " " + str(maxari) + '\n'
        print("DSET", args.dataset, "TEST NUMBER", args.idtest, "RED METHOD", redstring, "TIME[Pretrain, clustering, total]", [pretrain_time, clustering_time, total_time], "NMI", maxnmi, "ARI", maxari)
        with open('results.txt', 'a') as f:
            f.write(cadena)



if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default='IP', choices=['IP', 'PU', 'SV', 'UH18', 'HICO1', 'AVIRIS1'])
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=800, type=int)
    parser.add_argument('--gamma', default=0.1, type=float, help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=10, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--cae_weights', default=None, help='This argument must be given')
    parser.add_argument('--save_dir', default='results/temp')
    parser.add_argument('--idtest', default=0, type=int)
    parser.add_argument('--redm', type=str, default=None)
    parser.add_argument('--ncomp', default=25, type=int)
    parser.add_argument('--noise', default=0, type=float)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    if args.dataset in ['IP', 'PU', 'SV', 'UH18', 'HICO1', 'AVIRIS1']:
        _, or_labels, _, _ = load_hyper(args.dataset, args.ncomp, args.redm, spatialsize=5, remove_bg=False, noise=args.noise)
        x, y, args.n_clusters, _ = load_hyper(args.dataset, args.ncomp, args.redm, spatialsize=5, remove_bg=False, noise=args.noise)
        if args.dataset in ['HICO1']:
            args.n_clusters = 3
        elif args.dataset in ['AVIRIS1']:
            args.n_clusters = 3

    # prepare the DCEC model
    dcec = DCEC(input_shape=x.shape[1:], filters=[32, 32, 32, args.n_clusters], n_clusters=args.n_clusters)
    dcec.model.summary()

    # begin clustering.
    dcec.compile(loss=['kld', 'mse', 'sp_loss'], loss_weights=[args.gamma, 1, 1], optimizer='adam')
    dcec.fit(x, y=y, labels=or_labels, tol=args.tol, maxiter=args.maxiter,
             update_interval=args.update_interval,
             save_dir=args.save_dir,
             cae_weights=args.cae_weights)
