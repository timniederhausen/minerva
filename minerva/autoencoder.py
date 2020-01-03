import logging
import typing

from ConfigSpace.conditions import InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter
from tensorflow.python.keras import backend as K, Input, Model
from tensorflow.python.keras.callbacks import Callback, History
from tensorflow.python.keras.layers import Layer, Dense, Lambda
from tensorflow.python.keras.losses import binary_crossentropy
import numpy as np
import tensorflow as tf

from minerva.util import make_curved_topology

log = logging.getLogger(__name__)


class UpdateKSparseLevel(Callback):
    r"""Update sparsity level at the beginning of each epoch.

    This callback replaces the |k| value of each KSparse layer at the start of every epoch.
    """

    def on_epoch_begin(self, epoch, logs=None):
        for layer in self.model.layers:
            if layer.name.startswith('KSparse'):
                K.set_value(layer.k, int(layer.sparsity_levels[epoch]))


class KSparse(Layer):
    r"""k-sparse Keras layer.

    .. seealso:

       * https://arxiv.org/pdf/1312.5663.pdf
    """

    def __init__(self, sparsity_levels: np.ndarray, k: typing.Optional[tf.Variable], **kwargs):
        r"""Construct a new k-sparse layer.

        :param sparsity_levels: Per-epoch sparsity levels for this layer.
        :param k: Optional tensorflow variable for the layer's current sparsity level.
                  Using this is recommended if the network's big and |k| is the same for all layers.
        :param kwargs: Additional :class:`tensorflow.python.keras.layers.Layer` parameters.
        """
        super().__init__(trainable=False, **kwargs)
        self.sparsity_levels = sparsity_levels
        if k is None:
            k = tf.Variable(sparsity_levels[0], dtype=tf.int32, name='k')
        self.k = k

    def call(self, inputs, mask=None):
        def sparse():
            # number of dimensions in input might be < |k|. account for that
            actual_k = tf.minimum(K.shape(inputs)[-1] - 1, self.k)
            # multiply all values greater than the k smallest with 1, the rest with 0
            kth_smallest = tf.sort(inputs)[..., K.shape(inputs)[-1] - 1 - actual_k]
            return inputs * K.cast(K.greater(inputs, kth_smallest[:, None]), K.floatx())

        return K.in_train_phase(sparse, inputs)

    def get_config(self):
        config = {'sparsity_levels': self.sparsity_levels.tolist()}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def make_deep_autoencoder(dims, act='relu', init='glorot_uniform',
                          layer_decorator: typing.Optional[typing.Callable[[Layer, bool], Layer]] = None,
                          compile=True) -> typing.Tuple[Model, Model]:
    r"""Build a fully-connected symmetric autoencoder model.

    :param dims: Per-layer neuron counts for the entire AE.
    :param act: Activation function for all neurons (except input and output layers)
    :param init: Initialization for the weights of every neuron.
    :param layer_decorator: Callable that can modify every layer of the network.
    :param compile: Compile the model now or leave it up to the caller.
    :return: (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    input_img = Input(shape=(dims[0],), name='input')
    n_stacks = len(dims) - 1
    assert n_stacks > 0

    x = input_img

    # internal layers in encoder
    for i in range(n_stacks - 1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)
        if layer_decorator:
            x = layer_decorator(x, is_output=False)

    # encoder output layer (the last hidden encoding layer in the autoencoder)
    # features are extracted from here
    x = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)
    if layer_decorator:
        x = layer_decorator(x, is_output=False)
    encoded = x

    # internal layers in decoder
    for i in range(n_stacks - 1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)
        if layer_decorator:
            x = layer_decorator(x, is_output=False)

    # output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    if layer_decorator:
        x = layer_decorator(x, is_output=True)
    decoded = x

    encoder = Model(inputs=input_img, outputs=encoded, name='encoder')
    autoencoder = Model(inputs=input_img, outputs=decoded, name='AE')
    if compile:
        autoencoder.compile(optimizer='adadelta', loss='mse')
    return autoencoder, encoder


def calculate_sparsity_levels(initial_sparsity, final_sparsity, n_epochs):
    r"""Calculate sparsity levels per epoch.

    :param initial_sparsity: Initial value of the linear sampling.
    :param final_sparsity: Final value of the linear sampling.
    :param n_epochs: Number of epochs the network will be trained for.
    :return: An array containing sparsity levels for all epochs.
    """
    return np.hstack((np.linspace(initial_sparsity, final_sparsity, n_epochs // 2, dtype=np.int),
                      np.repeat(final_sparsity, (n_epochs // 2) + 1)))[:n_epochs]


def make_ksparse_autoencoder(dims, sparsity_levels, **kwargs) -> typing.Tuple[Model, Model]:
    r""":func:`.make_deep_autoencoder` for k-sparse autoencoders.

    :param dims: Per-layer neuron counts for the entire AE.
    :param sparsity_levels: Per-epoch sparsity levels for this layer.
    :param kwargs: Additional parameters that can override AE properties (e.g. activation function).
    :return: (autoencoder, encoder)-tuple
    """
    sparsity_levels = np.array(sparsity_levels)
    k = tf.Variable(sparsity_levels[0], dtype=tf.int32, name='k')

    def sparsity_regularizer(layer, is_output):
        if is_output:
            return layer
        name = 'KSparse_' + layer.name.split('/', 1)[0]
        return KSparse(sparsity_levels=sparsity_levels, k=k, name=name)(layer)

    return make_deep_autoencoder(dims, layer_decorator=sparsity_regularizer, **kwargs)


# from Keras VAE sample:

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    r""" Reparameterization trick by sampling from an isotropic unit Gaussian.

    :param args: mean and log of variance of Q(z|X)
    :return: sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def make_variational_autoencoder(original_dim, intermediate_dim=512, latent_dim=2,
                                 reconstruction_loss_fn=binary_crossentropy):
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=(original_dim,), name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, z, name='encoder')

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs))
    vae = Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = reconstruction_loss_fn(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return vae, encoder


KSPARSE_PRUNE_FACTOR = 3
BATCH_SIZE = 256


def calc_layer_sizes(x_shape: tuple, cfg: dict) -> typing.List[int]:
    r"""Guess the most suitable layer sizes for a given Configuration and dataset shape.

    :param x_shape: Shape of the input dataset the AE will be trained on.
    :param cfg: ConfigurationSpace values that will be used for this AE.
    :return: Per-layer neuron counts for the entire AE.
    """
    input_dim = x_shape[-1]
    latent_dim = int(cfg['latent_dim'])
    num_hidden_layers = cfg['num_hidden_layers']
    # TODO: other topologies?
    # +1 to account for fixed-size input layer
    return make_curved_topology(input_dim, latent_dim, 1 + num_hidden_layers)


def build_autoencoder(cfg: dict, x_shape: typing.Optional[tuple] = None,
                      dims: typing.Optional[typing.List[int]] = None,
                      **kwargs) -> (Model, Model):
    r"""Build an AE model based on the given ConfigurationSpace values.

    :param cfg: ConfigurationSpace values that shall be used for this AE.
    :param x_shape: Shape of the dataset this AE will be trained on. Optional if :paramref:`dims` is set.
    :param dims: Per-layer neuron counts for the entire AE. Optional if :paramref:`x_shape` is set.
    :param kwargs: Additional parameters that can override AE properties (e.g. activation function).
    :return: The desired autoencoder model.
    """
    if dims is None:
        assert x_shape is not None
        dims = calc_layer_sizes(x_shape, cfg)

    if cfg['ae_type'] == 'deep_ksparse':
        sparsity_levels = calculate_sparsity_levels(cfg['k'], cfg['k'] // KSPARSE_PRUNE_FACTOR, cfg['epochs'])
        return make_ksparse_autoencoder(dims, sparsity_levels, **kwargs)
    else:
        return make_deep_autoencoder(dims, **kwargs)


def train_autoencoder(x: np.ndarray, cfg: dict, autoencoder: Model) -> History:
    r"""Train an already built AE model on new data.

    :param x: The data the AE shall be trained on.
    :param cfg: ConfigurationSpace values that were used to construct this AE.
    :param autoencoder: The constructed AE model.
    :return: The training history.
    """
    callbacks = None
    if cfg['ae_type'] == 'deep_ksparse':
        callbacks = [UpdateKSparseLevel()]
    return autoencoder.fit(x, x, callbacks=callbacks, epochs=cfg['epochs'], batch_size=BATCH_SIZE)


def build_and_train_autoencoder(x: np.ndarray, cfg: dict) -> (Model, Model, History):
    r"""Helper function to build and train an autoencoder with only one call.

    .. seealso::

        :func:`build_autoencoder`
        :func:`train_autoencoder`

    :param x: The input dataset.
    :param cfg: ConfigurationSpace values that shall be used for this AE.
    :return: (autoencoder, encoder, training history)-tuple
    """
    autoencoder, encoder = build_autoencoder(cfg, x_shape=x.shape)
    return autoencoder, encoder, train_autoencoder(x, cfg, autoencoder)


def _get_layer_index(layer: Layer) -> int:
    r"""Get the index of a layer relative to the network part it's in.

    :param layer: The layer whose index shall be retrieved.
    :return: Index relative to the layer's network part.
    """
    if layer.name.startswith('encoder_'):
        return int(layer.name[8:])
    if layer.name.startswith('decoder_'):
        return int(layer.name[8:])
    raise RuntimeError('Cannot compute layer index for ' + layer.name)


def build_autoencoder_from_existing(old_x_shape: tuple, new_x_shape: tuple,
                                    old_weights: str, cfg: dict,
                                    freeze_weights=False) -> typing.Tuple[Model, Model]:
    r"""Build an autoencoder for a new dataset, based on an existing AE for a different DS.

    The shape of the new AE's neural network will be as close as possible to the shape of the old AE.

    :param old_x_shape: Shape of the previous dataset.
    :param new_x_shape: Shape of the new dataset.
    :param old_weights: Weights for an AE trained on the old dataset.
    :param cfg: ConfigurationSpace values that shall be used for both AEs.
    :param freeze_weights: If true, the first half of copied weights gets frozen.
    :return: (autoencoder, encoder)-tuple
    """
    oldae, olde = build_autoencoder(cfg, x_shape=old_x_shape)

    dims = calc_layer_sizes(old_x_shape, cfg)
    dims[0] = new_x_shape[-1]
    newae, newe = build_autoencoder(cfg, dims=dims, compile=False)

    oldae.load_weights(old_weights)

    old_layers: typing.Dict[str, Layer] = {layer.name: layer for layer in oldae.layers}
    half_size = len(dims) // 2
    for layer in newae.layers:  # type: Layer
        if layer.trainable and layer.name not in ('input', 'encoder_0', 'decoder_0',):
            try:
                layer.trainable = freeze_weights and _get_layer_index(layer) >= half_size
                layer.set_weights(old_layers[layer.name].get_weights())
            except Exception:
                log.exception('Failed to copy layer weights for %s', layer.name)

    newae.compile(optimizer='adadelta', loss='mse')
    return newae, newe


def make_config_space(dataset_shape: tuple) -> ConfigurationSpace:
    r"""Build a ConfigurationSpace object encompassing all different autoencoder types and parameters.

    The resulting ConfigurationSpace object depends on the given dataset shape.
    Adapting it to different datasets requires manual adjustments (see :func:`build_autoencoder_from_existing`).

    :param dataset_shape: Shape of the dataset this ConfigurationSpace should be valid for.
    :return: A ConfigurationSpace object tied to the given dataset size.

    .. note:

       Changes to this function invalidate previous optimization results.
    """
    input_dim = dataset_shape[-1]

    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()

    ae_type = CategoricalHyperparameter('ae_type', ['deep', 'deep_ksparse'], default_value='deep_ksparse')
    act_type = CategoricalHyperparameter('act_type', ['relu', 'sigmoid', 'tanh'], default_value='relu')
    epochs = UniformIntegerHyperparameter('epochs', 1, 50, default_value=10)
    cs.add_hyperparameters([ae_type, act_type, epochs])

    num_layers = UniformIntegerHyperparameter('num_hidden_layers', 1, 5, default_value=3)
    latent_dim = UniformIntegerHyperparameter('latent_dim', 2, input_dim // 2, default_value=2)
    cs.add_hyperparameters([num_layers, latent_dim])

    ksparse_k = UniformIntegerHyperparameter('k', 50, 200, default_value=200)
    cs.add_hyperparameter(ksparse_k)
    cs.add_condition(InCondition(child=ksparse_k, parent=ae_type, values=['deep_ksparse']))
    return cs
