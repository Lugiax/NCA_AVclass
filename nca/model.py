from tensorflow.python.framework import convert_to_constants
from google.protobuf.json_format import MessageToDict
from tensorflow.keras.layers import Conv2D, Dropout
from abc import abstractmethod
import matplotlib.pylab as pl
import tensorflow as tf
import numpy as np
import PIL
import json
import os

from nca.losses import batch_l2_loss
from nca.archs import ARCHS


from typing import Any, List, Union, Optional, Tuple


class Base(tf.keras.Model):
    lr = 1e-3
    lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [30000, 70000], [lr, lr * 0.1, lr * 0.01]
    )
    trainer = tf.keras.optimizers.Adam(lr_sched)

    def __init__(self, config: dict):
        """
        El objeto de configuración es el resultado de la lectura de
        utils.cargar_config. Este devuelve un diccionario leído de un
        archivo YAML.
        :param config: Datos de configuración
        :type config: dict
        """
        super().__init__()
        self.automata_shape = config["AUTOMATA_SHAPE"]
        self.channel_n = config["CHANNEL_N"]
        self.extra_chnl = 0
        if config["AGREGAR_GIRARD"]:
            self.extra_chnl += 3
        elif config["SIN_MASCARA"]:
            self.extra_chnl -= 1
        self.fire_rate = config["CELL_FIRE_RATE"]
        self.add_noise = config["ADD_NOISE"]
        self.capas_config = config["CAPAS"]
        self.config = config

        # Capa de percepción
        self.perceive = tf.keras.Sequential(
            [
                Conv2D(80, 3, activation=tf.nn.relu, padding="SAME"),
            ]
        )

        # Construcción del modelo
        raw_model = self.load_yaml_model(self.capas_config) + [
            Conv2D(
                self.channel_n,
                1,
                activation=None,
                kernel_initializer=tf.zeros_initializer,
            )
        ]
        self.dmodel = tf.keras.Sequential(raw_model)

        self(
            tf.zeros(
                [
                    1,
                    self.automata_shape[0],
                    self.automata_shape[1],
                    self.channel_n + 4 + self.extra_chnl,
                ]
            )
        )  # dummy calls to build the model

    def load_yaml_model(self, yaml_list: List[List[Any, int]]) -> List[Any]:
        """_summary_

        :param yaml_list: Arquitectura de la red en un formato:
                          [[capa1, param1],
                           [capa2, param2],
                           ...]
                         Las capas deben ser especificadas en archs.py
        :type yaml_list: List[List[Any, int]]
        :return: Lista de cada capa de la red
        :rtype: List[Any]
        """
        _capas = []
        for c in yaml_list:
            _capas.append(ARCHS[c[0]](c[1]))
        return _capas

    def guardar_pesos(self, filename):
        self.save_weights(filename)

    def cargar_pesos(self, filename):
        self.load_weights(filename)

    @tf.function
    def train_step(self, x, y):
        iter_n = self.config["N_ITER"]
        with tf.GradientTape() as g:
            for i in tf.range(iter_n):
                x = self(x, training=True)
            loss = batch_l2_loss(self, x, y)
        grads = g.gradient(loss, self.weights)
        grads = [g / (tf.norm(g) + 1e-8) for g in grads]
        self.trainer.apply_gradients(zip(grads, self.weights))
        return x, loss

    @tf.function
    def classify(self, x):
        """
        Devuelve las últimas dos capas ocultas. Estas son las clasificaciones
        """
        return x[:, :, :, -2:]

    @tf.function
    def predict(self, x, binary=False):
        """
        Devuelve las máscaras binarias de las clasificaciones.
        """
        x0 = self.initialize(x)
        for _ in range(20):
            x0 = self(x0)

        y_pred = self.classify(x0)
        if not binary:
            return y_pred
        else:
            mask = x[..., -1]
            is_vessel = tf.cast(mask > 0.1, tf.float32)
            y_pred = y_pred * tf.expand_dims(is_vessel, -1)  # Delimita los vasos

            return y_pred > 0

    @abstractmethod
    @tf.function
    def initialize(self):
        pass

    @abstractmethod
    @tf.function
    def call(self):
        pass

    @abstractmethod
    def fit(self):
        pass


class Masked(Base):
    @tf.function
    def initialize(self, images, normalize=True):
        state = tf.zeros(
            [
                images.shape[0],
                self.automata_shape[0],
                self.automata_shape[1],
                self.channel_n,
            ]
        )
        images = tf.image.resize(images, self.automata_shape, antialias=True)
        images = tf.reshape(
            images,
            [-1, self.automata_shape[0], self.automata_shape[1], 4 + self.extra_chnl],
        )
        return tf.cast(tf.concat([images, state], -1), tf.float32)

    @tf.function
    def call(self, x, fire_rate=None, manual_noise=None, training=False):
        self.training = training
        img_norm, gray, state = tf.split(
            x, [3 + self.extra_chnl, 1, self.channel_n], -1
        )
        ds = self.dmodel(self.perceive(x))
        if self.add_noise:
            if manual_noise is None:
                residual_noise = tf.random.normal(tf.shape(ds), 0.0, 0.02)
            else:
                residual_noise = manual_noise
            ds += residual_noise

        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = (
            tf.random.uniform(
                tf.shape(x[..., 3 + self.extra_chnl : 4 + self.extra_chnl])
            )
            <= fire_rate
        )
        living_mask = gray > 0.1
        residual_mask = update_mask & living_mask
        ds *= tf.cast(residual_mask, tf.float32)
        state += ds

        return tf.concat([img_norm, gray, state], -1)

    @abstractmethod
    def fit(self, x_train, y_train_pic):
        loss_log = []
        for i in range(self.config["EPOCHS"]):
            b_idx = np.random.randint(
                0, x_train.shape[0] - 1, size=self.config["BATCH_SIZE"]
            )
            x0 = self.initialize(tf.gather(x_train, b_idx))
            y0 = tf.gather(y_train_pic, b_idx)

            x, loss = self.train_step(x0, y0)

            step_i = len(loss_log)
            loss_log.append(loss.numpy())

            if step_i % 100 == 0:
                with open(os.path.join(self.config["SAVE_DIR"], "loss.txt"), "w") as f:
                    f.write(", ".join([str(i) for i in loss_log]))
                save_plot_loss(loss, self.config)
                save_batch_vis(self, x0, y0, x, step_i, self.config)

            if step_i % 5000 == 0:
                self.guardar_pesos(
                    os.path.join(self.config["SAVE_DIR"], "model/%07d" % step_i)
                )

            print(
                "\r step: %d, log10(loss): %.3f" % (len(loss_log), np.log10(loss)),
                end="",
            )

        self.guardar_pesos(os.path.join(self.config["SAVE_DIR"], "model/last"))


class Unmasked(Base):
    @tf.function
    def initialize(self, images, normalize=True):
        state = tf.zeros(
            [
                images.shape[0],
                self.automata_shape[0],
                self.automata_shape[1],
                self.channel_n,
            ]
        )
        images = tf.image.resize(images, self.automata_shape, antialias=True)
        images = tf.reshape(
            images,
            [-1, self.automata_shape[0], self.automata_shape[1], 4 + self.extra_chnl],
        )
        return tf.cast(tf.concat([images, state], -1), tf.float32)

    @tf.function
    def call(self, x, fire_rate=None, manual_noise=None, training=False):
        self.training = training
        # self.extra_chnl debe ser -1 (sin máscara) o 2 (girard)
        img_norm, state = tf.split(x, [4 + self.extra_chnl, self.channel_n], -1)
        ds = self.dmodel(self.perceive(x))
        if self.add_noise:
            if manual_noise is None:
                residual_noise = tf.random.normal(tf.shape(ds), 0.0, 0.02)
            else:
                residual_noise = manual_noise
            ds += residual_noise

        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = tf.random.uniform(tf.shape(x[..., :1])) <= fire_rate
        living_mask = tf.expand_dims(tf.math.reduce_max(state[..., -2:], axis=-1), -1)
        dilated_mask = tf.nn.dilation2d(
            living_mask,
            filters=tf.ones((3, 3, living_mask.shape[-1])),
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC",
            dilations=[1, 1, 1, 1],
        )
        residual_mask = update_mask & (dilated_mask > 0)
        ds *= tf.cast(residual_mask, tf.float32)
        state += ds
        concatenada = tf.concat([img_norm, state], -1)
        return concatenada

    def fit(self, x_train, y_train_pic):
        loss_log = []
        for i in range(self.config["EPOCHS"]):
            b_idx = np.random.randint(
                0, x_train.shape[0] - 1, size=self.config["BATCH_SIZE"]
            )
            x0 = self.initialize(tf.gather(x_train, b_idx))
            y0 = tf.gather(y_train_pic, b_idx)

            x, loss = self.train_step(x0, y0)

            step_i = len(loss_log)
            loss_log.append(loss.numpy())

            if step_i % 100 == 0:
                with open(os.path.join(self.config["SAVE_DIR"], "loss.txt"), "w") as f:
                    f.write(", ".join([str(i) for i in loss_log]))
                save_plot_loss(loss, self.config)
                save_batch_vis(self, x0, y0, x, step_i, self.config)

            if step_i % 5000 == 0:
                self.guardar_pesos(
                    os.path.join(self.config["SAVE_DIR"], "model/%07d" % step_i)
                )

            print(
                "\r step: %d, log10(loss): %.3f" % (len(loss_log), np.log10(loss)),
                end="",
            )

        self.guardar_pesos(os.path.join(self.config["SAVE_DIR"], "model/last"))

## Funciones auxiliares
color_lookup = tf.constant([
            [128, 0, 0],  #Para arterias, Rojo
            [0, 128, 128], #Para venas, Azul
            [0, 0, 0], # This is the default for digits/vasos sanguíneos
            [255, 255, 255] # This is the background.
            ])

backgroundWhite = True
def color_labels(x, y_pic, disable_black=False, dtype=tf.uint8):
    # works for shapes of x [b, r, c] and [r, c]
    if x.shape[-1]==4:
        mask = x[..., -1]
    elif x.shape[-1]==3:
        mask = tf.math.reduce_max(y_pic, axis=-1)
    black_and_white = tf.fill(list(mask.shape) + [2], 0.01)
    is_gray = tf.cast(mask > 0.1, tf.float32)
    is_not_gray = 1. - is_gray
    y_pic = y_pic * tf.expand_dims(is_gray, -1) # forcibly cancels everything outside of it.
  
    # if disable_black, make is_gray super low.
    if disable_black:
        is_gray *= -1e5
        # this ensures that you don't draw white in the digits.
        is_not_gray += is_gray

    bnw_order = [is_gray, is_not_gray] if backgroundWhite else [is_not_gray, is_gray]
    black_and_white *= tf.stack(bnw_order, -1)

    rgb = tf.gather(
      color_lookup,
      tf.argmax(tf.concat([y_pic, black_and_white], -1), -1))
    if dtype == tf.uint8:
        return tf.cast(rgb, tf.uint8)
    else:
        return tf.cast(rgb, dtype) / 255.


def classify_and_color(ca, x, y0=None, disable_black=False):
    if y0 is not None:
        return color_labels(
                x[:,:,:,:4], y0, disable_black, dtype=tf.float32)
    else:
        return color_labels(
            x[:,:,:,:4], ca.classify(x), disable_black, dtype=tf.float32)
        


def save_plot_loss(loss_log, config):
    pl.figure(figsize=(6, 3))
    pl.title('Loss history (log10)')
    pl.plot(np.log10(loss_log), '.', alpha=0.4)
    pl.xlabel('Epoch')
    pl.ylabel(f'log10(L)')
    pl.savefig(os.path.join(config['SAVE_DIR'], 'figures', 'loss.png'))


def np2pil(a):
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1)*255)
    return PIL.Image.fromarray(a)

def imwrite(f, a, fmt=None):
    a = np.asarray(a)
    if isinstance(f, str):
        fmt = f.rsplit('.', 1)[-1].lower()
        if fmt == 'jpg':
            fmt = 'jpeg'
        f = open(f, 'wb')
    np2pil(a).save(f, fmt, quality=95)

def save_batch_vis(ca, x0, y0, x, step_i, config):
    vis_1 = np.hstack(x0[..., :3])
    vis0 = np.hstack(classify_and_color(ca, x0, y0).numpy())
    vis1 = np.hstack(classify_and_color(ca, x).numpy())
    vis = np.vstack([vis_1, np.hstack(x0[..., 3:6]), vis0, vis1])\
                     if config['AGREGAR_GIRARD'] else np.vstack([vis_1, vis0, vis1])
    imwrite(os.path.join(config['SAVE_DIR'], 'figures', 'batches_%04d.jpg'%step_i), vis)


def export_model(ca, base_fn, args):
    ca.save_weights(base_fn)
    with open(os.path.split(base_fn)[0]+'config.txt','w') as f:
        f.write(f"""
CHANNEL_N = {args.n_channels}
CELL_FIRE_RATE = Revisar en automata. Generalmente 0.5
N_ITER = {args.n_iter}
BATCH_SIZE = {args.n_batch}
ADD_NOISE = {args.add_noise}
DATA = {args.db_name}
AGREGAR_GIRARD = {args.agregar_girard}
SELEM_SIZE = {args.selem_size if args.agregar_girard else None}


Arquitectura:
{args.model_name},
Conv2D(self.channel_n, 1, activation=None,
            kernel_initializer=tf.zeros_initializer),
            """)

    cf = ca.call.get_concrete_function(
        x=tf.TensorSpec([None, None, None, args.n_channels+4+args.extra_channels]),
        fire_rate=tf.constant(0.5),
        manual_noise=tf.TensorSpec([None, None, None, args.n_channels]))
    cf = convert_to_constants.convert_variables_to_constants_v2(cf)
    graph_def = cf.graph.as_graph_def()
    graph_json = MessageToDict(graph_def)
    graph_json['versions'] = dict(producer='1.14', minConsumer='1.14')
    model_json = {
        'format': 'graph-model',
        'modelTopology': graph_json,
        'weightsManifest': [],
    }
    with open(base_fn+'.json', 'w') as f:
        json.dump(model_json, f)

if __name__ == "__main__":
    print("HOLA!!!")
