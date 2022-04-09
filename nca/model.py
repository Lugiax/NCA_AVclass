from keras.layers import Conv2D
from tensorflow_datasets.typing import Tensor
import matplotlib.pylab as pl
import tensorflow as tf
import numpy as np
import wandb

from nca.losses import batch_l2_loss
from nca.archs import ARCHS
from nca.dataset_gen import DSFactory


from typing import List, TypedDict, Iterable, Any, Tuple, Optional

###  Annotation types

class NCAConfig(TypedDict):
    #Arquitectura del autómata
    channel_n: int #Número de canales ocultos
    cell_fire_rate: float #0-1
    iter_n: int #Iteraciones del autómata
    automata_shape: List[int] #[H,W]
    
    #Parámetros de entrenamiento
    batch_imgs_n: int #imágenes diferentes por batch
    image_augment_n: int #aumento de datos por imagen
    crop_shape: List[int] #Dimensiones del recorte 
                          #principal, [H,W]
    epochs: int #número de épocas

    #Arquitectura red
    capas: List[List[Any]] #Lista de listas con la configuración
                           #de la red.

###


class Base(tf.keras.Model):
    lr = 1e-3
    lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [30000, 70000], [lr, lr * 0.1, lr * 0.01]
    )
    trainer = tf.keras.optimizers.Adam(lr_sched)

    def __init__(self, config: NCAConfig):
        super().__init__()
        self.base_config = config

        # Capa de percepción
        self.perceive = tf.keras.Sequential(
            [
                Conv2D(80, 3, activation=tf.nn.relu, padding="SAME"),
            ]
        )

        # Construcción del modelo
        raw_model = self.load_yaml_model(self.base_config['capas']) + [
            Conv2D(
                self.base_config['channel_n'],
                1,
                activation=None,
                kernel_initializer=tf.zeros_initializer,
            )
        ]
        self.dmodel = tf.keras.Sequential(raw_model)

        #LLamada para construir el modelo
        self(
            tf.zeros(
                [
                    1,
                    self.base_config['automata_shape'][0],
                    self.base_config['automata_shape'][1],
                    self.base_config['channel_n'] + 3
                ]
            )
        ) 
        
        
    @staticmethod
    def load_yaml_model(yaml_list: list):
        """Función para 

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

    def guardar_pesos(self, filename:str):
        self.save_weights(filename)

    def cargar_pesos(self, filename:str):
        self.load_weights(filename)

    @tf.function
    def train_step(self, x: Tensor, y: Tensor) -> Tuple[Tensor, float]:
        """Paso de entrenamiento. Toma los dos tensores X y Y e
        itera sobre ellos, cambiando sus valores según la función
        de transición.

        :param x: Datos de entrenamiento independientes
        :type x: Tensor
        :param y: Etiquetas de entrenamiento de X
        :type y: Tensor
        :return: Tupla con: El último estados de X, y el valor de pérdida
        promedio del batch
        :rtype: Tuple[Tensor, float]
        """        
        
        with tf.GradientTape() as g:
            for i in tf.range(self.base_config["iter_n"]):
                x = self(x, training=True)
            loss = batch_l2_loss(self, x, y)
        grads = g.gradient(loss, self.weights)
        grads = [g / (tf.norm(g) + 1e-8) for g in grads]
        self.trainer.apply_gradients(zip(grads, self.weights))
        return x, loss

    @tf.function
    def classify(self, x: Tensor)->Tensor:
        """
        Devuelve las últimas dos capas ocultas. Estas son las clasificaciones
        """
        return x[:, :, :, -2:]

    @tf.function
    def predict(self, x: Tensor, binary:bool=False) -> Tensor:
        """
        Devuelve las máscaras binarias de las clasificaciones.
        """
        x0 = self.initialize(x)
        for _ in range(self.base_config['iter_n']):
            x0 = self(x0)

        y_pred = self.classify(x0)
        if not binary:
            return y_pred
        else:
            mask = x[..., -1]
            is_vessel = tf.cast(mask > 0.1, tf.float32)
            y_pred = y_pred * tf.expand_dims(is_vessel, -1)  # Delimita los vasos

            return y_pred > 0

    
    @tf.function
    def initialize(self):
        raise NotImplementedError

    @tf.function
    def call(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError




class NCA(Base):
    @tf.function
    def initialize(self, images: Tensor)->Tensor:
        """Función para inicializar el lote de entrada. 
        Aquí se crean las capas ocultas y se organizan las
        capas para el correcto funcionamiento del autómata

        :param images: Tensor de imágenes de entrenamiento
        :type images: Tensor
        :return: Tensor de estado del autómata
        :rtype: Temsor
        """        
        automata_shape = self.base_config['automata_shape']
        state = tf.zeros(
            [
                images.shape[0],
                automata_shape[0],
                automata_shape[1],
                self.base_config['channel_n'],
            ]
        )
        images = tf.image.resize(images, automata_shape, antialias=True)
        images = tf.reshape(
            images,
            [-1, automata_shape[0], automata_shape[1], 3],
        )
        return tf.cast(tf.concat([images, state], -1), tf.float32)

    @tf.function
    def call(self, x:Tensor, fire_rate: Optional[float]=None, training:bool=False)->Tensor:
        """Función de llamada del autómata. Aquí se define cómo
        se llevará a cabo cada paso del autómata. Los datos se
        hacen pasar primero por la capa de percepción, luego 
        se evalua la función de transición y se hace un tratado
        de los datos para actuallizar el estado.

        :param x: Datos de entrada (estado previo del autómata)
        :type x: Tensor
        :param fire_rate: Umbral mínimo para el encendido de las
        células, predefinido a None
        :type fire_rate: Optional[float], optional
        :param training: Parámetro para definir si el proceso es de
        entrenamiento, predefinido a False
        :type training: bool, optional
        :return: Tensor del estado del autómata actualizado
        :rtype: Tensor
        """        
        self.training = training
        img_norm, state = tf.split(x, [3, self.base_config['channel_n']], -1)
        ds = self.dmodel(self.perceive(x))
        if self.base_config['add_noise']:
            ds += tf.random.normal(tf.shape(ds), 0.0, 0.02)

        if fire_rate is None:
            fire_rate = self.base_config['cell_fire_rate']
            
        update_mask = tf.random.uniform(tf.shape(x[..., :1])) <= fire_rate
        living_mask = tf.expand_dims(tf.math.reduce_max(state[..., -2:], axis=-1), -1)
        
        #Dilation for removing noise
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

    def fit(self, data_factory:Iterable):
        """Proceso de entrenamiento del autómata. Como único
        parámetro necesita un generador de datos. Así como está
        definido en el dataset_gen.

        :param data_factory: Generador de datos
        :type data_factory: Iterable
        """        
        #data_factory = DSFactory(self.base_config['imgs_dir'], self.base_config['automata_shape'])
        loss_log = []
        for i in range(self.base_config["epochs"]):
            print(f'Época {i+1}, loss:', end='')
            train_data = data_factory.generar(self.base_config['batch_imgs_n'],
                                              self.base_config['image_augment_n'],
                                              self.base_config['crop_shape'])
            
            x0 = self.initialize(train_data[..., :3])
            y0 = train_data[..., 3:]

            x, loss = self.train_step(x0, y0)
            print(loss)

            step_i = len(loss_log)
            loss_log.append(loss.numpy())

            """if step_i % 100 == 0:
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
        """

if __name__ == "__main__":
    from nca.utils import cargar_config
    from PIL import Image
    
    print("Probando modelo")
    
    config = cargar_config('data/base_inference_model.yaml')
    model = NCA(config)
    
    img = Image.open('data/f_001.jpg')
    img_arr = np.asarray(img)/255.
    x = tf.constant(img_arr[None, ...])
    y = model.predict(x)
    import matplotlib.pyplot as plt
    print(y.shape)
    plt.imshow(y[0, ..., -1], 'gray')
    plt.show()
    plt.imshow(y[0, ..., -2], 'gray')
    plt.show()