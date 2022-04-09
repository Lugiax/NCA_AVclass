#Pruebas del generador de datos
#Usar https://www.tensorflow.org/api_docs/python/tf/data/Dataset
#REVISAR https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
#Se hace aumento de datos

#Buscar en tf data augmentation


import tensorflow as tf
from tensorflow_datasets.typing import Tensor
from keras import layers
from pathlib import Path
from typing import Iterator, Tuple
from PIL import Image
import numpy as np
import pdb


from dataclasses import dataclass, field
@dataclass(frozen=True)
class DSFactory:
    """Generador de imágenes para el entrenamiento. Recibe como
    parámetros de inicialización el directorio de las imágenes
    que deben estar en una carpeta con la nomenclatura:

        img_dir/
            - f_xxxx.xxxx #para la imagen rgb de fondo de ojo
            - a_xxxx.xxxx #para la máscara de las arterias
            - v_xxxx.xxxx #para la máscara de ls venas

    :param imgs_dir: Ruta hacia el conjunto de datos
    :type imgs_dir: str
    :param output_shape: Tamaño de las imágenes de salida
    :type output_shape: Tuple[int, int]
    """
    imgs_dir: str
    output_shape: Tuple[int, int] = field(default_factory=tuple)
    
    @staticmethod
    def _augment_spatial_gen(crop_size:Tuple[int, int],
                             output_shape: Tuple[int, int]
                             )-> tf.keras.layers.Layer:
        """Generador de la capa de aumento de datos espacial.

        :param crop_size: Tamaño del corte de la imagen. Al ser imágenes de altas
        resoluciones, se recomienda valores de cropping altos.
        :type crop_size: Tuple[int, int]
        :param output_shape: Tamaño de las imágenes de salida
        :type output_shape: Tuple[int, int]
        :return: Capa de keras para hacer la transformación espacial en las imágenes
        :rtype: tf.keras.layers.Layer
        """
        spatial_augmentation = [
            layers.RandomFlip(),
            layers.RandomRotation(factor=0.2,
                                  fill_mode='reflect'),
            layers.RandomCrop(*crop_size),
            layers.Resizing(*output_shape)
        ]
        return tf.keras.Sequential(spatial_augmentation)
        
        
    @staticmethod
    def _augment_color_gen() -> tf.keras.layers.Layer:
        """Capa para realizar un aumento de datos en las imágenes RGB.

        :return: Capa de keras para la transformación radiométrica
        :rtype: tf.keras.layers.Layer
        """
        rgb_augmentation = [
            layers.RandomContrast(factor = 0.2)
        ]
        return tf.keras.Sequential(rgb_augmentation)
        
        
    def generar(self, n_imgs:int, n_augment:int, crop_size:float) -> Tensor:
        """
        Devuelve n_imgs*n_augment=N número de imágenes con sus 
        respectivas máscaras de venas y arterias en un tensor de 
        forma NxHxWxC donde H y W es el alto y ancho de las imágenes
        y C son los canales de las imágenes: Las 3 primeras de la
        imagen RGB, la 4ta de la máscara de arterias y la 5ta la 
        máscara de venas.

        :param n_imgs: Número de imágenes diferentes a leer
        :type n_imgs: int
        :param n_augment: Cantidad de imágenes aumentadas
        :type n_augment: int
        :param crop_size: Tamaño del corte sobre la imagen original
        :type crop_size: float
        :return: Tensor de tamaño NxHxWx5 con los datos leídos
        y aumentados
        :rtype: Tensor
        """
        augment_spatial = self._augment_spatial_gen(crop_size, self.output_shape)
        augment_rgb = self._augment_color_gen()
    
        data = []
        for data_imgs in self.dispatch_images(n_imgs):
            color_data = data_imgs[..., :3]
            masks_data = data_imgs[..., 3:]
            for _ in range(n_augment):
                color_augmented = augment_rgb(color_data)
                spatial_augmented = augment_spatial(
                                        tf.concat([color_augmented, masks_data],
                                                axis = -1)
                                    )
                data.append(spatial_augmented)

        return tf.stack(data)
            
    
    def dispatch_images(self, n_imgs:int = -1) -> Iterator:
        """Lee y genera las imágenes que serán utilizadas para
        el dataset

        :param n_imgs: Número de imágenes a utilizar. Si el valor
        es igual a -1, se leen todas las imágenes disponibles.
        Defaults to -1
        :type n_imgs: int, optional
        :return: Generador de imágenes
        :rtype: Iterator
        :yield: Tensor de HxWx5 donde los primeros 3 canales
        corresponden a la imagen RGB, el cuarto a la máscara
        de venas y el quinto a la de venas.
        :rtype: Iterator[Tensor]
        """
        
        p = Path(self.imgs_dir)
        fundus_imgs = sorted(p.glob('f_*.*'))
        artery_imgs = sorted(p.glob('a_*.*'))
        vein_imgs   = sorted(p.glob('v_*.*'))
        
        assert len(fundus_imgs)==len(artery_imgs)==len(vein_imgs),\
            'Debe haber solo dos máscaras por cada imagen de fondo de ojo'
        
        idxs = np.random.randint(len(fundus_imgs), size=n_imgs)
        
        for idx in idxs:
            farr = np.asarray(Image.open(fundus_imgs[idx]), dtype=np.float32)/255.
            aarr = np.asarray(Image.open(artery_imgs[idx]), dtype=np.float32)[..., None]
            varr = np.asarray(Image.open(vein_imgs[idx]), dtype=np.float32)[..., None]
            
            yield np.concatenate([farr, aarr, varr], axis=-1)
            del farr, aarr, varr
            
            

if __name__=='__main__':
    print('Probando generador de datasets')
    p = DSFactory(imgs_dir='data/', output_shape=(100,100))
    d = p.generar(n_imgs=1, n_augment=3, crop_size=(1000,1000))
    import matplotlib.pyplot as plt
    for i in range(d.shape[0]):
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(d[i, ..., :3])
        ax[1].imshow(d[i, ..., 3], 'gray')
        ax[2].imshow(d[i, ..., 4], 'gray')
        plt.show()
