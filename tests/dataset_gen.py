#Pruebas del generador de datos
#Usar https://www.tensorflow.org/api_docs/python/tf/data/Dataset
#Se hace aumento de datos

#Buscar en tf data augmentation


import tensorflow as tf
from tensorflow_datasets.typing import Tensor
from typing import Generator, Tuple


class DSLoader(tf.data.Dataset):
    @staticmethod
    def _generator(img_dir:str, n_samples:int, n_augmentations:int,
                output_img_shape:int) -> Tuple[Tensor, Tensor]:
        pass
        
    
    @staticmethod
    def _read_files(img_dir:str):
        pass
    
    def __new__(cls, img_dir:str, n_samples:int, n_augmentations:int,
                output_img_shape:int=1000):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature = (
                tf.TensorSpec((n_samples, output_img_shape, output_img_shape, 3),
                              dtype=tf.float32),
                tf.TensorSpec((n_samples, output_img_shape, output_img_shape, 2),
                              dtype=tf.float32)
            ),
            args = (img_dir, n_samples, n_augmentations, output_img_shape)
            
        )