from keras.layers import Conv2D, Activation, Add
import tensorflow as tf

class bloque_res_simple(tf.keras.layers.Layer):
    """Bloque tipo Resnet simple. El proceso se bifurca
    después de la primera capa de convolución, para al final
    obener la suma de ambas bifuraciones.
    """
    def __init__(self, size=None):
        super(bloque_res_simple, self).__init__()
        #self.padre = padre #El modelo al que pertenece

        self.bloq_0 = Conv2D(size, 1, padding="SAME")

        self.bloq_1 = Conv2D(size, 1, padding="SAME")
        self.activ_1 = Activation('relu')

        self.bloq_2 = Conv2D(size, 1, padding="SAME")
        self.activ_2 = Activation('relu')

        self.bloq_3 = Conv2D(size, 1, padding="SAME")

        self.activ_final = Activation('relu')

        self.suma = Add()

    def call(self, x):
        _x0 = self.bloq_0(x)

        _x1 = self.bloq_1(_x0)
        _x1 = self.activ_1(_x1)

        _x1 = self.bloq_2(_x1)
        _x1 = self.activ_2(_x1)

        _x1 = self.bloq_3(_x1)

        _xf = self.suma([_x0, _x1])

        return self.activ_final(_xf)


class conv(tf.keras.layers.Layer):
    """
    Capa simple de convolución
    """
    def __init__(self, size=None):
        super(conv, self).__init__()
        self.conv = Conv2D(size, 1, padding="SAME", activation='relu')

    def call(self, x):
        return self.conv(x)

#Definición de todas las arquitecturas disponibles
ARCHS = {
    'bloque_res_simple': bloque_res_simple,
    'simple': conv
}