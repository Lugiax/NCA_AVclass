from tensorflow.keras.layers import Conv2D, Activation, Add
import tensorflow as tf

class bloque_res_simple(tf.keras.layers.Layer):
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
    def __init__(self, size=None):
        super(conv, self).__init__()
        self.conv = Conv2D(size, 1, padding="SAME", activation='relu')

    def call(self, x):
        return self.conv(x)

"""
MODELS = {
    'simple':   [
                Conv2D(80, 1, padding='SAME', activation='relu'),  
                ],
    'mini_res': [
                bloque_res_simple(80)
                ]
}

def get_hidden(name):
    assert name in MODELS, 'Modelo no existente'
    return MODELS.get(name)
"""
ARCHS = {
    'bloque_res_simple': bloque_res_simple,
    'simple': conv
}