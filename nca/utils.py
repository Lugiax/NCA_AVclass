from skimage.morphology import disk, binary_dilation, square
from skimage.filters import gaussian, threshold_multiotsu
from skimage.filters.rank import median
from skimage.measure import regionprops, label
from skimage.transform import resize, rescale
from skimage.io import imread
import matplotlib.pylab as pl
import tensorflow as tf
import numpy as np
import PIL.Image
import yaml
import re
import os


def segment(img, sigma, umbral=None):

    #plt.imshow(img)
    #plt.title('Original')
    #plt.show()
    _x, _y = np.meshgrid(np.linspace(-1,1, img.shape[1]), np.linspace(-1,1, img.shape[0]))
    dst = np.sqrt(_x*_x + _y*_y)
    gauss_curve = np.exp( -( dst**2 / ( 2 * 1.4**2) ))
    #plt.imshow(gauss_curve)
    #plt.show()

    chns = []
    for k in range(3):
        chn_img = img[..., k]*gauss_curve
        norm_img = ( chn_img - np.min(chn_img)) / (np.max(chn_img)-np.min(chn_img) )
        norm_img = gaussian(norm_img, sigma)
        umbral_chn = threshold_multiotsu(norm_img[norm_img>0.55], 5)[-1] if umbral is None else umbral
        bin_img = norm_img >= umbral_chn
        chns.append(bin_img)

    segmented = np.logical_or(chns[1], chns[2]) # Antes era: np.logical_and(chns[0], np.logical_or(chns[1], chns[2]))
                                                # El canal rojo solo mete ruido
    return segmented

def get_optical_disk_center(im, scale=None, sigma=13, ra=75): 
    if scale is None: #Se escala la Imagen
        scale = 1000/min(im.shape[:2])

    if scale<1:
        shape_scaled = [int(s*scale) for s in im.shape[:2]]
        im = resize(im, shape_scaled, anti_aliasing=True)
    else:
        im = rescale(im, scale, order=2, multichannel=True)

    im_seg = segment(im, sigma=sigma)
    separadas = label(im_seg, connectivity=2)
    regions = regionprops(separadas)
    candidatas = []
    todas = []
    for r in regions:
        redondez = 4*np.pi*r.area/(r.perimeter**2)
        if redondez >= 0.5:
            razon = np.abs(np.sqrt(r.area/np.pi) - ra) ##Falta revisar bien cómo usar la Ra
            candidatas.append((r.centroid, razon))
        todas.append((r.centroid, redondez))
    
    reverse = False
    if len(candidatas) == 0:
        candidatas = todas
        reverse = True
 
    mejor = sorted(candidatas, key = lambda x: x[1], reverse=reverse)[0]#Ordenar según la razón o redondez
    return [int(v/scale) for v in mejor[0]]# cy, cx

def normalize_girard(batch, sigma=2, selem_size=5):
    def join(r,g,b):
        return np.concatenate((r[..., np.newaxis], g[..., np.newaxis], b[..., np.newaxis]), axis=2)
    new_batch = []
    for j in range(batch.shape[0]):
        rgb = batch[j, ..., :3]
        R = np.uint(rgb[..., 0]*255)
        G = np.uint(rgb[..., 1]*255)
        B = np.uint(rgb[..., 2]*255)
        difs = []
        medians_std = []
        for c in (R,G,B):
            med = median(c, square(selem_size))
            dif = c-med
            difs.append(dif)
            medians_std.append(np.std(dif))

        new_batch.append(join(*[sigma*dif/std+0.5 for dif, std in zip(difs, medians_std)]))

    return np.stack(new_batch) #tf.cast(tf.stack(new_batch), tf.float32)

def cargar_dataset(fname, agregar_girard=False, selem_size=5):
    with open(fname, 'rb') as f:
        x_train = np.load(f)
        y_train_pic =np.load(f)
        x_test = np.load(f)
        y_test_pic = np.load(f)

    original_channels=x_train.shape[-1]
    new_idx_train = np.random.randint(0, x_train.shape[0]-1, size=x_train.shape[0])
    new_idx_test = np.random.randint(0, x_test.shape[0]-1, size=x_test.shape[0])

    if agregar_girard:
        x_train = tf.cast(
                tf.concat((x_train[new_idx_train, ..., :3],
                            normalize_girard(x_train[new_idx_train, ..., :3], selem_size=selem_size),
                            np.ceil(x_train[new_idx_train ,..., -1:])),
                            axis=-1), tf.float32)
        x_test = tf.cast(
                tf.concat((x_test[new_idx_test, ..., :3],
                            normalize_girard(x_test[new_idx_test, ..., :3], selem_size=selem_size),
                            np.ceil(x_test[new_idx_test ,..., -1:])),
                            axis=-1), tf.float32)
    else:
        x_train = tf.cast(
                tf.concat((x_train[new_idx_train, ..., :3],
                            np.ceil(x_train[new_idx_train ,..., -1:])),
                            axis=-1), tf.float32)
        x_test = tf.cast(
                tf.concat((x_test[new_idx_test, ..., :3],
                            np.ceil(x_test[new_idx_test ,..., -1:])),
                            axis=-1), tf.float32)

    y_train_pic = tf.cast(y_train_pic[new_idx_train], tf.float32)
    y_test_pic = tf.cast(y_test_pic[new_idx_test], tf.float32)

    #No encontré otra forma mejor para quitar la "máscara de las imágenes que no la tienen"
    return x_train[..., :original_channels], y_train_pic, x_test[..., :original_channels], y_test_pic


def cargar_config(fname, automata_shape=None, save_dir='.'):
    with open(fname, 'r') as f:
        config = yaml.load(f)
    
    config['AUTOMATA_SHAPE'] = [int(re.findall(r'(\d+)aut', config['DB_FNAME'])[0])]*2\
                                    if automata_shape is None else automata_shape
    config['SAVE_DIR'] = save_dir
    return config




def abrir_img(path, shape=None):
    if path.split('.')[-1] in ['tif', 'tiff']:
        img = np.array(PIL.Image.open(path))
    else:
        img = imread(path)
    if shape is not None:
        img = resize(img, shape)
    return img


def calcular_metricas(y_true, y_pred):
    if tf.is_tensor(y_true):
        t, p = y_true.numpy(), y_pred.numpy()
    else:
        t, p = y_true[::], y_pred[::]
    
    assert t.shape == p.shape and len(t.shape) in [3,4], 'Revisar las dimensiones de los tensores'

    if len(t.shape)==3:
        #Agregar una dimensión a los arreglos tridimensionales
        t = t[None, ...]
        p = p[None, ...]
    
    ta, tv = t[..., 0], t[..., 1]
    pa, pv = p[..., 0], p[..., 1]
    n = t.shape[0]

    conf_mats = [[tf.math.confusion_matrix(ta[i].ravel(),
                                          pa[i].ravel()),
                  tf.math.confusion_matrix(tv[i].ravel(),
                                          pv[i].ravel())]
                    for i in range(n)]

    metricas = {
            'acc':     [[],[]],
            'prec':    [[],[]],
            'sensit':  [[],[]],
            'specif':  [[],[]],
            'dice':    [[],[]],
            'jaccard': [[],[]]
        }
    
    for conf_mat in conf_mats:
        for k in range(2):
            TP, FP, FN, TN = conf_mat[k].numpy().ravel()
            metricas['acc'][k].append( (TP + TN) / (TP + TN + FP + FN) )
            metricas['prec'][k].append( TP / (TP + FP))
            metricas['sensit'][k].append( TP / (TP + FN))
            metricas['specif'][k].append( TN / (FP + TN))
            metricas['dice'][k].append( 2*TP / (2*TP + FP + FN))
            metricas['jaccard'][k].append( TP / (TP + FP + FN))
            
    return(metricas)


if __name__ == '__main__':
    y_pred = np.random.randint(0,2, size = (10,1000,1000,2))
    y_true = np.random.randint(0,2, size = (10,1000,1000,2))
    
    print(calcular_metricas(y_true, y_pred))