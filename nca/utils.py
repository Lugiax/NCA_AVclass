from skimage.filters import gaussian, threshold_multiotsu
from skimage.measure import regionprops, label
from skimage.transform import resize, rescale
import tensorflow as tf
import numpy as np
import yaml



def cargar_config(fname, save_dir='.'):
    with open(fname, 'r') as f:
        config = yaml.safe_load(f)
    config['SAVE_DIR'] = save_dir
    return config


def segment(img, sigma, umbral=None):
    _x, _y = np.meshgrid(np.linspace(-1,1, img.shape[1]), np.linspace(-1,1, img.shape[0]))
    dst = np.sqrt(_x*_x + _y*_y)
    gauss_curve = np.exp( -( dst**2 / ( 2 * 1.4**2) ))

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