import os
import copy
import numpy as np
import scipy.io as sio
from spectral.io import envi
from band_mapper import BandMapper
from sklearn.decomposition import PCA, FastICA, NMF, KernelPCA, LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler, StandardScaler




def load_hyper(dsetname, ncomp, redm, spatialsize=5, remove_bg=True, noise=0):
    data, labels, num_class = loadData(dsetname, num_components=ncomp, reduction_method=redm, noise=noise)
    shapeor = data.shape[:2]
    pixels, labels = createImageCubes(data, labels, windowSize=spatialsize, removeZeroLabels = remove_bg)
    pixels = pixels.reshape((pixels.shape[0], pixels.shape[1], pixels.shape[2], pixels.shape[3], 1))
    return pixels, labels, num_class, shapeor


def loadData(name, num_components=25, reduction_method=None, noise=0):
    data_path = os.path.join(os.getcwd(),'data')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SV':
        data = sio.loadmat(os.path.join(data_path, 'salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'paviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
    elif name == 'HICO1':
        data = envi.open('./data/HICO_1.hdr', './data/HICO_1.bin').asarray()
        data = data[100:-100,100:-100]
        labels = None
    elif name == 'AVIRIS1':
        data = envi.open('./data/AVIRIS_1.hdr', './data/AVIRIS_1.bin').asarray()
        data = data[200:-200,200:-200]
        labels = None
    elif name == 'UH13':
        data = sio.loadmat(os.path.join(data_path, 'houston.mat'))['houston']
        labels = sio.loadmat(os.path.join(data_path, 'houston_gt.mat'))['houston_gt_tr']
        labels += sio.loadmat(os.path.join(data_path, 'houston_gt.mat'))['houston_gt_te']
    elif name == 'UH18':
        import cv2
        data = sio.loadmat(os.path.join(data_path, 'houston2018.mat'))['houston2018']
        labels = sio.loadmat(os.path.join(data_path, 'houston2018_gt.mat'))['houston2018_gt']
        labels = cv2.resize(labels, dsize=(labels.shape[1]//2, labels.shape[0]//2), interpolation=cv2.INTER_NEAREST).astype("uint8")
    else:
        print("NO DATASET")
        exit()
    shapeor = data.shape
    #
    #
    # ruido = 0.05
    # data = data.reshape(-1, data.shape[-1])
    # mm   = MinMaxScaler().fit(data)
    # data = mm.transform(data)
    # data = data.reshape(shapeor)
    # noise = np.random.normal(0, ruido, data.shape)
    # imnoise = data + noise
    # data = np.clip(imnoise,0,1)
    # data = data.reshape(-1, data.shape[-1])
    # data = mm.inverse_transform(data)
    # data = data.reshape(shapeor)
    #
    if noise != 0:
        ruido = noise
        print("RUIDO", ruido)
        data = data.reshape(-1, data.shape[-1])
        mm   = MinMaxScaler().fit(data)
        data = mm.transform(data)
        data = data.reshape(shapeor)
        np.random.seed(123) # you can remove this line
        noise = np.random.normal(0, ruido, data.shape)
        imnoise = copy.deepcopy(data)
        imnoise[:,:,25] += noise[:,:,25]
        imnoise[:,:,55] += noise[:,:,55]
        data = np.clip(imnoise,0,1)
        data = data.reshape(-1, data.shape[-1])
        data = mm.inverse_transform(data)
        data = data.reshape(shapeor)
    data = data.reshape(-1, data.shape[-1])
    if num_components != None:
        print("reduction_method", reduction_method)
        if reduction_method == 'pca':
            data = PCA(n_components=num_components).fit_transform(data)
        elif reduction_method == 'ica':
            data = FastICA(n_components=num_components).fit_transform(data)
        elif reduction_method == 'smsi':
            mapper = BandMapper()
            data = mapper.map(data=data, resulting_bands_count=num_components)
        else: num_components = data.shape[-1]
        shapeor = np.array(shapeor)
        shapeor[-1] = num_components
    data = StandardScaler().fit_transform(data)
    data = data.reshape(shapeor).astype("float32")
    num_class = len(np.unique(labels)) - 1
    return data, labels, num_class


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    if y is not None:
        patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            if y is not None:
                patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels and y != None:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    if y is not None:
        return patchesData, patchesLabels.astype("int")
    else:
        return patchesData, None
