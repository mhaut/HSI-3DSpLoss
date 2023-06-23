# Novel Spectral Loss Function for Unsupervised Hyperspectral Image Segmentation
The Code for "Novel Spectral Loss Function for Unsupervised Hyperspectral Image Segmentation". [https://ieeexplore.ieee.org/document/10159412]
```
Á. Pérez-García, M. E. Paoletti, J. M. Haut and J. F. López.
Novel Spectral Loss Function for Unsupervised Hyperspectral Image Segmentation
IEEE Geoscience and Remote Sensing Letters
DOI: 10.1109/LGRS.2023.3288809
July 2023.
```

![SPLOSS](./images/SPLOSS.jpg)


## Run code

```
python DCEC.py --dataset IP

# with dimensionality reduction
python DCEC.py --dataset IP --redm pca --ncomp 25
python DCEC.py --dataset IP --redm ica --ncomp 25
python DCEC.py --dataset IP --redm smsi --ncomp 25

#with noise
python DCEC.py --dataset IP --noise 0.25

# with pretrain cae weights
python DCEC.py --dataset IP --cae_weights ./pretrain_cae_model.h5
```


Reference code 1: https://github.com/XifengGuo/DCEC
Reference code 2: https://gitlab.com/jnalepa/rnns_for_hsi/
