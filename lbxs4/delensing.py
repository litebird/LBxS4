import curvedsky as cs
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os
import pickle as pl


class Delenser:

    def __init__(self,libdir,filt_lib,mt_lib,lmax=1024,elmin=150,klmin=2):
        self.libdir = libdir
        self.filt_lib = filt_lib
        self.mt_lib = mt_lib
        self.lmax = lmax
        self.elmin = elmin
        self.klmin = klmin

    
    def wiener_E(self,idx):
        return self.filt_lib.wiener_E(idx)

    
    def wiener_k(self,idx):
        return self.mt_lib.coadd(idx)
    
    def lensing_B(self,idx):
        wElm = self.wiener_E(idx)[:self.lmax +1,:self.lmax +1]
        klm = self.wiener_k(idx)[:self.lmax +1,:self.lmax +1]
        if klm.shape[0] < 1025:
            Klm = np.zeros((1025,1025))
            Klm[:klm.shape[0],:klm.shape[1]] = klm
            klm = Klm

        return cs.delens.lensingb(self.lmax,self.elmin,self.lmax,self.klmin,self.lmax, wElm, klm, gtype='k')
    
