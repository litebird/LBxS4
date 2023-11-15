from lbxs4.foreground import Foregrounds,CompSep
from lbxs4.noise import NoiseModel
from lbxs4.cmb import CMBLensed
import numpy as np
import healpy as hp



class LBSky:
    
    def __init__(self,nside):
        self.nside = nside
        self.fg = Foregrounds(nside)
        self.noise = NoiseModel(nside)
        self.cmb = CMBLensed(nside)
        self.lb_inst = self.noise.lb_inst
    
    def TQU(self,freq,idx,convolve=True):
        fwhm = np.radians(self.lb_inst.get_fwhm(freq)/60)
        if convolve:
            return hp.smoothing(self.fg.TQU(freq) + self.cmb.TQU(idx),fwhm=fwhm,pol=True) + self.noise.noise_freq(freq,idx)
        else:
            return self.fg.TQU(freq) + self.cmb.TQU(idx) + self.noise.noise_freq(freq,idx)
    
    def TEB(self,freq,idx,convolve=True):
        return hp.map2alm(self.TQU(freq,idx,convolve=convolve))
    
    def TQU_freq(self,idx,convolve=True):
        TQU = []
        for tag in self.lb_inst.tag:
            TQU.append(self.TQU(tag,idx,convolve=convolve))
        return np.array(TQU)
    
    def TEB_freq(self,idx,convolve=True):
        TEB = []
        for tag in self.lb_inst.tag:
            TEB.append(self.TEB(tag,idx,convolve=convolve))
        return np.array(TEB)
    




    
    def CompSep_TQU(idx):
        pass