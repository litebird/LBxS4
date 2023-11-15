import healpy as hp
from .config import NOISEDIR
import os
from lbxs4.instrument import LiteBIRD



class NoiseModel:
    
    
    def __init__(self,nside=512):
        self.nside = nside
        self.libdir = NOISEDIR
        self.lb_inst = LiteBIRD()
    
    def noise_freq(self,freq,idx):
        self.lb_inst.check_band(freq)
        directory = os.path.join(self.libdir,f'{freq}')
        fname = os.path.join(directory,f'{freq}_wn_map_0512_mc_{idx:04d}.fits')
        return hp.read_map(fname,(0,1,2))
