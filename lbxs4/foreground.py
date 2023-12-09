import healpy as hp
import numpy as np
from lbxs4.config import FGDIR
from lbxs4.instrument import LiteBIRD
import os


class Foregrounds:

    def __init__(self,nside,complexity='low'):
        self.nside = nside
        self.lb_inst = LiteBIRD()

        if complexity not in ['low','medium','high']:
            raise ValueError('wrong complexity, try low,medium,high!')
        self.models = self.__selectFG__(complexity)



    def __selectFG__(self,which):
        return {
                'low': ['d9','s4','f1','a1','co1'],
                'medium' : ['d10','s5','f1','a1','co3'],
                'high': ['d12','s7','f1','a2','co3']
                }[which]
    
    def __select_dir__(self,which):
        return {
                'd9': 'dust_d9',
                'd10': 'dust_d10',
                'd12': 'dust_d12',
                's4': 'synchrotron_s4',
                's5': 'synchrotron_s5',
                's7': 'synchrotron_s7',
                'f1': 'freefree_f1',
                'a1': 'ame_a1',
                'a2': 'ame_a2',
                'co1': 'co_co1',
                'co3': 'co_co3'
                }[which]

    def TQU(self,freq):
        #TODO: save the higher nside maps to disk
        self.lb_inst.check_band(freq)
        fgmap = np.zeros((3,hp.nside2npix(self.nside)))
        for model in self.models:
            fgdir = self.__select_dir__(model)
            directory = os.path.join(FGDIR,fgdir)
            in_nside = self.lb_inst.get_nside(freq)
            fname = f"litebird_{fgdir}_uKCMB_{freq}_nside{int(in_nside)}.fits"
            print(f"Reading {fname}...")
            if in_nside != self.nside:
                tmp_map = hp.read_map(os.path.join(directory,fname),(0,1,2)) # type: ignore
                tmp_alm = hp.map2alm(tmp_map)
                fgmap += hp.alm2map(tmp_alm,self.nside)
                del (tmp_map,tmp_alm)
            else:
                fgmap += hp.read_map(os.path.join(directory,fname),(0,1,2)) # type: ignore
        return fgmap



class CompSep:

    def __init__(self,method='HILC'):
        pass
