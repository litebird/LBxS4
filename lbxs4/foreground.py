import healpy as hp
import numpy as np
from lbxs4.config import FGDIR
from lbxs4.instrument import LiteBIRD
import os


class Foregrounds:

    def __init__(self,libdir,nside,complexity='low',version='v1'):
        self.libdir = os.path.join(libdir,'FG_downgraded')
        os.makedirs(self.libdir,exist_ok=True)
        self.nside = nside
        self.lb_inst = LiteBIRD()
        self.version = version
        self.complexity = complexity

        if complexity not in ['low','medium','high']:
            raise ValueError('wrong complexity, try low,medium,high!')
        self.models = self.__selectFG__(complexity)



    def __selectFG__(self,which):
        return {
                'low': ['d9','s4','f1','a1','co1'],
                'medium' : ['d10','s5','f1','a1','co3'],
                'high': ['d12','s7','f1','a2','co3']
                }[which]
    
    def __select_dir_v0__(self,which):
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
    
    def __select_dir_v1__(self,which):
        return {
                'low': 'combined_foregrounds_lowcomplexity',
                'medium': 'combined_foregrounds_mediumcomplexity',
                'high': 'combined_foregrounds_highcomplexity'
                }[which]
        
    
    def TQU(self,freq):
        if self.version == 'v0':
            return self.TQU_v0(freq)
        elif self.version == 'v1':
            return self.TQU_v1(freq)
        else:
            raise ValueError('wrong version, try v0,v1!')
    
    def TQU_v1(self,freq):
        self.lb_inst.check_band(freq)
        directory = os.path.join(FGDIR,self.__select_dir_v1__(self.complexity))
        in_nside = self.lb_inst.get_nside(freq)
        fname = f"litebird_combined_foregrounds_{self.complexity}complexity_uKCMB_{freq}_nside{int(in_nside)}.fits"
        if in_nside != self.nside:
            fname_ud = os.path.join(self.libdir,f"litebird_combined_foregrounds_{self.complexity}complexity_uKCMB_{freq}_nside{int(self.nside)}.fits")
            if not os.path.isfile(fname_ud):
                tmp_map = hp.read_map(os.path.join(directory,fname),(0,1,2))
                tmp_alm = hp.map2alm(tmp_map)
                tmp_map_d = hp.alm2map(tmp_alm,self.nside)
                hp.write_map(fname_ud,tmp_map_d)
                del (tmp_map,tmp_alm)
                return tmp_map_d
            else:
                return hp.read_map(fname_ud,(0,1,2))
        else:
            return hp.read_map(os.path.join(directory,fname),(0,1,2))



    def TQU_v0(self,freq):
        self.lb_inst.check_band(freq)
        fgmap = np.zeros((3,hp.nside2npix(self.nside)))
        for model in self.models:
            fgdir = self.__select_dir_v0__(model)
            directory = os.path.join(FGDIR,fgdir)
            in_nside = self.lb_inst.get_nside(freq)
            fname = f"litebird_{fgdir}_uKCMB_{freq}_nside{int(in_nside)}.fits"
            if in_nside != self.nside:
                fname_ud = os.path.join(self.libdir,f"litebird_{fgdir}_uKCMB_{freq}_nside{int(self.nside)}.fits")
                if not os.path.isfile(fname_ud):
                    tmp_map = hp.read_map(os.path.join(directory,fname),(0,1,2)) # type: ignore
                    tmp_alm = hp.map2alm(tmp_map)
                    tmp_map_d = hp.alm2map(tmp_alm,self.nside)
                    hp.write_map(fname_ud,tmp_map_d)
                    fgmap += tmp_map_d
                    del (tmp_map,tmp_alm,tmp_map_d)
                else:
                    fgmap += hp.read_map(fname_ud,(0,1,2))
            else:
                fgmap += hp.read_map(os.path.join(directory,fname),(0,1,2)) # type: ignore
        return fgmap



