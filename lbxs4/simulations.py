from lbxs4.foreground import Foregrounds
from lbxs4.noise import NoiseModel
from lbxs4.cmb import CMBLensed
import lbxs4.utils as utils
import numpy as np
import healpy as hp
from tqdm import tqdm
from fgbuster import harmonic_ilc_alm,CMB
import os
import pickle as pl


class INST:
    def __init__(self,beam,frequency):
        self.Beam = beam
        self.fwhm = beam
        self.frequency = frequency


class LBSky:
    
    def __init__(self,libdir,nside=512,beam=30):
        self.libdir = os.path.join(libdir,'CompSep')
        self.hilc_alms_dir = os.path.join(self.libdir,'HILC_alms')
        self.hilc_weights_dir = os.path.join(self.libdir,'HILC_weights')
        self.hilc_noise_dir = os.path.join(self.libdir,'HILC_noise')
        os.makedirs(self.hilc_alms_dir,exist_ok=True)
        os.makedirs(self.hilc_weights_dir,exist_ok=True)
        os.makedirs(self.hilc_noise_dir,exist_ok=True)

        self.nside = nside
        self.lmax = 3*nside-1
        self.fg = Foregrounds(libdir,nside)
        self.noise = NoiseModel(nside)
        self.cmb = CMBLensed(nside)
        self.lb_inst = self.noise.lb_inst

        self.components = [CMB()]
        self.instrument = INST(None,self.lb_inst.center_frequency)
        self.bins = np.arange(1000) * 50


        ### hard coded for now
        self.nilc_dir = '/global/cfs/cdirs/cmbs4xlb/v1/component_separated/cs_products_LB/medium/nilc_standB2_b0b5'
        self.nilc_mask = hp.read_map('/global/cfs/cdirs/cmbs4xlb/v1/component_separated/cs_products_LB/masks/mask_PlaGAL_fsky80.fits')
        self.nilc_fsky = np.average(self.nilc_mask)
        self.beam = self.beam = hp.gauss_beam(np.radians(beam/60),lmax = self.lmax)
    
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
        for tag in tqdm(self.lb_inst.tag,desc=f'Frequncy alms of index {idx:04d}',unit='freq',leave=True):
            TEB.append(self.TEB(tag,idx,convolve=convolve))
        return np.array(TEB)
    
    def TEB_freq_deconv(self,idx):
        TEB = self.TEB_freq(idx)
        lmax = 3*self.nside-1
        for i in tqdm(range(len(TEB)),desc=f'Deconvolving frequency alms of index {idx:04d}',unit='freq',leave=True):
            freq = self.lb_inst.tag[i]
            fwhm = np.radians(self.lb_inst.get_fwhm(freq)/60)
            bl = hp.gauss_beam(fwhm=fwhm,lmax=lmax,pol=True).T
            hp.almxfl(TEB[i][0],1/bl[0],inplace=True)
            hp.almxfl(TEB[i][1],1/bl[1],inplace=True)
            hp.almxfl(TEB[i][2],1/bl[2],inplace=True)
        return TEB


    
    def HILC(self,idx):
        alm_fname = os.path.join(self.hilc_alms_dir,f'HILC_alms_{idx:04d}.pkl')
        weights_fname = os.path.join(self.hilc_weights_dir,f'HILC_weights_{idx:04d}.pkl')
        if not os.path.isfile(alm_fname):
            alms =  self.TEB_freq_deconv(idx)
            result = harmonic_ilc_alm(self.components,self.instrument,alms,self.bins)
            del alms
            pl.dump(result.s[0],open(alm_fname,'wb'))
            pl.dump(result.W,open(weights_fname,'wb'))
            cleaned = result.s[0].copy()
            del result
            return cleaned
        else:
            return pl.load(open(alm_fname,'rb'))
    
    def apply_harmonic_W(self,W, alms): 
        lmax = hp.Alm.getlmax(alms.shape[-1])
        res = np.full((W.shape[-2],) + alms.shape[1:], np.nan, dtype=alms.dtype)
        start = 0
        for i in range(0, lmax+1):
            n_m = lmax + 1 - i
            res[..., start:start+n_m] = np.einsum('...lcf,f...l->c...l',
                                                W[..., i:, :, :],
                                                alms[..., start:start+n_m])
            start += n_m
        return res
    
    def HILC_noise(self,idx):
        noise_fname = os.path.join(self.hilc_noise_dir,f'HILC_noise_{idx:04d}.pkl')
        weights_fname = os.path.join(self.hilc_weights_dir,f'HILC_weights_{idx:04d}.pkl')
        if not os.path.isfile(noise_fname):
            nalms = []
            for i in tqdm(range(len(self.lb_inst.tag)),desc=f'Noise alms of index {idx:04d}',unit='freq',leave=True):
                freq = self.lb_inst.tag[i]
                fwhm = np.radians(self.lb_inst.get_fwhm(freq)/60)
                noise_map = self.noise.noise_freq(freq,idx)
                noise_alms = hp.map2alm(noise_map)
                bl = hp.gauss_beam(fwhm=fwhm,lmax=3*self.nside-1,pol=True).T
                hp.almxfl(noise_alms[0],1/bl[0],inplace=True)
                hp.almxfl(noise_alms[1],1/bl[1],inplace=True)
                hp.almxfl(noise_alms[2],1/bl[2],inplace=True)
                nalms.append(noise_alms)
            nalms = np.array(nalms)
            W = pl.load(open(weights_fname,'rb'))
            ncl = self.apply_harmonic_W(W,nalms)
            del nalms
            pl.dump(ncl,open(noise_fname,'wb'))
            return ncl
        else:
            return pl.load(open(noise_fname,'rb'))
    
    def HILC_ncl(self,idx):
        nalm = self.HILC_noise(idx)[0]
        return hp.alm2cl(nalm[0]),hp.alm2cl(nalm[1]),hp.alm2cl(nalm[2])

    def NILC_Elm(self,idx,mask=False):
        fname = os.path.join(self.nilc_dir,f'E_{idx:04d}_reso30acm.fits')
        if not os.path.isfile(fname):
            raise ValueError(f'NILC alms for index {idx:04d} not found')
        else:
            if mask:
                emap = hp.read_map(fname) * self.nilc_mask 
            else:
                emap = hp.read_map(fname)
            return hp.map2alm(emap)

    def NILC_ncl(self,idx):
        return hp.read_cl(os.path.join(self.nilc_dir, f'cl_E_nres_medium_nilc_standB2_b0b5_{idx:04d}_reso30acm.fits'))


class S4Sky:

    def __init__(self,nside=1024,beam=2.1):
        self.nside = nside
        self.lmax = 3*nside-1
        self.path = '/global/cfs/cdirs/cmbs4xlb/v1/component_separated/chwide/nilc_Emaps/fits'
        self.mask = hp.ud_grade(hp.read_map('/global/cfs/cdirs/cmbs4xlb/v1/component_separated/chwide/nilc_Emaps/masks/chwide_clip0p3relhits_NSIDE2048.fits'),nside)
        self.nilc_fsky = np.average(self.mask)
        mask80 = hp.ud_grade(hp.read_map('/global/cfs/cdirs/cmbs4xlb/v1/component_separated/cs_products_LB/masks/mask_PlaGAL_fsky80.fits'),nside)
        self.nilc_mask = utils.change_coord(self.mask,['C','G'])*mask80
        del mask80
        self.nilc_fsky = np.average(self.nilc_mask)
        self.beam = hp.gauss_beam(np.radians(beam/60),lmax=self.lmax)

    def NILC_Elm(self,idx):
        fname = os.path.join(self.path,f'NILC_CMB-S4_CHWIDE-Emap_NSIDE2048_fwhm2.1_CHLAT-only_medium_cos-NSIDE2048-lmax4096_mc{idx:03d}.fits')
        if not os.path.isfile(fname):
            raise ValueError(f'NILC alms for index {idx:04d} not found')
        else:
            emap = hp.read_map(fname)
            return hp.map2alm(emap,lmax=self.lmax)
        
    def NILC_ncl(self,idx):
        return hp.anafast(hp.read_map(os.path.join(self.path, f'NILC_CMB-S4_CHWIDE-Enoise_NSIDE2048_fwhm2.1_CHLAT-only_medium_cos-NSIDE2048-lmax4096_mc{idx:03d}.fits')))[:self.lmax+1]