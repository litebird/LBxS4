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
    def __init__(self,nside=512,beam=30):
        self.nside = nside
        self.lmax = 3*nside-1
        self.path = '/global/cfs/cdirs/cmbs4xlb/v1/component_separated/cs_products_LB/medium/nilc_standB2_b0b5_rotated'
        mask = hp.read_map('/global/cfs/cdirs/cmbs4xlb/v1/component_separated/cs_products_LB/masks/mask_PlaGAL_fsky80.fits')
        self.mask = utils.change_coord(hp.ud_grade(mask,self.nside),['G','C'])
        self.fsky = np.average(self.mask)
        self.beam = self.beam = hp.gauss_beam(np.radians(beam/60),lmax = self.lmax)


    def NILC_Elm(self,idx,):
        fname = os.path.join(self.path,f'E_rotatedCMB_{idx:04d}_reso30acm.fits')
        if not os.path.isfile(fname):
            raise ValueError(f'NILC alms for index {idx:04d} not found')
        else:
            emap = hp.read_map(fname)
            elm = hp.map2alm(emap,lmax=self.lmax)
            rot = hp.Rotator(coord=['G', 'C'])
            rot.rotate_alm(elm,inplace=True)
            emap = hp.alm2map(elm,self.nside)*self.mask
            return hp.map2alm(emap)

    def NILC_ncl(self,idx):
        return hp.read_cl(os.path.join(self.path, f'cl_E_rotatedCMB_nres_medium_nilc_standB2_b0b5_{idx:04d}_reso30acm.fits'))


class S4Sky:

    def __init__(self,nside=1024,beam=2.1):
        self.nside = nside
        self.lmax = 3*nside-1
        self.path = '/global/cfs/cdirs/cmbs4xlb/v1/component_separated/chwide/nilc_Emaps/fits'
        mask =hp.read_map('/global/cfs/cdirs/cmbs4xlb/v1/component_separated/chwide/masks_common/chwide_clip0p3relhits_NSIDE2048.fits')
        mask80 = hp.read_map('/global/cfs/cdirs/cmbs4xlb/v1/component_separated/cs_products_LB/masks/mask_PlaGAL_fsky80.fits')
        mask = hp.ud_grade(mask,nside)
        mask80 = utils.change_coord(hp.ud_grade(mask80,nside),['G','C'])
        self.mask = mask80*mask
        del (mask,mask80)
        self.fsky = np.average(self.mask)
        self.beam = hp.gauss_beam(np.radians(beam/60),lmax=self.lmax)

    def NILC_Elm(self,idx):
        fname = os.path.join(self.path,f'NILC_CMB-S4_CHWIDE-Emap_NSIDE2048_fwhm2.1_CHLAT-only_medium_cos-NSIDE2048-lmax4096_mc{idx:03d}.fits')
        if not os.path.isfile(fname):
            raise ValueError(f'NILC alms for index {idx:04d} not found')
        else:
            emap = hp.read_map(fname)*hp.ud_grade(self.mask,2048)
            return hp.map2alm(emap,lmax=self.lmax)

    def NILC_ncl(self,idx):
        return hp.anafast(hp.read_map(os.path.join(self.path, f'NILC_CMB-S4_CHWIDE-Enoise_NSIDE2048_fwhm2.1_CHLAT-only_medium_cos-NSIDE2048-lmax4096_mc{idx:03d}.fits')))[:self.lmax+1]