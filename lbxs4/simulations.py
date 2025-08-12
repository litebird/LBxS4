from lbxs4.foreground import Foregrounds
from lbxs4.noise import NoiseModel
from lbxs4.cmb import CMBLensed
import lbxs4.utils as utils
import numpy as np
import healpy as hp
from tqdm import tqdm
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
        self.pathb = '/global/cfs/cdirs/cmbs4xlb/v1/component_separated/cs_products_LB/medium/mcnilc'
        mask = hp.read_map('/global/cfs/cdirs/cmbs4xlb/v1/component_separated/cs_products_LB/masks/mask_PlaGAL_fsky80.fits')
        self.mask = utils.change_coord(hp.ud_grade(mask,self.nside),['G','C'])
        self.fsky = np.average(self.mask)
        self.beam = hp.gauss_beam(np.radians(beam/60),lmax = self.lmax)
        self.beamb = hp.gauss_beam(np.radians(70.5/60),lmax = self.lmax)

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
    
    def NILC_Blm(self,idx):
        fname = os.path.join(self.pathb,f"B_{idx:04}_reso70.5acm_ns64.fits")
        if not os.path.isfile(fname):
            raise ValueError(f'NILC alms for index {idx:04d} not found')
        else:
            local_nside = 64
            lmax = 3* local_nside - 1
            bmap = hp.read_map(fname)
            blm = hp.map2alm(bmap,lmax=lmax)
            #rot = hp.Rotator(coord=['G', 'C'])
            #rot.rotate_alm(blm,inplace=True)
            bmap = hp.alm2map(blm,local_nside)*hp.ud_grade(self.mask,local_nside)
            return hp.map2alm(bmap)

    def NILC_ncl(self,idx):
        return hp.read_cl(os.path.join(self.path, f'cl_E_rotatedCMB_nres_medium_nilc_standB2_b0b5_{idx:04d}_reso30acm.fits'))

    def NILC_nclb(self,idx):
        return hp.read_cl(os.path.join(self.pathb, f'cl_B_nres_medium_mcnilc_{idx:04}_reso70.5acm_lmax150.fits'))


class S4Sky:

    def __init__(self,nside=1024,beam=2.1):
        self.nside = nside
        self.lmax = 3*nside-1
        self.path = '/global/cfs/cdirs/cmbs4xlb/v1/component_separated/chwide/nilc_EBmaps/'
        self.phipath = '/global/cfs/cdirs/cmbs4xlb/v1/lensingrec/chwide_qe_v1.1'
        mask =hp.read_map('/global/cfs/cdirs/cmbs4xlb/v1/component_separated/chwide/masks/dust_mask_10pc-9dsmooth_3dC2_fgres_nside2048.fits')
        mask80 = hp.read_map('/global/cfs/cdirs/cmbs4xlb/v1/component_separated/cs_products_LB/masks/mask_PlaGAL_fsky80.fits')
        mask = hp.ud_grade(mask,nside)
        mask80 = utils.change_coord(hp.ud_grade(mask80,nside),['G','C'])
        self.mask = mask80*mask
        del (mask,mask80)
        self.fsky = np.average(self.mask)
        self.beam = hp.gauss_beam(np.radians(beam/60),lmax=self.lmax)

    def NILC_Elm(self,idx):
        fname = os.path.join(self.path,f'NILC_CMB-S4_CHWIDE-EBmap_NSIDE2048_fwhm2.1_CHLAT-only_medium_NSIDE2048-lmax4096_mc{idx:03d}.fits')
        if not os.path.isfile(fname):
            raise ValueError(f'NILC alms for index {idx:04d} not found')
        else:
            emap = hp.read_map(fname)*self.mask
            return hp.map2alm(emap,lmax=self.lmax)

    def NILC_ncl(self,idx):
        return hp.anafast(hp.read_map(os.path.join(self.path, f'NILC_CMB-S4_CHWIDE-EBresidual-noise_NSIDE2048_fwhm2.1_CHLAT-only_medium_NSIDE2048-lmax4096_mc{idx:03d}.fits')))[:self.lmax+1]
    
    def Philm(self,idx):
        return hp.read_alm(os.path.join(self.phipath,f'plm_reff_p_p_{idx:04}.fits'))
    
    def klm(self,idx):
        phi = self.Philm(idx)
        lmax = hp.Alm.getlmax(len(phi))
        l = np.arange(lmax+1,dtype=int)
        fl = l*(l+1)/2 
        return hp.almxfl(phi,fl)

    def N0(self,idx):
        n0L = np.loadtxt(os.path.join(self.phipath,f'Nlzero_semianalytic_reff_{idx:04}.txt'))
        L, n0 = n0L[:,0],n0L[:,1]
        fl = L*(L+1)/2
        return n0 * (fl**2)
    
    def N0_mean(self,n=20,lmax=None):
        N0 = []
        for i in range(n):
            N0.append(self.N0(i))
        N0 = np.array(N0).mean(axis=0)
        if lmax is not None:
            if len(N0) < lmax+1:
                N0 = np.concatenate((N0,np.zeros(lmax+1-len(N0))))
            else:
                N0 = N0[:lmax+1]
        return N0





    