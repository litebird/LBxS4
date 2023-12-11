import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pl
import toml
import healpy as hp
import curvedsky as cs
from tqdm import tqdm
import cmb
from lbxs4.config import MASKDIR, DATDIR
from lbxs4.utils import cli

#from simulation import 

class Filtering:
   
    def __init__(self,lib_dir,sim_lib,maskpath,beam=15,verbose=False):
        self.libdir = os.path.join(lib_dir,'Filtering')
        os.makedirs(self.libdir,exist_ok=True)
        self.sim_lib = sim_lib
        self.mask = hp.ud_grade(hp.read_map(maskpath),self.sim_lib.nside)
        self.fsky = np.average(self.mask)

        
        self.Tcmb = 2.726e6
        
        self.nside = self.sim_lib.nside
        self.lmax =  3*self.nside - 1
        self.cl_len = cmb.read_camb_cls(os.path.join(DATDIR,'FFP10_wdipole_lensedCls.dat'),ftype='lens',output='array')
        self.nsim = None

        self.beam = hp.gauss_beam(np.radians(beam/60),lmax = self.lmax)
        self.Bl = np.reshape(self.beam,(1,self.lmax+1))
        #needed for filtering

        self.ninv = np.reshape(np.array((self.mask,self.mask)),(2,1,hp.nside2npix(self.nside)))

        

   
    def convolved_TEB(self,idx):
        """
        convolve the component separated map with the beam

        Parameters
        ----------
        idx : int : index of the simulation
        """
        T,E,B = self.sim_lib.HILC(idx)
        hp.almxfl(T,self.beam,inplace=True)
        hp.almxfl(E,self.beam,inplace=True)
        hp.almxfl(B,self.beam,inplace=True)
        return T,E,B
        

    def TQU_to_filter(self,idx):
        """
        Change the convolved ALMs to MAPS

        Parameters
        ----------
        idx : int : index of the simulation
        """
        T,E,B = self.convolved_TEB(idx)

        return hp.alm2map([T,E,B],nside=self.nside)

    
    def NL(self,idx):
        """
        array manipulation of noise spectra obtained by ILC weight
        for the filtering process
        """
        nt,ne,nb = self.sim_lib.HILC_ncl(idx)
        ne = ne/ self.Tcmb**2
        nb = nb/ self.Tcmb**2
        return np.reshape(np.array((cli(ne[:self.lmax+1]*self.beam**2),
                          cli(nb[:self.lmax+1]*self.beam**2))),(2,1,self.lmax+1))

    def cinv_EB(self,idx,test=False):
        """
        C inv Filter for the component separated maps

        Parameters
        ----------
        idx : int : index of the simulation
        test : bool : if True, run the filter for 10 iterations
        """
        fsky = f"{self.fsky:.2f}".replace('.','p')
        fname = os.path.join(self.libdir,f"cinv_EB_{idx:04d}_fsky_{fsky}.pkl")
        if not os.path.isfile(fname):
            TQU = self.TQU_to_filter(idx)
            QU = np.reshape(np.array((TQU[1]*self.mask,TQU[2]*self.mask)),
                            (2,1,hp.nside2npix(self.nside)))/self.Tcmb
            
            iterations = [1000]
            stat_file = 'stat.txt' 
            if test:
                print(f"Cinv filtering is testing {idx}")
                iterations = [10]
                stat_file = os.path.join(self.libdir,'test_stat.txt')

            E,B = cs.cninv.cnfilter_freq(2,1,self.nside,self.lmax,self.cl_len[1:3,:self.lmax+1],
                                        self.Bl, self.ninv,QU,chn=1,itns=iterations,filter="",
                                        eps=[1e-5],ro=10,inl=self.NL(idx),stat=stat_file)
            if not test:
                pl.dump((E,B),open(fname,'wb'))
        else:
            E,B = pl.load(open(fname,'rb'))
        
        return E,B
    
    def wiener_E(self,idx):
        E,_ = self.cinv_EB(idx)
        clee = self.cl_len[1,:self.lmax+1]
        return cs.utils.almxfl(self.lmax,self.lmax,E,clee)
    
    def wiener_B(self,idx):
        _,B = self.cinv_EB(idx)
        clbb = self.cl_len[2,:self.lmax+1]
        return cs.utils.almxfl(self.lmax,self.lmax,B,clbb)

    def plot_cinv(self,idx):
        """
        plot the cinv filtered Cls for a given idx

        Parameters
        ----------
        idx : int : index of the simulation
        """
        E,_ = self.cinv_EB(idx)
        _,ne,_ = self.sim_lib.HILC_ncl(idx)
        cle = cs.utils.alm2cl(self.lmax,E)
        plt.figure(figsize=(8,8))
        plt.loglog(cle,label='E')
        plt.loglog(1/self.cl_len[1,:])
    
    def plot_wE(self,idx):
        E = self.wiener_E(idx)
        we = cs.utils.alm2cl(self.lmax,E)
        plt.figure(figsize=(5,5))
        plt.loglog(we)
        plt.loglog(self.cl_len[1,:])
        plt.xlim(2,1000)
        plt.ylim(1e-18,1e-14)
        plt.legend(['Wiener E','Input E'], fontsize=15)
        plt.xlabel('$\ell$',fontsize=20)
        plt.ylabel('$C_\ell$',fontsize=20)

