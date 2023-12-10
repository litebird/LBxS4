import numpy as np
import os
import mpi
import matplotlib.pyplot as plt
import pickle as pl
import toml
import healpy as hp
import curvedsky as cs
from tqdm import tqdm
import cmb
from lbxs4.config import MASKDIR, DATDIR

#from simulation import 

class Filtering:
   
    def __init__(self,lib_dir,sim_lib,maskpath,beam=None,verbose=False):
        self.libdir = os.path.join(lib_dir,'Filtering')
        self.sim_lib = sim_lib
        self.mask = hp.ud_grade(hp.read_map(maskpath),self.sim_lib.nside)
        self.fsky = np.average(self.mask)

        
        self.Tcmb = 2.726e6
        
        self.nside = self.sim_lib.nside
        self.lmax =  3*self.nside - 1
        self.cl_len = os.path.join(DATDIR,'FFP10_wdipole_lensedCls.dat')
        self.nsim = None

        #needed for filtering

        self.ninv = np.reshape(np.array((self.mask,self.mask)),(2,1,hp.nside2npix(self.nside)))

        

   
    def convolved_TEB(self,idx):
        """
        convolve the component separated map with the beam

        Parameters
        ----------
        idx : int : index of the simulation
        """
        #T,E,B = self.sim_lib.get_cleaned_cmb(idx)
        #hp.almxfl(T,self.beam,inplace=True)
        #hp.almxfl(E,self.beam,inplace=True)
        #hp.almxfl(B,self.beam,inplace=True)
        #return T,E,B
        T,E,B = self.sim_lib.HILC(idx)
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
        fname = os.path.join(self.lib_dir,f"cinv_EB_{idx:04d}_fsky_{fsky}.pkl")
        if not os.path.isfile(fname):
            TQU = self.TQU_to_filter(idx)
            QU = np.reshape(np.array((TQU[1]*self.mask,TQU[2]*self.mask)),
                            (2,1,hp.nside2npix(self.nside)))/self.Tcmb
            
            iterations = [1000]
            stat_file = '' 
            if test:
                self.vprint(f"Cinv filtering is testing {idx}")
                iterations = [10]
                stat_file = os.path.join(self.lib_dir,'test_stat.txt')

            E,B = cs.cninv.cnfilter_freq(2,1,self.nside,self.lmax,self.cl_len[1:3,:],
                                        self.Bl, self.ninv,QU,chn=1,itns=iterations,filter="",
                                        eps=[1e-5],ro=10,inl=self.NL,stat=stat_file)
            if not test:
                pl.dump((E,B),open(fname,'wb'))
        else:
            E,B = pl.load(open(fname,'rb'))
        
        return E,B

    def plot_cinv(self,idx):
        """
        plot the cinv filtered Cls for a given idx

        Parameters
        ----------
        idx : int : index of the simulation
        """
        _,B = self.cinv_EB(idx)
        _,_,nb = self.sim_lib.noise_spectra(self.sim_lib.nsim)
        clb = cs.utils.alm2cl(self.lmax,B)
        plt.figure(figsize=(8,8))
        plt.loglog(clb,label='B')
        plt.loglog(1/(self.cl_len[2,:]  + nb))

    def wiener_EB(self,idx):
        """
        Not implemented yet
        useful for delensing
        """
        E, B = self.cinv_EB(idx)
        pass

    def run_job_mpi(self):
        """
        MPI job for filtering
        """
        job = np.arange(mpi.size)
        for i in job[mpi.rank::mpi.size]:
            eb = self.cinv_EB(i)

    def run_job(self):
        """
        MPI job for filtering
        """
        jobs = np.arange(self.sim_lib.nsim)
        for i in tqdm(jobs, desc='Cinv filtering', unit='sim'):
            eb = self.cinv_EB(i)
            del eb
