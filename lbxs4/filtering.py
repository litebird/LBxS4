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
from lbxs4.utils import cli,change_coord
from lbxs4.simulations import LBSky, S4Sky


#from simulation import 

class Filtering:
   
    def __init__(self,lib_dir,sim_lib,maskpath,beam=15,s4lib=None,coadd_E=False,verbose=False):
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
    
    # def S4_to_filter(self,idx):
    #     if self.s4lib is None:
    #         raise ValueError("S4 library is not defined")
    #     T,E,B = self.s4lib.comp_sep_alm(idx)
    #     Q,U = hp.alm2map_spin([E,B*0],nside=self.nside,spin=2)


    
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



class FiltCoadd:


    def __init__(self,libdir,lblib=None,s4lib=None,coadd=False):
        
        self.lblib = lblib
        self.s4lib = s4lib
        self.coadd = coadd
        self.cl_len = cmb.read_camb_cls(os.path.join(DATDIR,'FFP10_wdipole_lensedCls.dat'),ftype='lens',output='array')
        

        self.option = ''



        if lblib is not None:
            assert isinstance(lblib,LBSky), "lblib should be an instance of LBSky"
            print("INFO:LiteBIRD simulation library is loaded")
            self.option = 'LB'
        
        if s4lib is not None:
            assert isinstance(s4lib,S4Sky), "s4lib should be an instance of S4Sky"
            print("INFO:CMB-S4 simulation library is loaded")
            self.option = 'S4'

        if coadd:
            assert (lblib is not None) and (s4lib is not None), "Both libraries should be defined for coaddition"
            print("INFO:Coaddition is enabled")
            self.option = 'LBxS4'
        else:
            if (lblib is not None) and (s4lib is not None):
                raise ValueError("Without coaddition, Only one library should be defined")
        
        self.libdir = os.path.join(libdir,f'FiltCoadd_{self.option}')
        os.makedirs(self.libdir,exist_ok=True)

        self.Tcmb = 2.726e6

        self.nside = None
        self.lmax = None
        self.beam = None
        self.fsky = None
        self.ninv = None
        self.mask = None
        self.__find_internal_params__()

    def __find_internal_params__(self):
        if self.option == 'LB':
            self.nside = self.lblib.nside
            self.lmax = 3*self.nside - 1
            self.beam = np.reshape(self.lblib.beam,(1,self.lmax+1))
            self.fsky = self.lblib.nilc_fsky
            self.mask = self.lblib.nilc_mask
            self.ninv = np.reshape(np.array((self.lblib.nilc_mask,self.lblib.nilc_mask)),(2,1,hp.nside2npix(self.nside)))
        elif self.option == 'S4':
            self.nside = self.s4lib.nside
            self.lmax = 3*self.nside - 1
            self.beam = np.reshape(self.s4lib.beam,(1,self.lmax+1))
            self.fsky = self.s4lib.nilc_fsky
            self.mask = self.s4lib.nilc_mask
            self.ninv = np.reshape(np.array((self.s4lib.nilc_mask,self.s4lib.nilc_mask)),(2,1,hp.nside2npix(self.nside)))
        elif self.option == 'LBxS4':
            self.nside = self.s4lib.nside
            self.lmax = 3*self.nside - 1
            __beam__ = np.zeros((2,self.lmax+1))
            __beam__[0] = self.lblib.beam if self.lblib.lmax == self.s4lib.lmax else np.append(self.lblib.beam,np.zeros(self.s4lib.lmax-self.lblib.lmax))
            __beam__[1] = self.s4lib.beam
            self.beam = __beam__
            self.mask = hp.ud_grade(self.lblib.nilc_mask,self.s4lib.nside) * self.s4lib.nilc_mask
            __ninv__ = np.zeros((2,2,hp.nside2npix(self.nside)))
            __ninv__[:,0,:] = self.mask
            __ninv__[:,1,:] = self.mask
            self.ninv = __ninv__
            
            self.fsky = np.average(self.mask)
        else:
            raise ValueError("Unknown option")
    

    def QU(self,idx):
        if self.option == 'LB':
            Elm = self.lblib.NILC_Elm(idx)
            Q,U = hp.alm2map_spin([Elm,Elm*0],self.nside,2,self.lmax)
            del Elm
            QU = np.reshape(np.array((Q*self.mask,U*self.mask)),
                            (2,1,hp.nside2npix(self.nside)))/self.Tcmb
            return QU
        elif self.option == 'S4':
            Elm = self.s4lib.NILC_Elm(idx)
            _QU_ = hp.alm2map_spin([Elm,Elm*0],self.nside,2,self.lmax)
            del Elm
            Q,U = change_coord(np.array(_QU_),['C','G'])
            del _QU_
            QU = np.reshape(np.array((Q*self.mask,U*self.mask)),
                            (2,1,hp.nside2npix(self.nside)))/self.Tcmb
            return QU
        elif self.option == 'LBxS4':
            lbElm = self.lblib.NILC_Elm(idx)
            lbQU = hp.alm2map(np.array([lbElm*0,lbElm,lbElm*0]),self.nside)[1:]
            del lbElm
            s4Elm = self.s4lib.NILC_Elm(idx)
            _s4QU_ = hp.alm2map(np.array([s4Elm*0,s4Elm,s4Elm*0]),self.nside,)[1:]
            del s4Elm
            s4QU = change_coord(np.array(_s4QU_),['C','G'])
            del _s4QU_
            lbs4_QU = np.zeros((2,2,hp.nside2npix(self.nside)))
            lbs4_QU[0,0,:], lbs4_QU[1,0,:] = hp.ud_grade(lbQU,self.nside)*self.ninv[:,1,:][1]
            lbs4_QU[0,1,:], lbs4_QU[1,1,:] = s4QU*self.ninv[:,1,:][1]
            del (lbQU,s4QU)
            return lbs4_QU/self.Tcmb
        else:
            raise ValueError("Unknown option")
        
    
    def __invNL__(self,lib,idx,shaped=True):
        ne = lib.NILC_ncl(idx)
        if len(ne) < self.lmax+1:
            ne = np.append(ne,np.ones(self.lmax+1-len(ne))*ne[-1])
        ne = ne/self.Tcmb**2
        if shaped:
            return np.reshape(np.array((cli(ne[:self.lmax+1]),cli(ne[:self.lmax+1]*0))),(2,1,self.lmax+1))
        else:
            return cli(ne)


    def invNL(self,idx):
        if self.option == 'LB':
            return self.__invNL__(self.lblib,idx)
        elif self.option == 'S4':
            return self.__invNL__(self.s4lib,idx)
        elif self.option == 'LBxS4':
            lbs4_NL = np.zeros((2,2,self.lmax+1))
            lbs4_NL[0,0,:] = self.__invNL__(self.lblib,idx,shaped=False)
            lbs4_NL[0,1,:] = self.__invNL__(self.s4lib,idx,shaped=False)
            return lbs4_NL
        else:
            raise ValueError("Unknown option")
    
    def filtEmode(self,idx,wiener=False,eps=1e-4,status=None):
        fname = os.path.join(self.libdir,f"cinv_E_{idx:04d}.pkl")
        if not os.path.isfile(fname):
            QU = self.QU(idx)
            NL = self.invNL(idx)
            if (self.option == 'LB') or (self.option == 'S4'):
                channels = 1
            else:
                channels = 2
            E,B = cs.cninv.cnfilter_freq(2,channels,self.nside,self.lmax,self.cl_len[1:3,:self.lmax+1],
                                    self.beam, self.ninv,QU,chn=1,itns=[1000],filter="",
                                    eps=[eps],ro=10,inl=NL,stat=status)
            del B
            pl.dump(E,open(fname,'wb'))
            del (QU,NL)
        else:
            E = pl.load(open(fname,'rb'))
        
        if wiener:
            clee = self.cl_len[1,:self.lmax+1]
            E = cs.utils.almxfl(self.lmax,self.lmax,E,clee)
            return E
        else:
            return E

    def plot_W_E(self,idx):
        E = self.filtEmode(idx,wiener=True)
        we = cs.utils.alm2cl(self.lmax,E)
        plt.figure(figsize=(5,5))
        plt.loglog(we)
        plt.loglog(self.cl_len[1,:])
        plt.xlim(2,1000)
        plt.ylim(1e-18,1e-14)
        plt.legend(['Wiener E','Input E'], fontsize=15)
        plt.xlabel('$\ell$',fontsize=20)
        plt.ylabel('$C_\ell$',fontsize=20)
        



        

