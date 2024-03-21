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
            self.fsky = self.lblib.fsky
            self.mask = self.lblib.mask
            self.ninv = np.reshape(np.array((self.lblib.mask,self.lblib.mask)),(2,1,hp.nside2npix(self.nside)))
        elif self.option == 'S4':
            self.nside = self.s4lib.nside
            self.lmax = 3*self.nside - 1
            self.beam = np.reshape(self.s4lib.beam,(1,self.lmax+1))
            self.fsky = self.s4lib.fsky
            self.mask = self.s4lib.mask
            self.ninv = np.reshape(np.array((self.s4lib.mask,self.s4lib.mask)),(2,1,hp.nside2npix(self.nside)))
        elif self.option == 'LBxS4':
            self.nside = self.s4lib.nside
            self.lmax = 3*self.nside - 1
            __beam__ = np.zeros((2,self.lmax+1))
            __beam__[0] = self.lblib.beam if self.lblib.lmax == self.s4lib.lmax else np.append(self.lblib.beam,np.zeros(self.s4lib.lmax-self.lblib.lmax))
            __beam__[1] = self.s4lib.beam
            self.beam = __beam__
            self.mask = hp.ud_grade(self.lblib.mask,self.s4lib.nside) * self.s4lib.mask
            __ninv__ = np.zeros((2,2,hp.nside2npix(self.nside)))
            __ninv__[:,0,:] = self.mask
            __ninv__[:,1,:] = self.mask
            self.ninv = __ninv__
            
            self.fsky = np.average(self.mask)
        else:
            raise ValueError("Unknown option")
    
    def __QU__(self,Elm,which):
        Q,U = hp.alm2map([Elm*0,Elm,Elm*0],self.nside)[1:]
        QU = np.reshape(np.array((Q*self.mask,U*self.mask)),(2,1,hp.nside2npix(self.nside)))/self.Tcmb
        del (Q,U)
        return QU
    

    def QU(self,idx):
        if self.option == 'LB':
            Elm = self.lblib.NILC_Elm(idx)
            QU = self.__QU__(Elm,'lb')
            del Elm
            return QU
        elif self.option == 'S4':
            Elm = self.s4lib.NILC_Elm(idx)
            QU = self.__QU__(Elm,'s4')
            del Elm
            return QU
        elif self.option == 'LBxS4':
            lbElm = self.lblib.NILC_Elm(idx)
            lbQU = self.__QU__(lbElm,'lb')
            del lbElm
            s4Elm = self.s4lib.NILC_Elm(idx)
            s4QU = self.__QU__(s4Elm,'s4')
            del s4Elm
            lbs4_QU = np.zeros((2,2,hp.nside2npix(self.nside)))
            lbs4_QU[0,0,:], lbs4_QU[1,0,:] = lbQU
            lbs4_QU[0,1,:], lbs4_QU[1,1,:] = s4QU
            del (lbQU,s4QU)
            return lbs4_QU
        else:
            raise ValueError("Unknown option")

            
    def __invNL__(self,lib,idx,shaped=True):
        ne = lib.NILC_ncl(idx)/self.Tcmb**2
        if len(ne) < self.lmax+1:
            ne = np.append(ne,np.ones(self.lmax+1-len(ne))*ne[-1])
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

    def plot_W_E(self,idx,lmin=2,lmax=1000,ymin=1e-18,ymax=1e-14):
        E = self.filtEmode(idx,wiener=True)
        we = cs.utils.alm2cl(self.lmax,E)
        plt.figure(figsize=(5,5))
        plt.loglog(we/self.fsky)
        plt.loglog(self.cl_len[1,:])
        #plt.xlim(lmin,lmax)
        #plt.ylim(ymin,ymax)
        plt.legend(['Wiener E','Input E'], fontsize=15)
        plt.xlabel('$\ell$',fontsize=20)
        plt.ylabel('$C_\ell$',fontsize=20)
        



        

