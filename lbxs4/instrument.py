import os
from typing import Any
import numpy as np
from astropy.table import QTable
from astropy import units as u
from lbxs4.config import DATDIR

class LiteBIRD:

    def __init__(self,fname='litebird_instrument_model.tbl'):
        fname = os.path.join(DATDIR,fname)
        table = QTable.read(fname,format="ascii.ipac")
        self.center_frequency= table['center_frequency'].value
        self.fwhm = table['fwhm'].value
        self.tag = table['tag'].value
        self.nside = table['nside'].value
        self.tag2index = dict(zip(self.tag,np.arange(len(self.tag))))

    def __get_index__(self,band):
        assert band in self.tag, "band not found in LiteBIRD,try LiteBIRD().tag"
        return self.tag2index[band]

    def check_band(self,band):
        assert band in self.tag, "band not found in LiteBIRD,try LiteBIRD().tag"
    
    def get_fwhm(self,band):
        return self.fwhm[self.__get_index__(band)]
    
    def get_nside(self,band):
        return self.nside[self.__get_index__(band)]
    
    def get_frequency(self,band):
        return self.center_frequency[self.__get_index__(band)]

    

                            