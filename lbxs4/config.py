import os

__BASEDIR__ = '/global/cfs/cdirs/cmbs4xlb/v1/'

__MODDIR__ =  os.path.dirname(os.path.realpath(__file__))

DATDIR = os.path.join(__MODDIR__,'..','Data')
CMBDIR = os.path.join(__BASEDIR__,'cmb')
FGDIR = os.path.join(__BASEDIR__,'fg','lb','galactic')
NOISEDIR = os.path.join(__BASEDIR__,'noise','lb')
MASSDIR = os.path.join(__BASEDIR__,'mass','alm')
SPECTRADIR = os.path.join(__BASEDIR__,'mass','spec')