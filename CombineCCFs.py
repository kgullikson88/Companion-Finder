import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.optimize import leastsq
from astropy.io import fits
import astropy.time as time
from astropy import units, constants
import FittingUtilities
import GenericSearch
from lmfit import Model
import statsmodels.api as sm
import h5py
import pandas as pd 
import time
import pickle
import seaborn as sns
from collections import defaultdict

BARY_DF = pd.read_csv('data/psi1draa_140p_28_37_ASW.dat', sep=' ', skipinitialspace=True, header=None)


def get_rv_correction(filename):
    header = fits.getheader(filename)
    jd = header['HJD']
    date = BARY_DF.ix[np.argmin(abs(BARY_DF[0]-jd))]
    return (date[1] + date[5] - date[2])*units.m.to(units.km)
    #return (date[2] + date[5] - date[1])*units.m.to(units.km)
    #return (date[5])*units.m.to(units.km)


def get_prim_rv(filename):
    header = fits.getheader(filename)
    jd = header['HJD']
    date = BARY_DF.ix[np.argmin(abs(BARY_DF[0]-jd))]
    return date[2]*units.m.to(units.km)


def get_centroid(x, y):
    return np.sum(x*y) / np.sum(y)


def fit_gaussian(x, y):
    gauss = lambda x, C, A, mu, sig: C + A*np.exp(-(x-mu)**2 / (2.*sig**2))
    errfcn = lambda p, x, y: (y - gauss(x, *p))**2

    #pars = [1, -0.5, 0, 10]
    pars = [0, 0.5, 0, 10]
    fitpars, success = leastsq(errfcn, pars, args=(x, y))
    #plt.plot(x, y)
    #plt.plot(x, gauss(x, *pars))
    #plt.plot(x, gauss(x, *fitpars))
    #plt.show()
    #time.sleep(1)
    return fitpars


def get_ccfs(T=4000, vsini=5, logg=4.5, metal=0.5, hdf_file='Cross_correlations/CCF.hdf5', xgrid=np.arange(-400, 400, 1), addmode='simple'):
    """
    Get the cross-correlation functions for the given parameters, for all stars
    """
    ccfs = []
    filenames = []
    rv_shift = {} if T > 6000 else pickle.load(open('rvs.pkl'))
    with h5py.File(hdf_file) as f:
        starname = 'psi1 Dra A'
        date_list = f[starname].keys()
        for date in date_list:
            datasets = f[starname][date].keys()
            for ds_name in datasets:
                ds = f[starname][date][ds_name]
                if (ds.attrs['T'] == T and ds.attrs['vsini'] == vsini and
                            ds.attrs['logg'] == logg and ds.attrs['[Fe/H]'] == metal and
                            ds.attrs['addmode'] == addmode):
                    vel, corr = ds.value
                    #ccf = spline(vel[::-1]*-1, (1.0-corr[::-1]))
                    ccf = spline(vel[::-1]*-1, corr[::-1])
                    fname = ds.attrs['fname']
                    vbary = get_rv_correction(fname)
                    
                    #cont = FittingUtilities.Continuum(xgrid, ccf(xgrid-vbary), fitorder=2, lowreject=2.5, highreject=5)
                    #normed_ccf = ccf(xgrid-vbary)/cont
                    cont = FittingUtilities.Continuum(xgrid, ccf(xgrid-vbary), fitorder=2, lowreject=5, highreject=2.5)
                    normed_ccf = ccf(xgrid-vbary) - cont
                    
                    if T <= 6000:
                        centroid = rv_shift[fname]
                        #top = 1.0
                        #amp = 1.0 - min(normed_ccf)
                        top = 0.0
                        amp = max(normed_ccf)
                        #idx = np.argmin(np.abs(xgrid-centroid))
                        #amp = normed_ccfs[idx]
                    else:
                        gauss_pars = fit_gaussian(xgrid, normed_ccf)
                        centroid = gauss_pars[2]
                        amp = gauss_pars[1]
                        top = gauss_pars[0]
                        amp = 0.5
                    print(centroid, fname)

                    #cont = FittingUtilities.Continuum(xgrid, ccf(xgrid-vbary+centroid), fitorder=2, lowreject=2.5, highreject=5)
                    #normed_ccf = (ccf(xgrid-vbary+centroid) / cont - top)  * 0.5/abs(amp) + top
                    cont = FittingUtilities.Continuum(xgrid, ccf(xgrid-vbary+centroid), fitorder=2, lowreject=5, highreject=2.5)
                    normed_ccf = (ccf(xgrid-vbary+centroid) - cont)  * 0.5/abs(amp)

                    filenames.append(fname)
                    ccfs.append(normed_ccf)
                    rv_shift[fname] = centroid

    if T > 6000:
        pickle.dump(rv_shift, open('rvs.pkl', 'w'))

    return np.array(ccfs), filenames



def CombineSmoothedCCFS():
    # Parse command line arguments
    T = 4000
    vsini = 5
    logg = 4.5
    metal = 0.0
    addmode = 'simple'
    for arg in sys.argv[1:]:
        if "-T" in arg:
            T = int(arg.split("=")[1])
        elif "-v" in arg:
            vsini = int(arg.split("=")[1])
        elif "-l" in arg:
            logg = float(arg.split("=")[1])
        elif "-m" in arg:
            metal = float(arg.split("=")[1])
        elif '-a' in arg:
            addmode = arg.split('=')[1]

    
    summary = defaultdict(list)
    qvals, snr, ccfs, file_list = fit_q(T, vsini, logg, metal, plot=True, addmode=addmode)
    sys.exit()
    
    for Ti, temp in enumerate(range(3000, 6000, 100)):
        print('Finding the best q for T = {} K'.format(temp))
        plot = True if temp == T else False
        qvals, snr, ccfs, file_list = fit_q(temp, vsini, logg, metal, plot=plot)
        
        for q, s in zip(qvals, snr):
            summary['T'].append(temp)
            summary['q'].append(q)
            summary['S/N'].append(s)

    return pd.DataFrame(data=summary)
    


def fit_q(T, vsini, logg, metal, ccfs=None, original_files=None, plot=True, addmode='simple'):
    # Get all the ccfs with the requested parameters
    dV = 0.1
    c = constants.c.cgs.to(units.m/units.s).value
    xgrid = np.arange(-400, 400+dV/2., dV)
    if ccfs is None or original_files is None:
        ccfs, original_files = get_ccfs(T=T, vsini=vsini, logg=logg, metal=metal,
                                        hdf_file="Cross_correlations/CCF.hdf5",
                                        xgrid=xgrid, addmode=addmode)


    # Get the average ccf
    avg_ccf = np.mean(ccfs, axis=0)

    # Make plots if requested
    if plot:
        fig1, ax1 = plt.subplots(1, 1)
        fig2, ax2 = plt.subplots(1, 1)
        fig3, ax3 = plt.subplots(1, 1)
        fig4, ax4 = plt.subplots(1, 1)
        fig5, ax5 = plt.subplots(1, 1)

        ax1.imshow(ccfs, aspect='auto')
        #ax1.colorbar()

        for i in range(ccfs.shape[0]):
            ax2.plot(xgrid, ccfs[i], 'k-', alpha=0.1)

        ax2.plot(xgrid, avg_ccf, 'r-')

    # Normalize
    normed_ccfs = ccfs - avg_ccf

    if plot:
        low, high = np.min(normed_ccfs), np.max(normed_ccfs)
        rng = max(abs(low), abs(high))
        vmin = np.sign(low) * rng
        vmax = np.sign(high) * rng
        ax3.imshow(normed_ccfs, aspect='auto', vmin=vmin, vmax=vmax)
        #ax3.colorbar()



    # Get the stacked CCF for various values of q (mass-ratio)
    prim_vel = [get_prim_rv(f) for f in original_files]
    qvals = np.arange(0.1, 0.5, 0.01)
    space = 0.01
    plt.figure(5)
    snr = []
    for j, q in enumerate(qvals):
        total_ccf = np.zeros(normed_ccfs.shape[1])
        minvel = np.inf
        for i in range(normed_ccfs.shape[0]):
            ccf = spline(xgrid, normed_ccfs[i])
            vel = prim_vel[i] * (1. - 1./q)
            if vel < minvel:
                minvel = vel
            total_ccf += ccf(xgrid + vel)
        good = np.where(xgrid > xgrid[0] - minvel)[0]
        if plot:
            ax4.plot(xgrid[good], total_ccf[good]/float(normed_ccfs.shape[0]) + j*space, label='q = {:.3f}'.format(q))
        gauss_pars = fit_gaussian(xgrid[good], total_ccf[good]/float(normed_ccfs.shape[0]))
        const, amp, mu, sig = gauss_pars
        sig = abs(sig)
        noise_idx = np.where(abs(xgrid[good] - mu)/sig > 3)[0]
        noise = np.std(total_ccf[good][noise_idx]/float(normed_ccfs.shape[0]))
        snr.append(abs(amp)/noise)

    print('\nBest q = {:.3f}\n\n'.format(qvals[np.argmax(snr)]))

    if plot:
        ax4.legend(loc='best', fancybox=True)

        ax5.plot(qvals, snr)
        ax5.set_xlabel(r'$q \equiv M_s/M_p$')
        ax5.set_ylabel('Detection Significance')

        plt.show()

    return qvals, snr, ccfs, original_files




if __name__ == "__main__":
    df = CombineSmoothedCCFS()
    #df.to_csv('Summary.csv')