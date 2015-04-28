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

    pars = [1, -0.5, 0, 10]
    fitpars, success = leastsq(errfcn, pars, args=(x, y))
    #plt.plot(x, y)
    #plt.plot(x, gauss(x, *pars))
    #plt.plot(x, gauss(x, *fitpars))
    #plt.show()
    #time.sleep(1)
    return fitpars


def get_ccfs(T=4000, vsini=5, logg=4.5, metal=0.5, hdf_file='Cross_correlations/CCF.hdf5', xgrid=np.arange(-400, 400, 1)):
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
                            ds.attrs['logg'] == logg and ds.attrs['[Fe/H]'] == metal):
                    vel, corr = ds.value
                    ccf = spline(vel[::-1]*-1, (1.0-corr[::-1]))
                    fname = ds.attrs['fname']
                    vbary = get_rv_correction(fname)
                    
                    cont = FittingUtilities.Continuum(xgrid, ccf(xgrid-vbary), fitorder=2, lowreject=2.5, highreject=5)
                    normed_ccf = ccf(xgrid-vbary)/cont
                    centroid = get_centroid(xgrid, 1.0-normed_ccf)
                    
                    if T <= 6000:
                        centroid = rv_shift[fname]
                        top = 1.0
                        amp = 1.0 - min(normed_ccf)
                    else:
                        gauss_pars = fit_gaussian(xgrid, normed_ccf)
                        centroid = gauss_pars[2]
                        amp = gauss_pars[1]
                        top = gauss_pars[0]
                    print(centroid, fname)

                    cont = FittingUtilities.Continuum(xgrid, ccf(xgrid-vbary+centroid), fitorder=2, lowreject=2.5, highreject=5)
                    normed_ccf = (ccf(xgrid-vbary+centroid) / cont - top)  * 0.5/abs(amp) + top

                    filenames.append(fname)
                    ccfs.append(normed_ccf)
                    rv_shift[fname] = centroid

    if T > 6000:
        pickle.dump(rv_shift, open('rvs.pkl', 'w'))

    return np.array(ccfs), filenames


def fit_q(rv1, rv2):
    goodindices = np.where(rv2 < 0)[0]
    rv1 = rv1[goodindices]
    rv2 = rv2[goodindices]
    X = rv1.copy()
    X = sm.add_constant(X)
    fitter = sm.RLM(rv2, X, M=sm.robust.norms.TukeyBiweight())
    result = fitter.fit()
    a = result.params[1]
    b = result.params[0]
    a_err = result.bse[1]
    b_err = result.bse[0]
    C = (b + 12.29*a)/(1.0-a)
    C_var = ( (12.29*a/(1.0-a)*b_err)**2 + ( ((b+12.29)/(1-a) + (1-a)*(b+12.29*a)/(1-a)**2)*a_err )**2 )
    print "q = ", -1.0/a, " +/- ", 1.0/a**2 * a_err
    print "C = ", C, "+/-", np.sqrt(C_var)
    return lambda x: a*x + b


def CombineSmoothedCCFS():
    # Parse command line arguments
    T = 4000
    vsini = 5
    logg = 4.5
    metal = 0.0
    c = constants.c.cgs.to(units.m/units.s).value
    for arg in sys.argv[1:]:
        if "-T" in arg:
            T = int(arg.split("=")[1])
        elif "-v" in arg:
            vsini = int(arg.split("=")[1])
        elif "-l" in arg:
            logg = float(arg.split("=")[1])
        elif "-m" in arg:
            metal = float(arg.split("=")[1])

    # Get all the ccfs with the requested parameters
    dV = 0.1
    xgrid = np.arange(-400, 400+dV/2., dV)
    ccfs, original_files = get_ccfs(T=T, vsini=vsini, logg=logg, metal=metal,
                                    hdf_file="Cross_correlations/CCF_raw.hdf5",
                                    xgrid=xgrid)


    plt.figure(1)
    plt.imshow(ccfs, aspect='auto')
    plt.colorbar()

    plt.figure(3)
    for i in range(ccfs.shape[0]):
        plt.plot(xgrid, ccfs[i], 'k-', alpha=0.1)


    # Get the average ccf
    avg_ccf = np.mean(ccfs, axis=0)
    plt.plot(xgrid, avg_ccf, 'r-')

    # Normalize
    normed_ccfs = ccfs - avg_ccf

    plt.figure(2)
    plt.imshow(normed_ccfs, aspect='auto')
    plt.colorbar()



    # Get the stacked CCF for various values of q (mass-ratio)
    prim_vel = [get_prim_rv(f) for f in original_files]
    qvals = np.arange(0.1, 0.5, 0.01)
    #qvals = np.arange(0.20, 0.30, 0.025)
    space = 0.01
    plt.figure(5)
    snr = []
    for j, q in enumerate(qvals):
        total_ccf = np.zeros(normed_ccfs.shape[1])
        minvel = np.inf
        for i in range(normed_ccfs.shape[0]):
            ccf = spline(xgrid, normed_ccfs[i])
            vel = prim_vel[i] * (1. - 1./q)
            #print(vel)
            if vel < minvel:
                minvel = vel
            total_ccf += ccf(xgrid + vel)
        print(minvel)
        good = np.where(xgrid > xgrid[0] - minvel)[0]
        plt.plot(xgrid[good], total_ccf[good]/float(normed_ccfs.shape[0]) + j*space, label='q = {:.3f}'.format(q))
        gauss_pars = fit_gaussian(xgrid[good], total_ccf[good]/float(normed_ccfs.shape[0]))
        print(gauss_pars)
        const, amp, mu, sig = gauss_pars
        sig = abs(sig)
        noise_idx = np.where(abs(xgrid[good] - mu)/sig > 3)[0]
        noise = np.std(total_ccf[good][noise_idx]/float(normed_ccfs.shape[0]))
        snr.append(abs(amp)/noise)
    plt.legend(loc='best', fancybox=True)

    #TODO: This seems to work and gives mass ratios near 0.2-0.25. 
    #But, the significance is higher for higher temperatures, while I 
    #would think it should be for cooler temperatures (closer to the companion spectrum). 
    #Figure out what is going on and if I am in fact seeing the companion signature!

    plt.figure(6)
    plt.plot(qvals, snr)

    print('\nBest q = {:.3f}\n\n'.format(qvals[np.argmax(snr)]))

    plt.show()

    sys.exit()

    # Get the primary star rv for each observation
    rv_list = get_rv(all_ccf_files)

    #shift_ccfs = get_shifted_ccfs(rv_list, ccfs, xgrid)
    shift_ccfs = bary_correct_ccfs(rv_list, ccfs, xgrid)
    #for i in range(shift_ccfs.shape[0]):
    #    plt.plot(xgrid, shift_ccfs[i], 'k-')
    #    plt.title(str(i+1) + ": " + all_ccf_files[i])
    #    plt.grid(True)
    #    plt.show()
    #sys.exit()
    plt.figure(2)
    plt.imshow(shift_ccfs, aspect='auto')
    plt.colorbar()

    # Get velocities
    maxindices = np.argmax(shift_ccfs, axis=1)
    maxvels = xgrid[maxindices]
    measured_rvs = np.array(rv_list)/1000.0
    #goodindices = np.where(abs(maxvels) < 50.0)[0]
    #fit = np.poly1d(np.polyfit(measured_rvs[goodindices], maxvels[goodindices], 1))
    #print fit

    #Fit q
    fcn = fit_q(measured_rvs, maxvels)
    plt.figure(3)
    plt.scatter(measured_rvs, maxvels, c='black')
    plt.plot(measured_rvs, fcn(measured_rvs), 'r-', linewidth=2)
    plt.xlabel(r"dRV (km s$^{-1}$)")
    plt.ylabel(r"$RV_{CCF,\ast,2}$ (km s$^{-1}$)")

    plt.show()

    np.savetxt("Velocities.txt", np.transpose((measured_rvs, maxvels)))

    sys.exit()


    # Try a variety of mass-ratios to line up the ccfs
    qvals = np.linspace(0.05, 1.0, 100)
    #qvals = np.array((0.35, 0.4, 0.45))
    #qvals = np.arange(0.35, 0.5, 0.02)
    sigvals = []
    for q in qvals:
        shift_ccfs = get_shifted_ccfs(rv_list, ccfs, q, xgrid)
        total = np.sum(shift_ccfs, axis=0)
        cont = FittingUtilities.Continuum(xgrid, total, fitorder=5, lowreject=10, highreject=2)
        total -= cont
        left = np.searchsorted(xgrid, -300)
        right = np.searchsorted(xgrid, -100)
        std = np.std(total[left:right])
        mean = np.mean(total[left:right])
        sig = (max(total) - mean)/std
        #sig = (total[zero] - mean)/std
        #print "q = {} --> {} sigma".format(q, sig)
        sigvals.append(sig)
        #plt.plot(xgrid, total, 'k-')
        #plt.title("q = {}".format(q))
        #plt.show()


    bestq = qvals[np.argmax(sigvals)]
    shift_ccfs = get_shifted_ccfs(rv_list, ccfs, bestq, xgrid)
    total = np.sum(shift_ccfs, axis=0)
    cont = FittingUtilities.Continuum(xgrid, total, fitorder=5, lowreject=10, highreject=2)
    total -= cont
    left = np.searchsorted(xgrid, -300)
    right = np.searchsorted(xgrid, -100)
    std = np.std(total[left:right])
    mean = np.mean(total[left:right])
    sig = (max(total) - mean)/std
    print "\n Best mass ratio q = {} --> {} sigma".format(bestq, sig)

    plt.figure(4)
    plt.plot(xgrid, total/float(shift_ccfs.shape[0]), 'k-', linewidth=2)
    plt.xlabel("Velocity", fontsize=15)
    plt.ylabel("CCF Power", fontsize=15)
    plt.figure(2)
    plt.imshow(shift_ccfs, aspect='auto')
    plt.colorbar()
    plt.figure(3)
    plt.plot(qvals, sigvals, 'k-', linewidth=2)
    plt.xlabel("$q \equiv M_2 / M_1$", fontsize=15)
    plt.ylabel("CCF Significance", fontsize=15)

    #plt.plot(xgrid, np.sum(shift_ccfs, axis=0), label="After Secondary line-up")
    #plt.legend(loc='best')
    plt.show()



if __name__ == "__main__":
    CombineSmoothedCCFS()
    #CombineRawCCFS()