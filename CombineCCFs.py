import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from astropy.io import fits
import astropy.time as time
from astropy import units, constants
import FittingUtilities
import GenericSearch
from lmfit import Model
import statsmodels.api as sm
import h5py
import pandas as pd 

BARY_DF = pd.read_csv('data/psi1draa_140p_28_37_ASW.dat', sep=' ', skipinitialspace=True, header=None)


def get_rv_correction(filename):
    header = fits.getheader(filename)
    jd = header['HJD']
    date = BARY_DF.ix[np.argmin(abs(BARY_DF[0]-jd))]
    return (date[1] + date[5] - date[2])*units.m.to(units.km)



def get_ccfs(T=4000, vsini=5, logg=4.5, metal=0.5, hdf_file='Cross_correlations/CCF.hdf5', xgrid=np.arange(-400, 400, 1)):
    """
    Get the cross-correlation functions for the given parameters, for all stars
    """
    ccfs = []
    filenames = []
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
                    ccf = spline(vel[::-1]*-1, corr[::-1])
                    fname = ds.attrs['fname']
                    vbary = get_rv_correction(fname)
                    filenames.append(fname)
                    ccfs.append(ccf(xgrid-vbary))

    return np.array(ccfs), filenames


def get_rv(fileList, get_helcorr=False):
    # Read in the rv data
    bjd, rv = np.loadtxt("psi1draa_100_120_mcomb1.dat", usecols=(0,1), unpack=True)

    rv_shifts = []
    for fname in fileList:
        fitsfile = fname.split("/")[-1].split(".")[0] + ".fits"
        header = fits.getheader(fitsfile)
        t = time.Time(header['date'], format='isot', scale='utc')
        idx = np.argmin(abs(bjd - t.jd))
        vel = GenericSearch.HelCorr(header, observatory="McDonald")*1e3 if get_helcorr else rv[idx]
        rv_shifts.append(vel)

    return rv_shifts


def get_shifted_ccfs(rv_list, ccfs, xgrid):
    shift_ccfs = ccfs.copy()
    for i, (rv, corr, scorr) in enumerate(zip(rv_list, ccfs, shift_ccfs)):
        fcn = spline(xgrid, corr)
        scorr = fcn(xgrid + rv*1e-3 * (q - 1.0)/q)
        scorr[scorr < 0] = 0
        scorr[scorr > 0.3] = 0.3
        shift_ccfs[i] =scorr
    return shift_ccfs

def bary_correct_ccfs(rv_list, ccfs, xgrid):
    shift_ccfs = ccfs.copy()
    for i, (rv, corr, scorr) in enumerate(zip(rv_list, ccfs, shift_ccfs)):
        fcn = spline(xgrid, corr)
        scorr = fcn(xgrid - rv*1e-3)
        shift_ccfs[i] =scorr
    return shift_ccfs


def fit_q_old(rv1, rv2):
    model_fcn = lambda x,a,b: a*x + b
    fitter = Model(model_fcn)
    goodindices = np.where(abs(rv2) < 50.0)[0]
    result = fitter.fit(rv2[goodindices], x=rv1[goodindices], a=-2, b=0)
    print result.fit_report()
    a = result.params['a'].value
    a_err = result.params['a'].stderr
    b = result.params['b'].value
    b_err = result.params['b'].stderr
    C = (b + 12.29*a)/(1.0-a)
    C_var = ( (12.29*a/(1.0-a)*b_err)**2 + ( ((b+12.29)/(1-a) + (1-a)*(b+12.29*a)/(1-a)**2)*a_err )**2 )
    print "q = ", -1.0/a, " +/- ", 1.0/a**2 * a_err
    print "C = ", C, "+/-", np.sqrt(C_var)

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

    # Get the average ccf
    minval = np.min(ccfs)
    if minval < 0:
        ccfs += abs(minval) + 1e-3
    #ccfs += np.min(ccfs) + 1e-3
    avg_ccf = np.median(ccfs, axis=0)
    #avg_ccf -= np.median(avg_ccf)

    # Subtract the median of each ccf
    #meds = np.median(ccfs, axis=1)
    #ccfs = (ccfs.T - meds).T

    # Normalize
    print(np.min(ccfs), np.min(avg_ccf))
    print(avg_ccf)
    normed_ccfs = ccfs / avg_ccf

    plt.figure(2)
    plt.imshow(normed_ccfs, aspect='auto', vmin=-1, vmax=1)
    plt.colorbar()

    #for i in range(normed_ccfs.shape[0]):
    #    plt.figure(i+3)
    #    plt.plot(xgrid, normed_ccfs[i])
    plt.show()

    #plt.figure(2)
    #plt.plot(xgrid, ccfs[10])
    #plt.show()
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