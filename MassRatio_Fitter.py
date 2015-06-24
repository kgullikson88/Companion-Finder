"""
Fit the mass-ratio using emcee
"""

import numpy as np 
import emcee
import matplotlib.pyplot as plt 
import seaborn as sns
import triangle
from CombineCCFs import get_rv
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter
import os
import pandas as pd

sns.set_context('paper', font_scale=2.0)

# Functions for emcee
def lnlike_partial(pars, v, v_err, rv1_pred):
    q, dv, lnf = pars
    rv2_pred = -rv1_pred / q
    inv_sigma2 = 1.0/(np.exp(lnf)*v_err**2)
    return -0.5*np.nansum((rv2_pred - (v+rv1_pred-dv))**2 * inv_sigma2 - np.log(inv_sigma2))

def lnprior_partial(pars):
    q, dv, lnf = pars
    if 0 < q < 1 and -20 < dv < 20:
        return 0.0
    return -np.inf

def lnprob_partial(pars, v, v_err, rv1_pred):
    lp = lnprior_partial(pars)
    return lp + lnlike_partial(pars, v, v_err, rv1_pred) if np.isfinite(lp) else -np.inf


# Function to get the chains given a set of parameters for the primary fit
def fit_partial(T0, P, e, K1, w, t, rv2, rv2_err):
    """
    Get MCMC samples for the mass-ratio, velocity shift, and error scaling
    for the orbital parameters T0-w (fit by Stefano Meschiari)
    """
    # Get the predicted velocity of the primary at each time
    rv1_pred = get_rv(T0=T0, P=P, e=e, K1=K1, w=w*np.pi/180, t=t)

    # Initialize MCMC sampler
    initial_pars = [0.47, -5.4, -3.6]
    ndim = len(initial_pars)
    nwalkers = 100
    p0 = emcee.utils.sample_ball(initial_pars, std=[1e-6]*ndim, size=nwalkers)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_partial, args=(rv2, rv2_err, rv1_pred), threads=2)

    # Run the sampler
    pos, lp, state = sampler.run_mcmc(p0, 1000)

    # Save the last 500 (we will just have to hope that the sampler sufficiently burns-in in 500 steps. That is true in tests I've done)
    samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
    return samples


def lnlike_full(pars, t1, v1, v1_err, t2, v2, v2_err):
    K1, K2, P, T0, w, e, dv1, dv2, lnf = pars
    rv1_pred = np.array([get_rv(T0=T0, P=P, e=e, K1=K1, w=w, t=t) for t in t1])
    rv2_pred = -np.array([get_rv(T0=T0, P=P, e=e, K1=K2, w=w, t=t) for t in t2])
    
    inv_sigma2_1 = 1.0/v1_err**2
    inv_sigma2_2 = 1.0/(np.exp(lnf)*v2_err**2)
    s1 = np.nansum((rv1_pred - (v1-dv1))**2 * inv_sigma2_1 - np.log(inv_sigma2_1))
    s2 = np.nansum((rv2_pred - (v2 - rv2_pred*K1/K2 - dv2))**2 * inv_sigma2_2 - np.log(inv_sigma2_2))
    #print(s1)
    #print(s2)
    return -0.5*(s1+s2)
    #return -0.5*np.nansum((rv2_pred[first:] - (v[first:]+rv1_pred[first:]-dv))**2 * inv_sigma2 - np.log(inv_sigma2))

def lnprior_full(pars):
    """Gaussian prior
    """
    K1, K2, P, T0, w, e, dv1, dv2, lnf = pars
    #if 4 < K1 < 6 and K2 > K1 and 0.6 < e < 0.7 and 6000 < P < 8500 and 0.35 < w < 0.7 and -20 < dv1 < 20 and -20 < dv2 < 20 and lnf < 0:
    #if 3 < K1 < 7 and K2 > K1 and 0.2 < e < 1. and 5000 < P < 9000 and 0.15 < w < 0.9 and -20 < dv1 < 20 and -20 < dv2 < 20 and lnf < 0:
    #    return 0.0
    if K2 > K1 and -20 < dv1 < 20 and -20 < dv2 < 20 and lnf < 0:
        return -0.5*((K1-5.113)**2/0.1**2 + (P-7345)**2/1000**2 + (T0-2449824)**2/1000**2 + 
                     (w-0.506)**2/0.035**2 + (e-0.669)**2/0.016**2 + 
                     np.log(2*np.pi*(0.1**2 + 1000**2 + 1000**2 + 0.035**2 + 0.016**2)))
    return -np.inf

def lnprob_full(pars, t1, v1, v1_err, t2, v2, v2_err):
    lp = lnprior_full(pars)
    return lp + lnlike_full(pars, t1, v1, v1_err, t2, v2, v2_err) if np.isfinite(lp) else -np.inf


def full_sb2_fit(t1, rv1, rv1_err, t2, rv2, rv2_err):
    """
    Do a full SB2 fit.
    """
    initial_pars = [5.113, 5.113/0.469, 7345, 2449824, 29*np.pi/180., 0.669, 4.018, -5.38, -3.61]

    ndim = len(initial_pars)
    nwalkers = 300
    p0 = emcee.utils.sample_ball(initial_pars, std=[1e-6]*ndim, size=nwalkers)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_full, args=(t1, rv1, rv1_err, t2, rv2, rv2_err), threads=2)

    # Run for a while
    for i, result in enumerate(sampler.sample(p0, iterations=2000)):
        if i%10 == 0:
            print('Done with burn-in iteration {:03d}'.format(i))

    # Get the best chain position and resample from there
    pos, lnp, state = result
    best_pars = pos[np.argmax(lnp)]
    p1 = emcee.utils.sample_ball(best_pars, std=[1e-6]*ndim, size=nwalkers)

    # Run again (this is the production run)
    sampler.reset()
    for i, result in enumerate(sampler.sample(p1, iterations=2000)):
        if i%10 == 0:
            print('Done with production iteration {:03d}'.format(i))

    return sampler





def lnlike_sb1(pars, t1, v1, v1_err):
    K1, P, T0, w, e, dv1 = pars
    rv1_pred = get_rv(T0=T0, P=P, e=e, K1=K1, w=w, t=t1) + dv1
    
    inv_sigma2_1 = 1.0/v1_err**2
    s1 = np.nansum((rv1_pred - v1)**2 * inv_sigma2_1 - np.log(inv_sigma2_1))

    return -0.5*s1

def lnprior_sb1(pars):
    """Gaussian prior
    """
    K1, P, T0, w, e, dv1 = pars
    #if 3 < K1 < 7 and K2 > K1 and 0.6 < e < 0.7 and 6000 < P < 8500 and 0.35 < w < 0.7 and -20 < dv1 < 20 and -20 < dv2 < 20 and lnf < 0:
    if 3 < K1 < 7 and 0.2 < e < 1. and 5000 < P < 9000 and 0.15 < w < 0.9 and -20 < dv1 < 20:
        return 0.0
    return -np.inf

def lnprob_sb1(pars, t1, v1, v1_err):
    lp = lnprior_sb1(pars)
    return lp + lnlike_sb1(pars, t1, v1, v1_err) if np.isfinite(lp) else -np.inf


def sb1_fit(t1, rv1, rv1_err):
    """
    Fit the primary star rvs only. This should be consistent with Stefano's fit.
    """
    initial_pars = [5.113, 7345, 2449824, 29*np.pi/180., 0.669, 4.018]

    ndim = len(initial_pars)
    nwalkers = 300
    p0 = emcee.utils.sample_ball(initial_pars, std=[1e-6]*ndim, size=nwalkers)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_sb1, args=(t1, rv1, rv1_err), threads=2)

    # Run for a while
    for i, result in enumerate(sampler.sample(p0, iterations=2000)):
        if i%10 == 0:
            print('Done with burn-in iteration {:03d}'.format(i))
    return sampler
    
    """
    # Get the best chain position and resample from there
    pos, lnp, state = result
    best_pars = pos[np.argmax(lnp)]
    p1 = emcee.utils.sample_ball(best_pars, std=[1e-6]*ndim, size=nwalkers)

    # Run again (this is the production run)
    sampler.reset()
    for i, result in enumerate(sampler.sample(p1, iterations=2000)):
        if i%10 == 0:
            print('Done with production iteration {:03d}'.format(i))

    return sampler
    """


def plot(pars, t1, v1, v1_err, t2, v2, v2_err):
    K1, K2, P, T0, w, e, dv1, dv2, lnf = pars
    rv1_pred = get_rv(T0=T0, P=P, e=e, K1=K1, w=w, t=t1)
    rv2_pred = -get_rv(T0=T0, P=P, e=e, K1=K2, w=w, t=t2)
    tplot = np.linspace(min(min(t1), min(t2)), max(max(t2), max(t2)), 100)
    rv1_plot = get_rv(T0=T0, P=P, e=e, K1=K1, w=w, t=tplot)
    rv2_plot = get_rv(T0=T0, P=P, e=e, K1=-K2, w=w, t=tplot)
    
    inv_sigma2_1 = 1.0/v1_err**2
    inv_sigma2_2 = 1.0/(np.exp(lnf)*v2_err**2)
    s1 = np.nansum((rv1_pred - (v1-dv1))**2 * inv_sigma2_1 - np.log(inv_sigma2_1))
    s2 = np.nansum((rv2_pred - (v2 - rv2_pred*K1/K2 - dv2))**2 * inv_sigma2_2 - np.log(inv_sigma2_2))
    
    #fig, ax = plt.subplots()
    fig = plt.figure()
    gs = gridspec.GridSpec(5, 1)
    top = plt.subplot(gs[:3])
    resid1 = plt.subplot(gs[3], sharex=top)
    resid2 = plt.subplot(gs[4], sharex=top)
    fig.subplots_adjust(bottom=0.15, left=0.15, hspace=0.0)
    
    top.errorbar(t1, v1-dv1, yerr=v1_err, fmt='r^', label='Primary')
    top.errorbar(t2, v2 - rv2_pred*K1/K2 - dv2, yerr=v2_err*np.exp(lnf/2.), fmt='ko', label='Secondary')
    top.plot(tplot, rv1_plot, 'r-', alpha=0.5)
    top.plot(tplot, rv2_plot, 'k-', alpha=0.5)
    
    resid1.scatter(t1, v1-dv1 - rv1_pred)
    resid1.plot(t1, np.zeros(len(t1)), 'r--')
    resid1.set_ylabel('O-C (rv1)')
    print('RMS scatter on primary = {:.3f} km/s'.format(np.std(v1-dv1-rv1_pred)))
    
    resid2.scatter(t2, v2 - rv2_pred*K1/K2 - dv2 - rv2_pred)
    resid2.plot(t2, np.zeros(len(t2)), 'r--')
    resid2.set_ylabel('O-C (rv2)')
    print('RMS scatter on secondary = {:.3f} km/s'.format(np.std(v2 - rv2_pred*K1/K2 - dv2 - rv2_pred)))

    top.axes.get_xaxis().set_visible(False)
    resid1.axes.get_xaxis().set_visible(False)
    
    resid2.set_xlabel('JD - 2450000')
    top.set_ylabel('RV (km/s)')
    leg = top.legend(loc='best', fancybox=True)
    
    def tick_formatter(x, pos):
        return "{:.0f}".format(x - 2450000)

    MyFormatter = FuncFormatter(tick_formatter)
    top.xaxis.set_major_formatter(MyFormatter)

    return fig, [top, resid1, resid2]



def read_primary_chains():
    # Read in MCMC chains for the primary star parameters
    home = os.environ['HOME']
    fname = '{}/School/Research/McdonaldData/PlanetData/RV_fit/mcmc_samples/psidraa_els.txt'.format(home)
    chain = pd.read_fwf(fname)

    # Unit conversion
    chain['k'] /= 1000.0

    vals = chain[['tperi', 'period', 'ecc', 'k', 'lop']].get_values()
    return vals



def read_data(first=20):
    date, rv1, rv1_err, rv2, rv2_err = np.loadtxt('rv_data.npy')

    return date, rv1, rv1_err, date[first:], rv2[first:], rv2_err[first:]


def do_partial_fit(N=100):
    prim_pars = read_primary_chains()
    t1, rv1, rv1_err, t2, rv2, rv2_err = read_data()
    
    # Sample a subset of the chain parameters, since each one takes a little while
    sample_list = []
    for i, idx in enumerate(np.random.randint(0, prim_pars.shape[0], N)):
        print('\n{}/{}: Fitting mass-ratio for primary orbit parameters: '.format(i+1, N))
        print(prim_pars[idx])
        samp = fit_partial(*prim_pars[idx], t=t2, rv2=rv2, rv2_err=rv2_err)
        sample_list.append(samp)

    return np.concatenate(sample_list)


if __name__ == '__main__':
    t1, rv1, rv1_err, t2, rv2, rv2_err = read_data()

