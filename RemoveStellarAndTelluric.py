import numpy as np
import matplotlib.pyplot as plt
import os
import FittingUtilities
from scipy.linalg import svd, diagsvd
import HelperFunctions
from astropy.io import fits
from astropy import units, constants
import GenericSearch
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.interpolate import InterpolatedUnivariateSpline as spline, interp1d
import itertools

c = constants.c.cgs.to(units.km/units.s).value
V_TEMPLATE = 12.2  #barycentric velocity of the template observation
# badorders if ignoring I2 region
badorders = [17, 18, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
             35, 36, 37, 41, 42, 43, 44, 46, 49, 50, 53, 54, 55, 56, 57]
#
# badorders if removing I2 spectrum
#badorders = [17, 18, 41, 42, 43, 44, 46, 49, 50, 53, 54, 55, 56, 57]


def lnlike(rv, template, orders):
    diff = 0.0
    for t, o in zip(template, orders):
        o = o.copy()
        #if 500 < np.mean(o.x) < 600:
        #    continue
        o.x *= (1. + rv/c)
        o_fcn = spline(o.x, o.y/o.cont)
        t_fcn = spline(t.x, t.y/t.cont)
        xgrid = np.linspace(max(o.x[0], t.x[0]), min(o.x[-1], t.x[-1]), 2048)

        diff = np.sum((t_fcn(xgrid) - o_fcn(xgrid))**2)

        #diff += np.sum((t.y/t.cont - fcn(t.x*(1.+rv/c)))**2)
        #diff += np.sum((t.y/t.cont - fcn(t.x))**2)

        #if round(rv, 2) == 0.63:
        #    plt.figure(2)
        #    plt.plot(t.x, tmp, 'k-', alpha=0.6)
        #    plt.plot(t.x, dat, 'r-', alpha=0.2)
        #    plt.figure(1)

    #print(rv, diff)
    return diff

def fit_rv_shift_old(template, orders, guess=0):
    #for t, o in zip(template, orders):
    #    plt.plot(t.x, t.y/t.cont, 'k-', alpha=0.4)
    #    plt.plot(o.x, o.y/o.cont, 'r-', alpha=0.5)
    out = minimize_scalar(lnlike, bracket=(-40, 40), bounds=(-40, 40), args=(template, orders), method='bounded')
    return out.x

def fit_rv_shift(template, orders):
    template_lines = []
    order_lines = []
    for t, o in zip(template, orders):
        plt.plot(t.x, t.y/t.cont, 'k-', alpha=0.4)
        plt.plot(o.x, o.y/o.cont, 'r-', alpha=0.5)

        tlines = FittingUtilities.FindLines(t, tol=0.8)
        olines = FittingUtilities.FindLines(o, tol=0.8)
        if len(tlines) == 0 or len(tlines) != len(olines):
            continue
        template_lines.append(t.x[tlines])
        order_lines.append(o.x[olines])
    template_lines = np.hstack(template_lines)
    order_lines = np.hstack(order_lines)
    rv = (np.median(order_lines/template_lines) - 1.0)*c
    #plt.scatter(template_lines, order_lines)
    #plt.plot(plt.xlim(), plt.xlim(), 'r--')
    plt.scatter(template_lines, order_lines - template_lines*(1.+rv/c))
    plt.show()
    return rv


def make_matrices():
    # Find all the relevant files
    #allfiles = [f for f in os.listdir("./") if f.startswith("RV") and f.endswith("-0.fits")]
    allfiles = [f for f in os.listdir("./") if f.startswith("RV") and '-0' in f and f.endswith("corrected.fits")]

    # Read in the measured RV data
    bjd, rv = np.loadtxt("psi1draa_100_120_mcomb1.dat", usecols=(0,1), unpack=True)
    bjd2, vbary_arr = np.loadtxt("psi1draa_100p_28_37_ASW.dat", usecols=(0,5), unpack=True)

    # Put all the data in one giant list
    alldata = []
    print "Reading all data"
    for i, fname in enumerate(allfiles):
        header = fits.getheader(fname)
        jd = header['jd']
        #vbary = GenericSearch.HelCorr(header, observatory="McDonald")
        idx = np.argmin(abs(bjd2 - jd))
        bjd2_i = bjd2[idx]
        vbary = vbary_arr[idx] / 1000.0
        idx = np.argmin(abs(bjd - jd))
        bjd_i = bjd[idx]
        vstar = rv[idx]/1e3
        print(fname, vbary, vstar)
        vel = vbary - vstar  #Closest... but still not great
        #vel = -vbary + vstar  #NO
        #vel = vbary + vstar  #NO
        #vel = -vbary - vstar  #NO
        #vel = vstar  #NO
        #vel = -vstar #NO
        #vel = vbary  #Surprisingly not bad...
        orders = HelperFunctions.ReadExtensionFits(fname)[:-2]
        for j in badorders[::-1]:
            orders.pop(j)
        if i == 0:
            template = []
            for j, order in enumerate(orders):
                order.x *= (1.0+vel/c)
                orders[j] = order.copy()
                template.append(order.copy())
                alldata.append([order])
            #template = [o.copy() for o in orders]
        else:
            vel = fit_rv_shift_old(template, orders)
            print('RV adjustment = {}'.format(vel))
            for j, order in enumerate(orders):
                order.x *= (1.0+vel/c)
                orders[j] = order.copy()
                if i == 0:
                    alldata.append([order])
                    #template.append(order.copy())
                else:
                    alldata[j].append(order)
                    #plt.plot(order.x, order.y/order.cont, 'g-', alpha=0.5)
            #plt.show()


    # Interpolate each order to a single wavelength grid
    print "Interpolating to wavelength grid"
    xgrids = []
    c_cycler = itertools.cycle(('r', 'g', 'b', 'k'))
    ls_cycler = itertools.cycle(('-', '--', ':', '-.'))
    for i, order in enumerate(alldata):
        print "Order ", i+1
        firstwaves = [d.x[0] for d in order]
        lastwaves = [d.x[-1] for d in order]
        size = np.mean([d.size() for d in order])
        xgrid = np.linspace(max(firstwaves), min(lastwaves), size)
        for j, data in enumerate(order):
            tmp = FittingUtilities.RebinData(data, xgrid)
            order[j] = tmp.y/tmp.cont

            #if i == 20:
            #    col = c_cycler.next()
            #    if j%4 == 0:
            #        ls = ls_cycler.next()
            #    plt.plot(xgrid, order[j], color=col, ls=ls, alpha=0.8, label=allfiles[j])
            #plt.plot(xgrid, order[j], 'k-', alpha=0.2)
        alldata[i] = np.array(order)
        xgrids.append(xgrid)
    #plt.legend(loc='best')
    #plt.show()

    return allfiles, xgrids, alldata




def remove_constants(matrix, n=5):
    U, W, V_t = svd(matrix, full_matrices=True)
    W[:n] = 0.0
    M = W.shape[0]
    N = V_t.shape[1]
    S = diagsvd(W, M, N)

    return np.dot( np.dot(U, S), V_t)

def remove_constants_median(matrix):
    median = np.median(matrix, axis=0)
    return matrix - median



def output_spectra(allfiles, xgrids, matrices):
    for i, fname in enumerate(allfiles):
        numorders = len(matrices)
        outfilename = "{}_smoothed.fits".format(fname[:-5])
        columns = []
        for j in range(numorders):
            column = {"wavelength": xgrids[j],
                      "flux": matrices[j][i] + 10.0,
                      "continuum": 10.0*np.ones(xgrids[j].size),
                      "error": np.ones(xgrids[j].size)}
            columns.append(column)
        print "Outputting to ", outfilename
        HelperFunctions.OutputFitsFileExtensions(columns, fname, outfilename, mode="new")




if __name__ == "__main__":
    fnames, xgrids, matrices = make_matrices()
    plt.imshow(matrices[10], aspect='auto')
    plt.xlim((250, 400))
    plt.savefig('Original_tellcorr.pdf')
    plt.xlim((0, 2000))
    plt.show()
    for i, matrix in enumerate(matrices):
        print "Order ", i+1

        m = remove_constants(matrix, n=5)
        #m = remove_constants_median(matrix)

        sigma = m.std(axis=0)
        matrices[i] = m/sigma

    plt.imshow(matrices[10], aspect='auto')
    plt.savefig('Corrected_tellcorr.pdf')
    plt.show()
    output_spectra(fnames, xgrids, matrices)


