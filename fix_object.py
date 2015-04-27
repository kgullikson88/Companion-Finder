from astropy.io import fits
import sys

for fname in sys.argv[1:]:
    print(fname)
    hdulist = fits.open(fname, mode='update')
    if 'psi' in hdulist[0].header['OBJECT'].lower():
        hdulist[0].header['OBJECT'] = 'psi1 Dra A'
        hdulist.flush()
    hdulist.close()