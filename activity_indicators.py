"""
@author: Pia Cortes-Zuleta <pia.cortes@lam.fr>
Laboratoire d'Astropysique de Marseille

Description: Functions to compute stellar activity indicators from a given CCF.
List of indicators: BIS, Vspan, BiGauss, Vasy.

Last updated: 18/01/2021
    """
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import bisect
from tools import *


def getBIS(rv,ccf, plus=False, minus=False):

    if plus == False and minus == False:
        level = [0.1, 0.4, 0.6, 0.9]

    elif plus == True and minus == False:
        level = [0.1, 0.2, 0.8, 0.9]

    elif plus == False and minus == True:
        level = [0.25, 0.45, 0.55, 0.75]

    else:
        print("Option not supported. Proceding to compute standard BIS.")
        level = [0.1, 0.4, 0.6, 0.9]

    depth, bisector_line, wdth = bisector(rv,ccf)
    topBisector = bisector_line[(depth>level[2]) & (depth<level[3])]

    vTop = np.mean(topBisector)*1000
    bottomBisector = bisector_line[(depth>level[0]) & (depth<level[1])]
    vBottom = np.mean(bottomBisector)*1000

    BIS = vTop - vBottom

    return BIS

def getVSpan(rv, ccf):

    rv = np.array(rv)
    ccf = np.array(ccf)
    #rv = np.array(self.data['velocity']).copy()
    #ccf = np.array(self.data['CCF']).copy()

    # Min of the CCF is good proxy of the real RV
    imin = int(np.argmin(ccf))

    # Fit gaussian to obtain RV center and sigma of CCF
    pInit , cov = curve_fit(gauss, rv, ccf,
                    p0=[1.0,imin,2.0, np.max(ccf)])
    rv_center = pInit[1]

    # The 0.5 factor was included to obtain consistent results with BIS
    # In Boisse et al. 2011 the factor is 1
    sigma = 0.5*pInit[2]

    # Define upper and lower part of the CCF and its corresponding RV array
    upperCCF = ccf[(rv <= rv_center-sigma) | (rv >= rv_center+sigma)]
    upperRV = rv[(rv <= rv_center-sigma) | (rv >= rv_center+sigma)]

    lowerCCF = ccf[(rv <= rv_center-3*sigma) | ((rv_center-sigma < rv)
                & (rv < rv_center+sigma)) | (rv >= rv_center+3*sigma)]
    lowerRV = rv[(rv <= rv_center-3*sigma) | ((rv_center-sigma < rv) &
                (rv < rv_center+sigma)) | (rv >= rv_center+3*sigma)]

    # New denser RV array
    fakevel = np.arange(rv[0],rv[-1], 0.00001)

    # Fit gaussian profile in both upper and lower CCFs
    pUpper, cov = curve_fit(gauss, upperRV, upperCCF,
                p0=[1.0,rv_center, sigma, np.max(ccf)])

    pLower, cov = curve_fit(gauss, lowerRV, lowerCCF,
                p0=[1.0,rv_center, sigma,np.max(ccf)])

    # Get RV of previous fits
    fitCCFLower = gauss(fakevel, pLower[0], pLower[1], pLower[2], pLower[3])
    vLow = fakevel[np.argmin(fitCCFLower)]*1000

    fitCCFUpper = gauss(fakevel, pUpper[0], pUpper[1], pUpper[2], pUpper[3])
    vHigh = fakevel[np.argmin(fitCCFUpper)]*1000

    # VSpan = VHigh - VLow
    return (vHigh-vLow)

def getBiGauss(rv, ccf):

    rv = np.array(rv)
    ccf = np.array(ccf)

    fakeVel = np.arange(rv[0], rv[-1], 0.00001)

    pBiGauss, cov = curve_fit(biGauss, rv, ccf,
                    p0=[(np.max(ccf)-np.min(ccf)), self.rv, self.sigma,np.max(ccf), 0.1])

    fitBiGauss = biGauss(fakeVel, pBiGauss[0], pBiGauss[1], pBiGauss[2], pBiGauss[3], pBiGauss[4])
    rvBiGauss = fakeVel[np.argmin(fitBiGauss)]*1000

    pGauss , cov = curve_fit(gauss, rv, ccf,
                    p0=[(np.max(ccf)-np.min(ccf)), self.rv, self.sigma,np.max(ccf)])

    fitGauss = gauss(fakeVel, pGauss[0], pGauss[1], pGauss[2], pGauss[3])
    rvGauss = fakeVel[np.argmin(fitGauss)]*1000

    deltaV = rvBiGauss - rvGauss

    return deltaV
