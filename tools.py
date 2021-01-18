"""
@author: Pia Cortes-Zuleta <pia.cortes@lam.fr>
Laboratoire d'Astropysique de Marseille

Description: Tools needed to compute stellar activity indicators.
This includes getRV, bisector, gauss and bigauss profiles.

Last updated: 18/01/2021
    """
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline

def getRV(rv,ccf):
    """ Computes RV using a gauss profile fitting on a given CCF

    Parameters:
    rv (array): array of radial velocities
    ccf (array): cross-correlation function (CCF)

    Return:
    float: RV value

     """

    # Min of CCF
    imin = int(np.argmin(ccf))

    # Good RV guess
    rv_p0 = rv[imin]

    # Define blue and red wings of the CCF where the derivative changes sign
    width_blue = imin - np.max(np.where(np.gradient(ccf[0:imin])>0))
    width_red = np.min(np.where(np.gradient(ccf[imin:])<0))

    width = int(np.min([width_blue, width_red]))

    if width < 20: # Some cases the width is very short
        width = int(np.max([width_blue, width_red]))
    #ccf = ccf[imin-width_blue:imin+width_red]

    # Normalize CCF between 0 and 1 (continium in 1)
    ccf -= np.min(ccf)
    ccf /= np.min( ccf[ [imin - width,imin + width] ])

    ccf = ccf[imin-width_blue : imin+width_red]
    rv = rv[imin-width_blue : imin+width_red]

    ## Get RV from gaussian profile
    pInit , cov = curve_fit(gauss, rv, ccf,
                            p0=[-(np.max(ccf)-np.min(ccf)),rv_p0,1.0, np.max(ccf)])
    rv_target = pInit[1]
    return rv_target

def gauss(x,amp, x0,sigma, cont):
    """ Obtain inverted Gaussian profile of a given set of parameters """
    y = - amp * np.exp(-0.5*(x-x0)**2/sigma**2) +cont
    return y

def biGauss(x,amp,x0,sigma,cont,A):
    """ Computes BiGaussian profile from Figueira et al. 2013

    Parameters:
    x (array): array of values to compute the function
    amp (float): amplitude of the Gaussian
    x0 (float): center of the Gaussian
    sigma (float): sigma of the Gaussian
    continium (float): continium
    A (float): asymmetry parameter applied in the Gaussian's sigma

    Return:
    array: BiGaussian profile

    """
    return np.piecewise(x,
                        [x<cent, x>=cent],
                         [lambda x: amp * np.exp(-0.5 * ((x-x0)/(sigma*(1-A)))**2) + cont,
                          lambda x: amp * np.exp(-0.5 * ((x-x0)/(sigma*(1+A)))**2) + cont])


def bisector(rv, ccf, low_high_cut = 0.1):
    """ Computes the bisector line of a function of the depth of a given CCF.
        Based on Etienne Artigau's (UdM) version.

    Parameters:
    rv (array): array of radial velocities
    ccf (array): cross-correlation function (CCF)
    low_high_cut (float): limit value in the depth of the CCF

    Returns:
    array: depth
    array: bisector
    """
    # get minima
    imin = int(np.argmin(ccf))

    # get point where the derivative changes sign at the edge of the line
    # blue and red wings of the CCF
    width_blue =  imin - np.max(np.where(np.gradient(ccf[0:imin])>0))
    width_red = np.min(np.where(np.gradient(ccf[imin:])<0))

    # get the width from the side of the center that reaches
    # that point first
    width = int(np.min([width_blue, width_red]))

    # Some cases the width is very short because of low-quality CCFs
    # Let's keep the max instead
    if width < 20:
        width = int(np.max([width_blue, width_red]))

    # set depth to zero
    ccf -= np.min(ccf)

    # set continuum to one
    ccf /= np.min( ccf[ [imin - width,imin + width] ])

    # interpolate each side of the ccf slope at a range of depths
    depth = np.arange(low_high_cut,1-low_high_cut,0.001)

    # blue and red side of line
    g1 = (ccf[imin:imin - width:-1]>low_high_cut) & (ccf[imin:imin - width:-1]<(1-low_high_cut))
    spline1 = InterpolatedUnivariateSpline(ccf[imin:imin - width:-1][g1],rv[imin:imin - width:-1 ][g1], k=2)

    g2 = (ccf[imin : imin + width]>low_high_cut) & (ccf[imin : imin + width]<(1-low_high_cut))
    spline2 = InterpolatedUnivariateSpline(ccf[imin : imin + width][g2],rv[imin : imin + width][g2], k=2)

    # get midpoint
    bisector = (spline2(depth)+spline1(depth))/2

    # get bisector widht
    width_ccf = (spline2(depth)-spline1(depth))

    # define depth in the same way as Perryman, 0 is top, 1 is bottom
    depth = 1-depth

    return depth, bisector
