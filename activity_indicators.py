"""
@author: Pia Cortes-Zuleta <pia.cortes@lam.fr>
Laboratoire d'Astropysique de Marseille

Description: Functions to compute stellar activity indicators from a given CCF.
List of indicators: BIS, Vspan, BiGauss, Vasy.

Last updated: 30/08/2021
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

    depth, bisector_line = bisector(rv,ccf)
    topBisector = bisector_line[(depth>level[2]) & (depth<level[3])]

    vTop = np.mean(topBisector)*1000
    vTop_err = np.sqrt(np.sum((topBisector-vTop)**2)/(np.len(topBisector)-1))
    bottomBisector = bisector_line[(depth>level[0]) & (depth<level[1])]
    vBottom = np.mean(bottomBisector)*1000
    vBottom_err = np.sqrt(np.sum((bottomBisector-vBottom)**2)/(len(topBisector)-1))

    BIS = vTop - vBottom
    BIS_err = np.sqrt(vTop_err**2+vBottom_err**2)
    return BIS, BIS_err

def getVSpan(rv, ccf):

    rv = rv[~np.isnan(ccf)]
    ccf = ccf[~np.isnan(ccf)]
    #rv = np.array(self.data['velocity']).copy()
    #ccf = np.array(self.data['CCF']).copy()

    # Min of the CCF is good proxy of the real RV
    imin = int(np.argmin(ccf))
    rv_0 = rv[imin]
    # Fit gaussian to obtain RV center and sigma of CCF
    pInit , cov = curve_fit(gauss, rv, ccf,
                    p0=[1.0,rv_0,2.0, np.max(ccf)])
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
    #fakevel = np.arange(rv[0],rv[-1], 0.0001)

    # Fit gaussian profile in both upper and lower CCFs
    pUpper, covUpper = curve_fit(gauss, upperRV, upperCCF,
                p0=[1.0,rv_center, sigma, np.max(ccf)], method='trf')
    pUpper_err = np.sqrt(np.diag(covUpper))

    pLower, covLower = curve_fit(gauss, lowerRV, lowerCCF,
                p0=[1.0,rv_center, sigma,np.max(ccf)],method='trf')
    pLower_err = np.sqrt(np.diag(covLower))

    # Get RV of previous fits
    #fitCCFLower = gauss(fakevel, pLower[0], pLower[1], pLower[2], pLower[3])
    #vLow = fakevel[np.argmin(fitCCFLower)]*1000

    #fitCCFUpper = gauss(fakevel, pUpper[0], pUpper[1], pUpper[2], pUpper[3])
    #vHigh = fakevel[np.argmin(fitCCFUpper)]*1000

    vspan = (pUpper[1]-pLower[1])*1000
    vspan_err = np.sqrt(pUpper_err[1]**2+pLower_err[1]**2)*1000

    return vspan, vspan_err
    #return (vHigh-vLow)

def getBiGauss(rv, ccf):

    rv = np.array(rv)
    ccf = np.array(ccf)

    imin = int(np.argmin(ccf))
    rv_0 = rv[imin]
    #fakeVel = np.arange(rv[0], rv[-1], 0.00001)

    pBiGauss, covBiGauss = curve_fit(biGauss, rv, ccf,
                    p0=[(np.max(ccf)-np.min(ccf)), rv_0, 2.0, np.max(ccf), 0.1],
                    method = 'trf')
    pBiGauss_err = np.sqrt(np.diag(covBiGauss))

    #fitBiGauss = biGauss(fakeVel, pBiGauss[0], pBiGauss[1], pBiGauss[2], pBiGauss[3], pBiGauss[4])
    #rvBiGauss = fakeVel[np.argmin(fitBiGauss)]*1000

    pGauss , covGauss = curve_fit(gauss, rv, ccf,
                    p0=[(np.max(ccf)-np.min(ccf)),rv_0,2.0,np.max(ccf)],
                    method = 'trf')
    pGauss_err = np.sqrt(np.diag(covGauss))

    #fitGauss = gauss(fakeVel, pGauss[0], pGauss[1], pGauss[2], pGauss[3])
    #rvGauss = fakeVel[np.argmin(fitGauss)]*1000

    deltaV = (pBiGauss[1] - pGauss[1])*1000
    deltaV_err = np.sqrt(pBiGauss_err[1]**2 + pGauss_err[1]**2)*1000
    #deltaV = rvBiGauss - rvGauss

    return deltaV, deltaV_err

def getBISpol(path_ccf):
    level = [0.2, 0.4, 0.55, 0.8]

    path_files = glob.glob(path_ccf)
    path_files.sort()
    od = np.zeros(len(path_files))
    mjdmid = np.zeros(len(path_files))
    for i in range(len(path_files)):
        file = pf.open(path_files[i])
        od[i] = file[1].header['EXPNUM']
        mjdmid[i] = file[1].header['MJDMID']
    a=0
    c=0
    BIS_pol = []
    BIS_err_pol = []
    mjdmid_pol = []
    print(od)
    for c in range(100):
        print(c)
        i = c*4 + a
        try:
            if (od[i+1] - od[i] == 1) & (od[i+2] - od[i+1] == 1) & (od[i+3] - od[i+2] == 1):
                mjdmid_pol.append(np.mean(mjdmid[i:i+3]))
                file1 = pf.open(path_files[i])
                file2 = pf.open(path_files[i+1])
                file3 = pf.open(path_files[i+2])
                file4 = pf.open(path_files[i+3])
                depth, bis1 = bisector(file1[1].data['RV'], file1[1].data['COMBINED'])
                depth, bis2 = bisector(file2[1].data['RV'], file2[1].data['COMBINED'])
                depth, bis3 = bisector(file3[1].data['RV'], file3[1].data['COMBINED'])
                depth, bis4 = bisector(file4[1].data['RV'], file4[1].data['COMBINED'])
                #bis1 = bis1 - np.median(bis1)#file1[1].header['RV_OBJ']
                #bis2 = bis2 - file2[1].header['RV_OBJ']
                #bis3 = bis3 - file3[1].header['RV_OBJ']
                #bis4 = bis4 - file4[1].header['RV_OBJ']
                err = np.zeros(len(depth))
                bis_line = np.zeros(len(depth))

                for d in range(len(depth)):
                    #err[d] = np.std([bis1[d],bis2[d],bis3[d],bis4[d]])
                    x = np.array([bis1[d],bis2[d],bis3[d],bis4[d]])*1000
                    bis_line[d] = np.mean(x)
                    err[d] = np.sqrt(np.sum((x-bis_line[d])**2)/3)

                topBisector = bis_line[(depth>level[2]) & (depth<level[3])]
                topBisector_err = err[(depth>level[2]) & (depth<level[3])]

                vTop = np.mean(topBisector)
                #vTop_err = np.sqrt(np.sum(topBisector_err**2))
                vTop_err = np.sqrt(np.sum((topBisector-vTop)**2)/3.)

                bottomBisector = bis_line[(depth>level[0]) & (depth<level[1])]
                bottomBisector_err = err[(depth>level[0]) & (depth<level[1])]


                vBottom = np.mean(bottomBisector)
                #vBottom_err = np.sqrt(np.sum(bottomBisector_err**2))
                vBottom_err = np.sqrt(np.sum((bottomBisector-vBottom)**2)/3.)

                BIS = vTop - vBottom
                BIS_err = np.sqrt(vTop_err**2+vBottom_err**2)

                BIS_pol.append(BIS)
                BIS_err_pol.append(BIS_err)
                #print(table.iloc[i:i+4])
                #print(np.mean(table.iloc[i:i+4]))

            elif (od[i+1] - od[i] == 1) & (od[i+2] - od[i+1] == 1):
                mjdmid_pol.append(np.mean(mjdmid[i:i+2]))
                file1 = pf.open(path_files[i])
                file2 = pf.open(path_files[i+1])
                file3 = pf.open(path_files[i+2])
                depth, bis1 = bisector(file1[1].data['RV'], file1[1].data['COMBINED'])
                depth, bis2 = bisector(file2[1].data['RV'], file2[1].data['COMBINED'])
                depth, bis3 = bisector(file3[1].data['RV'], file3[1].data['COMBINED'])
                #bis1 = bis1 - file1[1].header['RV_OBJ']
                #bis2 = bis2 - file2[1].header['RV_OBJ']
                #bis3 = bis3 - file3[1].header['RV_OBJ']
                err = np.zeros(len(depth))
                bis_line = np.zeros(len(depth))

                for d in range(len(depth)):
                    #err[d] = np.std([bis1[d],bis2[d],bis3[d]])
                    x = np.array([bis1[d],bis2[d],bis3[d]])*1000
                    bis_line[d] = np.mean(x)
                    err[d] = np.sqrt(np.sum((x-bis_line[d])**2)/2)
                    #bis_line[d] = np.mean([bis1[d],bis2[d],bis3[d]])

                topBisector = bis_line[(depth>level[2]) & (depth<level[3])]
                topBisector_err = err[(depth>level[2]) & (depth<level[3])]

                vTop = np.mean(topBisector)
                #vTop_err = np.sqrt(np.sum(topBisector_err**2))
                vTop_err = np.sqrt(np.sum((topBisector-vTop)**2)/2.)

                bottomBisector = bis_line[(depth>level[0]) & (depth<level[1])]
                bottomBisector_err = err[(depth>level[0]) & (depth<level[1])]

                vBottom = np.mean(bottomBisector)
                #vBottom_err = np.sqrt(np.sum(bottomBisector_err**2))
                vBottom_err = np.sqrt(np.sum((bottomBisector-vBottom)**2)/2.)

                BIS = vTop - vBottom
                BIS_err = np.sqrt(vTop_err**2+vBottom_err**2)

                BIS_pol.append(BIS)
                BIS_err_pol.append(BIS_err)

                a = a-1
            elif (od[i+1] - od[i] == 1):
                mjdmid_pol.append(np.mean(mjdmid[i:i+1]))
                file1 = pf.open(path_files[i])
                file2 = pf.open(path_files[i+1])
                depth, bis1 = bisector(file1[1].data['RV'], file1[1].data['COMBINED'])
                depth, bis2 = bisector(file2[1].data['RV'], file2[1].data['COMBINED'])
                #bis1 = bis1 - file1[1].header['RV_OBJ']
                #bis2 = bis2 - file2[1].header['RV_OBJ']
                err = np.zeros(len(depth))
                bis_line = np.zeros(len(depth))
                for d in range(len(depth)):
                    #err[d] = np.std([bis1[d],bis2[d]])
                    #bis_line[d] = np.mean([bis1[d],bis2[d]])
                    x = np.array([bis1[d],bis2[d]])*1000
                    bis_line[d] = np.mean(x)
                    err[d] = np.sqrt(np.sum((x-bis_line[d])**2)/1)

                topBisector = bis_line[(depth>level[2]) & (depth<level[3])]
                topBisector_err = err[(depth>level[2]) & (depth<level[3])]

                vTop = np.mean(topBisector)
                #vTop_err = np.sqrt(np.sum(topBisector_err**2))
                vTop_err = np.sqrt(np.sum((topBisector-vTop)**2))

                bottomBisector = bis_line[(depth>level[0]) & (depth<level[1])]
                bottomBisector_err = err[(depth>level[0]) & (depth<level[1])]

                vBottom = np.mean(bottomBisector)
                #vBottom_err = np.sqrt(np.sum(bottomBisector_err**2))
                vBottom_err = np.sqrt(np.sum((bottomBisector-vBottom)**2))

                BIS = vTop - vBottom
                BIS_err = np.sqrt(vTop_err**2+vBottom_err**2)

                BIS_pol.append(BIS)
                BIS_err_pol.append(BIS_err)

                a = a-2
            else:
                a = a-3

        except Exception as e:
            print(e)
            #handleException(e)
            #raise
            #sys.exit(1)
    return mjdmid_pol, BIS_pol, BIS_err_pol
