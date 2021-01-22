"""
@author: Pia Cortes-Zuleta <pia.cortes@lam.fr>
Laboratoire d'Astropysique de Marseille

Description: Routine to compute stellar activity indicators in SPIRou reduced data.
Currently working BIS, VSpan and BiGaussian

Last updated: 18/01/2021
    """
from activity_indicators import *
from tools import *
import glob
from astropy.table import Table
from astropy.io import fits as pf
import matplotlib.pyplot as plt

def get_activity_ind(star, mask, path_files, doplot=False):

    keywords = ['DATE-OBS','MJDMID','BERV','EXTSN035','AIRMASS',
                'EXPTIME', 'CCFMFWHM', 'RV_CORR', 'RV_OBJ']

    files = glob.glob(path_files)
    files.sort()

    tbl = Table()

    tbl['FILES'] = files
    tbl['ODOMETER'] = np.zeros_like(tbl, dtype='U7')
    for i in range(len(tbl)):
        tbl['ODOMETER'][i] = tbl['FILES'][i].split('/')[-1].split('o')[0]

    tbl['RV_GAUSS'] = np.zeros_like(files,dtype = float) # RV from gaussian
    tbl['BIS'] = np.zeros_like(files, dtype=float)
    tbl['VSPAN'] = np.zeros_like(files, dtype=float)
    tbl['BIGAUSS'] = np.zeros_like(files, dtype=float)

    # get CCF and RV arrays
    for i in (range(len(files))):
        ccf_tbl = pf.getdata(files[i])
        ccf_RV = ccf_tbl['RV'] #velocity array
        ccf = ccf_tbl['COMBINED'] #combined CCF

        # Compute RV and indicators
        try:
            tbl['RV_GAUSS'][i] = getRV(ccf_RV, ccf)
        except Exception as e:
            tbl['RV_GAUSS'][i] = np.nan
        try:
            tbl['BIS'][i] = getBIS(ccf_RV, ccf)
        except Exception as e:
            tbl['BIS'][i] = np.nan
        try:
            tbl['VSPAN'][i] = getVSpan(ccf_RV, ccf)
        except Exception as e:
            tbl['VSPAN'][i] = np.nan
        try:
            tbl['BIGAUSS'][i] = getBiGauss(ccf_RV, ccf)
        except Exception as e:
            tbl['BIGAUSS'][i] = np.nan

        # Add keywords values from header
        hdr = pf.getheader(files[i],ext = 1)
        if i ==0:
            # now that we have a first header, we add the relevant columns to the CSV table
            for key in keywords:
                if key in hdr:
                    key_type = type(hdr[key])
                    # if we have a string, we set the table to accept long values (up to 99 characters)
                    if key_type == str:
                        key_type = '<U99'
                else:
                    # keyword not in header, we need to assume something. The safest is string
                    key_type = str

                # add the column to the CSV file
                tbl[key] = np.zeros_like(files,dtype = key_type)

        for key in keywords:
            if key in hdr:
                tbl[key][i] = hdr[key]

    batch_name = '{0}_{1}_activity'.format(star,mask)
    tbl.write('{0}.csv'.format(batch_name),overwrite = True)

    if doplot:
        fig, ax = plt.subplots(5,1, figsize=(15,16))
        ax[0].plot(tbl['MJDMID'], tbl['RV_CORR'], '.', color='black')
        ax[0].set_ylabel('RV_CORR [km/s]')
        ax[1].plot(tbl['MJDMID'], tbl['BIS'], '.', color='black')
        ax[1].set_ylabel('BIS [m/s]')
        ax[2].plot(tbl['MJDMID'], tbl['CCFMFWHM'],'.', color='black')
        ax[2].set_ylabel('CCF FWHM [km/s]')
        ax[3].plot(tbl['MJDMID'], tbl['VSPAN'],'.', color='black')
        ax[3].set_ylabel('VSPAN [m/s]')
        ax[4].plot(tbl['MJDMID'], tbl['BIGAUSS'], '.', color='black')
        ax[4].set_ylabel('BiGaussian [m/s]')
        plt.savefig(star+"_activity_timeseries.png",bbox_inches='tight')
        #plt.show()
