# Stellar activity indicators for SPIRou
Python codes and scripts to obtain stellar activity indicators in SPIRou data. To use this code, it requires the tcorr files from the APERO reduction, from where we will get the CCF. These files should be in the same directory. The output is a CSV file with the activity indicators values and other information from the header, as the RV corrected, FWHM of the CCF, signal to noise ratio, airmass, exposure time and julian date of the observations.

Currently working indicators:
* BIS (Queloz et al. 2001)
* VSpan (Boisse et al. 2010)
* BiGauss (Figueira et al. 2013).

## How to use it
```python
from activity-spirou import *

path = '/net/GSP/users/pcortes/DATA/GL205/*/*tcorr*_m2_weighted_rv_10_AB.fits'
get_activity_ind(star='Gl205', mask='M2_weighted_RV_10', path_files=path, doplot=True)     
```
