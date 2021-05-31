![image](https://github.com/patternizer/glosat-ben-nevis-reanalysis/blob/master/ben-nevis-850hPa-observations-18981101-18981231.png)
![image](https://github.com/patternizer/glosat-ben-nevis-reanalysis/blob/master/ben-nevis-correlations-0900.png)

# glosat-ben-nevis-reanalysis

Analyis of sub-daily temperature observations and 20CRv3 temperatures at surface and at 850 hPa during 1883-1904 at the Ben Nevis Observatory and Fort William to study the potential for using reananalysis at pressure altitude as a proxy to guide homogenisation of instrumental measurements. Part of land surface air temperature homogenisation efforts and ongoing work for the [GloSAT](https://www.glosat.org) project: www.glosat.org. 

## Contents

* `glosat-ben-nevis-reanalysis.py` - python script to read in sub-daily observations at the hourly, 3-hourly and daily timescale as well as 3-hourly 20CRv3 temperatures at surface and at the 850 hPa pressure level and to perform a comparison using OLS linear regression. The 850 hPa pressure level corresponds to the pressure altitude of the observatory that was manned for twenty years during the period 1883-1904 and which was situated on the top of Ben Nevis whose altitude is 1345m above sea level. 

The accompanying repo [glosat-station-pressure-altitude](https://github.com/patternizer/glosat-station-pressure-altitude) deduces the nearest 20CRv3 pressure level for a given altitude.

A selection of events described in the book 'Twenty years on Ben Nevis' by William T. Kilgour are plotted in finer temporal resolution.

## Instructions for use

The first step is to clone the latest glosat-ben-nevis-reanalysis code and step into the installed Github directory: 

    $ git clone https://github.com/patternizer/glosat-ben-nevis-reanalysis.git
    $ cd glosat-ben-nevis-reanalysis

Then create a DATA/ directory and copy to it the required datasets listed in glosat-ben-nevis-reanalysis.py. You can use the code in the repo [glosat-station-pressure-altitude](https://github.com/patternizer/glosat-20CRv3-t2m) to help extract the 20CRv3 timeseries you will need for the gridcell containing Ben Nevis and Fort William.

### Using Standard Python

The code is designed to run in an environment using Miniconda3-latest-Linux-x86_64.

    $ python glosat-ben-nevis-reanalysis.py

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)

