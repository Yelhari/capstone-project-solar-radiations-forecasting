# Intro
With an increasing number of installed utility-scale PV plants and a growing need for predictable energy generation, the solar industry has started paying attention to solar forecasting. The reasons behind this are:
1.Solar generation is variable in nature.Cloud cover causes this variability by impeding sunlight from hitting the solar panels.
2.Being able to predict solar output will make the electric grid work better under variable conditions like the variance of prices and costs of othe sources
Essentially, solar forecasting provides a way for grid operators to predict and balance energy generation and consumption. Assuming the grid operator has a mix of generating assets at their disposal, reliable solar forecasting lets that operator best optimize the way they dispatch their controllable units.
------
**The problem**  


In the other hand,Forecasts attempt to predict the future, which likely means they’re wrong more often than right. What happens when a solar forecast is wrong? The effects of incorrect solar forecasts really boil down to adding cost to the energy market.From the grid operator perspective, an inaccurate solar forecast means that they need to make up for unpredicted imbalance with shorter-term sources of power. These short-term sources tend to be costlier on a per unit basis, which also means that the extent of total inaccuracy is important. For instance, the total cost to make up a 10% error on a 20 MW and 100 MW plant will be different. This cost can then be passed through from the grid operator to the market participants.

IIn this project i used datasets issued from Nasa labs. through its Earth Science research program has long supported satellite systems and research providing data important to the study of climate and climate processes. These data include long-term climatologically averaged estimates of meteorological quantities and surface solar energy fluxes. The goal of the project is builiding a forecasting model with the maximum accuracy using different machine learning techninques.

---------




# Navigation
[PowerPoint slides ] which I used to present this project is in https://github.com/Yelhari/capstone-project-solar-radiations-forecasting

The environment while creating this project is under Jupyterlab Version 1.1.4. This file can be used to create the environment to run the book.

The entire project was done on one notebook. The book name is first_model.ipynb[Project]

In order to re-create analysis each cell can be done sequentially 



# Citations
Some code used in this project came from the internet. Here are the links

[Function to Grid search SARIMA model] (https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecast)

[Forward Step SARIMA model] (https://www.liip.ch/en/blog/time-series-prediction-a-short-comparison-of-best-practices)
[Deep learning Times series (LSTM)] (https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/structured_data/time_series.ipynb#scrollTo=HlhVGzPhmMYI)
# Importing Data and Tools
For the first part of my project i needed to import all of my tools and data I used Pandas and Numpy for Data reading I used Statsmodels and Sklearn to build my models and get metrics.I extracted data from the folowing website:https://power.larc.nasa.gov/data-access-viewer/

# Data Cleaning
For data cleaning I wanted to make sure my data fits the needs required for my process. After looking at my data I had a great understanding of what I needed to do to clean my data steps.

## What needs to be done?
1. Drop columns everything except opening price 
2. Drop potential stocks (If for whatever reason stocks didnt download correctly or have missing data)
3. Delete next day holding rows from dataset
4. Fix shape of my dataset
5. Normalize data(Later in Neural network techniques)

You can learn more about my process in the data cleaning section of my [Capstone notebook]https://github.com/Yelhari/capstone-project-solar-radiations-forecasting

# Data methodology 
NASA, through its Earth Science research program has long supported satellite systems and research providing data important to the study of climate and climate processes. These data include long-term climatologically averaged estimates of meteorological quantities and surface solar energy fluxes. Additionally, mean daily values of the based meteorological and solar data are provided in a time series format. These satellite and model-based products have been shown to be accurate enough to provide reliable solar and meteorological resource data over regions where surface measurements are sparse or nonexistent. The products offer two unique features – the data is global and, in general, contiguous in time. These two important characteristics, however, tend to generate very large data archives which can be intimidating for users, particularly those with little experience or resources to explore these large data sets. Moreover, the data products contained in the various NASA archives are often in formats that present challenges to new users. To foster the usage of the global solar and meteorological data, NASA’s Earth Science Division Applied Sciences Program supported, and continues to support, the development of user-friendly data sets formulated specifically for designated user communities and access to these data via a user friendly web based mapping portal.

The Surface meteorology and Solar Energy (SSE) project is one of the earlier activities funded by the Applied Science Program to foster use of NASA’s data holdings. The SSE data-delivery website is focused on providing easy access to parameters valued in the renewable energy industry (e.g. solar and wind energy) and was initially released in 1997. The solar and meteorological data contained in this first release was based on the 1993 NASA/World Climate Research Program Version 1.1 Surface Radiation Budget (SRB) science data and TIROS Operational Vertical Sounder (TOVS) data from the International Satellite Cloud Climatology Project (ISCCP). Release 2 of SSE was made public in 1999 with parameters specifically tailored to the needs of the renewable energy community. Subsequent releases of SSE - SSE-Release 3.0 in 2000, SSE-Release 4.0 in 2003, SSE-Release 5.0 in 2005, and SSE-Release 6.0 in 2008 – have continued to build upon an interactive dialog with potential customers resulting in updated parameters using the most recent NASA data as well as inclusion of new parameters that have been requested by the user community.

The POWER project was initiated in 2003 as an outgrowth of the Surface meteorology and Solar Energy project. The initial POWER project encompassed the SSE component and added two new datasets with applicability to the architectural (e.g. Sustainable Buildings) and agricultural (e.g. Agro-climatology) industries, with the continuing objective of improvements to and expansion of the focused parameters included in each section of POWER.

Recent upgrades to the SSE component of POWER were initiated to include Geographic Information System (GIS) functionality as an option to the data ordering/access process. SSE-GIS constituted the Release 7.0 version, but did provide updated data sets. The POWER Release-8 encompasses the three focused data components of POWER, SSE, Sustainable Building, and Agroclimatology, in a new responsive data portal built on upgrades to the underlying based meteorological data, and is designed to fit on desktop, tablet and smart phone platforms, and adds geospatially enabled online tools to facilitate data ordering and viewing as well as analysis of the solar and meteorological data.

The meteorological data/parameters in POWER Release-8 are based upon a single assimilation model from Goddard’s Global Modeling and Assimilation Office (GMAO). The updated meteorological data are derived from the GMAO Modern Era Retrospective-Analysis for Research and Applications (MERRA-2) assimilation model products and GMAO Forward Processing – Instrument Teams (FP-IT) GEOS 5.12.4 near-real time products. The MERRA-2 data spans the time period from 1981 to within several months of real time; the GEOS 5.12.4 data span the time period from the end of the MERRA-2 data stream to within several days of real time. The MERRA-2 and GEOS 5.12.4 versions are essentially the same and thus discontinuities that are often apparent between different assimilation models are minimized.

The solar based data/parameters in POWER Release-8 will continue to be based upon satellite observations with subsequent inversion to surface solar insolation by NASA’s Global Energy and Water Exchange Project /Surface Radiation Budget (SRB) and NASA’s Fast Longwave And SHortwave Radiative project (FLASHFlux).

The data/parameters in POWER Release-8 are provided on a global grid with a spatial resolution of 0.5° latitude by 0.5° longitude. Thus, this POWER Release 8.0 consolidates the SSE, Sustainable Building, and Agroclimatology components in a single data portal with updated and low latency solar and meteorological data products, and provides GIS compatible data formats and GIS-enabled web applications.

The purpose of this documentation is to describe the underlying solar and meteorological data sources, to provide estimates of the accuracy associated with the underlying data and resulting parameters, and to enumerate the data/parameters in each component of POWER Release-8.