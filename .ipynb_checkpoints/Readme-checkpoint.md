# Introduction

With an increasing number of installed utility-scale PV plants and a growing need for predictable energy generation, the solar industry has started paying attention to solar forecasting. The reasons behind this are:
1.Solar generation is variable in nature. Cloud cover causes this variability by impeding sunlight from hitting the solar panels.
2.Being able to predict solar output will make the electric grid work better under variable conditions like the variance of prices and costs of othe sources.
Essentially, solar forecasting provides a way for grid operators to predict and balance energy generation and consumption. Assuming the grid operator has a mix of generating assets at their disposal, reliable solar forecasting lets that operator best optimize the way they dispatch their controllable units.

------
# Business understanding of the problem


In the other hand,Forecasts attempt to predict the future, which likely means they’re wrong more often than right. What happens when a solar forecast is wrong? The effects of incorrect solar forecasts really boil down to adding cost to the energy market.From the grid operator perspective, an inaccurate solar forecast means that they need to make up for unpredicted imbalance with shorter-term sources of power. These short-term sources tend to be costlier on a per unit basis, which also means that the extent of total inaccuracy is important. For instance, the total cost to make up a 10% error on a 20 MW and 100 MW plant will be different. This cost can then be passed through from the grid operator to the market participants. With that said, we are trying in this project to build a Timeseries model to will forecast the solar radiation with the maximum accuracy possible based on an available history, which can be translated by aiming to have a model with the minimum RMSE root mean squared errors. Thus the finest model will reduce the loss and the cost of energy streramed in the grid.
### <img src="/images/Solar_radiation.jpg." width="800" align="center"/>

---------
# Exploratory data analysis and preprocessing 


In this project i used datasets issued from Nasa labs exactly from a data access viewer of this website:https://power.larc.nasa.gov/data-access-viewer/, it is user freindly inteface where you enter the coordinates and check also the variables that you are intersted in for your study and then submit the request. through its Earth Science research program that has long supported satellite systems and research it is providing important data to the study of climate and climate processes. These data include long-term climatologically averaged estimates of meteorological quantities and surface solar energy fluxes. The goal of the project is builiding a forecasting model with the maximum accuracy using different machine learning techninques especially times series and extend my research later using sequential algorithmes as LSTM and Neural networks. My project starts investigating the data of 47°23'34.5"N 124°18'47.5"W site from the date 1981/01/01 to 2020/04/15, it's located in Washington state, picking the previous site as a  sarting point was due to the potential of solar radiation over there. The dataset contains 14358 observations and 17 variables, here are the explanation of each of the variable that exists in the table:
- Value for missing model data cannot be computed or out of model availability range: -999
- Parameter(s): 
- PRECTOT MERRA2 1/2x1/2 Precipitation (mm day-1)
- PS MERRA2 1/2x1/2 Surface Pressure (kPa)
- T2M MERRA2 1/2x1/2 Temperature at 2 Meters (C)
- ALLSKY_SFC_SW_DWN SRB/FLASHFlux 1/2x1/2 All Sky Insolation Incident on a Horizontal Surface (kW-hr/m^2/day)
- WS50M MERRA2 1/2x1/2 Wind Speed at 50 Meters (m/s)
- TS MERRA2 1/2x1/2 Earth Skin Temperature (C)
- KT SRB/FLASHFlux 1/2x1/2 Insolation Clearness Index (dimensionless)
- ALLSKY_SFC_LW_DWN SRB/FLASHFlux 1/2x1/2 Downward Thermal Infrared (Longwave) Radiative Flux (kW-hr/m^2/day)
- WS10M MERRA2 1/2x1/2 Wind Speed at 10 Meters (m/s)
- T2M_MAX MERRA2 1/2x1/2 Maximum Temperature at 2 Meters (C)
- T2M_MIN MERRA2 1/2x1/2 Minimum Temperature at 2 Meters (C)
- QV2M MERRA2 1/2x1/2 Specific Humidity at 2 Meters (kg kg-1)
- CLRSKY_SFC_SW_DWN SRB/FLASHFlux 1/2x1/2 Clear Sky Insolation Incident on a Horizontal Surface (kW-hr/m^2/day)
We will focus on 3 variables that are direct measures of radiations: (The explaniationof each is above)
 .ALLSKY_SFC_SW_DWN SRB/FLASHFlux
 .ALLSKY_SFC_LW_DWN SRB/FLASHFlux
 .CLRSKY_SFC_SW_DWN SRB/FLASHFlux

I started the preporocessing by a creating a new feature 'Date' by concatenating the month, day and year features then change the "Date" column to non null datetime type before setting it to the index as a big step towards having a timeseries.By using pandas methode value.counts()we found out that there are many erroneous values that equals -999.00 values. Thus replacing the -999,0 with nan is going be easier for the coming steps of prepocessing. Approching filling the Nan values mostly with forward was based on the naive forecasting methode that will see later that consists of forecasting the futur as the same value of the previous day. We did backward filling for the Nan that are in the end of the data set. So we drop the dates that have same value continuously that was result of backfilling so we don't have a straight line that would represent originally NaN.

---------
# Modeling 

Before we step foreward in the modeling phase we ensure setting up the frequency in daily frequency. Then we decompose the timeseries into trends, seasonality and redsduals as the following graphs illustrate:
### <img src="images/decomposition.png" width="800" align="center"/>
Then we ran adfuller stationarity test it happens that our p value is small so we hat we Can reject the null hypothesis. 
How to train our models?
Generally in machine learning we split the data into train and test in order to see how well our model performs, but time series data is kind of special because it has an ordering. Thus we have to write a split function that maintains this ordering while taking a number of ordered observations. So we are not splitting our data by random but instead we leave the ordering and just take chunks of data for training and testing.
Generally in machine learning we split the data into train and test in order to see how well our model performs, but time series data is kind of special because it has an ordering. Thus we have to write a split function that maintains this ordering while taking a number of ordered observations. So we are not splitting our data by random but instead we leave the ordering and just take chunks of data for training and testing. We will define the train_test_split fonction. 

## Method 1 - Baseline model 

We will use a walk-forward validation method to evaluate model performance. This means that each time step in the test dataset will be enumerated, a model constructed on history data, and the forecast compared to the expected value. The observation will then be added to the training dataset and the process repeated.
Walk-forward validation is a realistic way to evaluate time series forecast models as one would expect models to be updated as new observations are made available.
Finally, forecasts will be evaluated using root mean squared error or RMSE. The benefit of RMSE is that it penalizes large errors and the scores are in the same units as the forecast values. it yields to an RMSE of 0.44.


## Method 2 – Holt’s Linear Trend method

Holt extended simple exponential smoothing to allow forecasting of data with a trend. It is nothing more than exponential smoothing applied to both level(the average value in the series) and trend. it yield although to a bigger RMSE about 0.75.


## Method 3 – Sarimax Model 

After decomposition it's not hard to notice the yearly seasonality that will force  the S parameters to 365 (yearly). Although i approached the finding of parameters by running a grid search. You can check the whole process in the last part of notebook. Picking the final parametes was based on the model that yields to the lowest value of AIC Akaike Information Criterion: is provided by ARIMA models fitted using statsmodels library. The Akaike information criterion (AIC) is an estimator of the relative quality of statistical models for a given set of data. Given a collection of models for the data, AIC estimates the quality of each model, relative to each of the other models. Thus, AIC provides a means for model selection.

A model that fits the data very well while using lots of features will be assigned a larger AIC score than a model that uses fewer features to achieve the same goodness-of-fit. Therefore, we are interested in finding the model that yields the lowest AIC value. To achieve this, perform following.
<img src="images/Sarimax.png" width="800" align="center"/>


# Navigation
The environment while creating this project is under Jupyterlab Version 1.1.4. This file can be used to create the environment to run the book.
The entire project was done on one notebook. The book name is first_model.ipynb

# Citations
Some code used in this project came from the internet. Here are the links

.(https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecast)

.(https://www.liip.ch/en/blog/time-series-prediction-a-short-comparison-of-best-practices)
.[Deep learning Times series (LSTM)] ( (https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/structured_data/time_series.ipynb#scrollTo=HlhVGzPhmMYI)



# Next steps
. Finishing the LSTM, ven it's going to be challenging for a 365days period, try a hybrid of algorithmes in order to enhance the model.
.Predict Solar radiation for the rest of 49 sites i already extracted from the website.





