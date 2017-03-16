## Description
This is the master repository for MSAN 621 group project - King County house sales. The repository includes our codes and notebook for data processing, model construction, and validation.

## Dataset
King County House Sales (https://www.kaggle.com/harlfoxem/housesalesprediction)
This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015. The dataset comprises of more than 21,000 data points and 19 features along with the target - home sales price. Features of the data include basic sales information such as sales date, home built date, living space, number of bedrooms and bathrooms, property quality features such as condition and views, as well as geospatial information such as zipcode and longitude/latitude.

## Team
Hannah Leiber, Valentin Vrzheshch, Matthew Wang, Ruixuan Zhang

## Objective
The objective is to predict the home sales price given the aforementioned features. Since no testing set is provided for this playground dataset, we will randomly select 20,000 records as our training set and the remaining as the test set.

## Model Description
The team explored with various regression models including multiple regressions, support vector machines, K-nearest-neighbor regressors, random forest regressors, extremely randomized trees, gradient boosted trees, and multilayer proceptrons. The models provided varied successes, with the bagging and boosting models achieving the highest scores.  

We aimed to further improve the prediction accuracy by combining the individual models using an ensemble. Ensemble approaches use a number of weak learners and combine their predictive power to construct a stronger combined model. In our case, we decided to implement the ensemble by stacking, or blending with a ridge regression as our combiner model. Our final model is a linear combination of gradient-boosted trees, the extra tree algorithm, k-nearest neighbors, and the multilayer perceptron (neural networks). 

## Results
The ensemble model produced a test error rate of about 11.8%. The error rate is measured as the ratio of the error to the actual sale price.
