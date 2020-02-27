# SVR script
# Created by Benjamin L Deck
#Last updated: 1.12.2020
#Support Vector Machine: Regression
#We are still concerned with classification issues. In typical SVM we know #that the only data that matters are the support vectors in terms of fitting #the model, we don't care about the data lying beyond the epsilon margin. #Similarly, SVR is also only concerned with a subset of the training data.

#Three different types of support vector regression: SVR, NuSVR and #linearSVR. Linear SVR is a faster implementation but assumes the #relationship between the support vectors is linear and thus only uses #linear kernel. NuSVR uses the parameter nu to control for the number of #support vectors, nu in NuSVR replaces the epsilon in regular SVR/SVM-- nu #is an upper bound on the fraction of training errors and lower bound on the #fraction of support vectors, e.g. if interval set is 0,1 the split is 0.5 #upper and 0.5 lower

#Similar to all other SVM models we need to classify a target vector. The #model will also accept X,y vectors. However, in SVR, the y or target are #float numbers

# This function anticipates an input of two csv files with a row per participant. Each column being mean connectivity. Additionally, The matrix reduction should be performed before inputing features and targets into this function.


# import all of the necessary modules and utilities
def conn_svr(features, targets, type_kernel, cost, epsi):
    """  SVR script Created by Benjamin L Deck
        We are still concerned with classification issues. In typical SVM we know that the only data that matters are the support vectors in terms of fitting
        the model, we don't care about the data lying beyond the epsilon margin. Three different types of support vector regression: SVR, NuSVR and linearSVR.
        Linear SVR is a faster implementation but assumes the relationship between the support vectors is linear and thus only uses a linear kernel. NuSVR uses the parameter nu to control for the number of support vectors, nu in NuSVR replaces the epsilon in regular SVR/SVM-- nu is an upper bound on the fraction of training errors and lower bound on the fraction of support vectors, e.g. if interval set is 0,1 the split is 0.5 upper and 0.5 lower

        Similar to most other machine learning paradigms in Sci-kit learn, we need to classify a target vector. The model will also accept X,y vectors. However, in SVR, the y or target are continuous.

        This function anticipates an input of two csv files with a row per participant. Each column being mean connectivity. Additionally, The matrix reduction should be performed before inputing features and targets into this function.


        features -- a csv of all features
        targets -- a csv of all targets
        typ_kernel -- a string denoting the type of kernel to apply as a hyper-parameter (i.e. 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
        cost -- the cost parameter (30 is recommended by Zhang et al., 2014; DeMarco et al., 2018)
        epsi -- the epsilon value the area surrounding the hyperplane in which there is no penalty to the r-squared value (float) (recommended is 0.1)
    """

    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn import preprocessing
    from sklearn.svm import SVR
    from sklearn.model_selection import permutation_test_score
    from sklearn.utils import shuffle
    from sklearn.metrics import accuracy_score
    import timeit
    import time



    # Turning all features/targets into float64, print all feature/target types
    #df_features = features.drop(features.index[[0]])
    df_features = features.astype(float)
    print(df_features.dtypes)
    df_targets = targets.astype(float)


    # convert features pandas dataframe to numpy array
    X = df_features.to_numpy()
    X # print the features to examine

    # convert the target vector to numpy array
    y = df_targets.to_numpy()
    y[1:20] # print the first 20 entries of array

    ## Create the training and test sets (this step is optional) if conducting permutation testing this step is not necessary
    #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=20)
    #print(X_train.shape)
    #print(X_test.shape)
    #print(y_train.shape)
    #print(y_test.shape)


    ## Feature Scaling
    # Here we will standardize the scale of each feature set. We do this becasue the scale or measurement units for each feature will not be uniform and thus we may end up with data points that should not necessarily be combined. We may have variables which vary much more than another. In this instance we may have a variable such as GRE score which has a much higher range and std compared with research or Letters of recommendation (LOR). In order to account for this we need to conduct feature scaling which will normalize and standardize all the measured features and center them around 0. This decreases the chance that one variable which has high variability inadvertently acccounts for a disproportionate amount of the variance in the model. For an example see [here](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html).  Also if interested in the understanding why the scaler doesn't get the scaled mean or your features to exactly '0' then read [this](https://stackoverflow.com/questions/40405803/mean-of-data-scaled-with-sklearn-standardscaler-is-not-zero).
    #

    X_scaled= preprocessing.scale(X)
    print(X_scaled)
    print(X_scaled.mean()) # this will most likely not be exactly zero as there is a limit to how close our computational abilities can get us to 'absolute zero'.
    print(X_scaled.std())
    scaler = preprocessing.StandardScaler().fit(X_scaled)
    print(scaler)
    print(scaler.mean_)
    scaler.transform(X_scaled)


    y_scaled=preprocessing.scale(y)
    print(y_scaled)
    print(y_scaled.mean()) # this will most likely not be exactly zero as there is a limit to how close our computational abilities can get us to 'absolute zero'.
    print(y_scaled.std())


    #If not using permutation testing do this next step where feature scaling will be applied to both test and train data
    # scaler= preprocessing.StandardScaler().fit(X_train)
    # print(scaler)
    # print(scaler.mean_)
    # scaler.transform(X_train)
    # scaler.transform(X_test)


    ## Creating the Hypothesis driven SVR model
    svr = SVR(kernel= type_kernel, epsilon = epsi, C = cost) # this creates an SVR model using the linear function. C = 30 given by Zhang et al., 2014 and DeMarco + Turkeltaub 2018. Epsilon is the margin of error around the hyperplane where the model is not penalized for missclassified data points. C = cost which is the penalty for missclassification where a higher cost encourages more support vector point making a more complex fix to the data.

    svr.fit(X_scaled, y_scaled.ravel())
    #svr.predict(y_test) # use this if training and testing split models

    pred_svr = svr.predict(X_scaled)

    # Obtain the r-squared value for the overall model.
    hyp_score = svr.score(X_scaled, y_scaled)
    hyp_coef= svr.coef_
    hyp_params = svr.get_params(deep=True)




    ###### Permutation testing
    #Here we need to shuffle (permute) the target vector (behavior/clinical scale)

    perm_svr = SVR(kernel= type_kernel,epsilon = epsi, C = cost)
    beta_arr = np.array([]) # beta weights as they are returned by fitting the model
    count = 0
    r_squared_arr = np.array([])
    weights = np.array([])

    #a while loop which first initializes the count at 0. Then the while loop is:


    start = time.time()


    for ii in range(10000):
        y_scaled = shuffle(y_scaled) # Will randomly permute the y-target vector
        perm_svr.fit(X_scaled, y_scaled.ravel()) # fits the randomly permuted target vector to the feature set (connections between and within each region)
        beta_arr = np.append(beta_arr, perm_svr.coef_) # appends the beta weights of each feature (i.e. connections between and within each region)
        r_squared_arr = np.append(r_squared_arr, perm_svr.score(X_scaled, y_scaled)) # appends the r^2 value for each model which will create the null distribution
        count = count + 1 # iterative counter which if ii = >=10000 it will kill the loop.


    end = time.time()
    print(end-start)





    plt.hist(r_squared_arr, bins=500);
    plt.axvline(np.median(r_squared_arr), color='k', linestyle='dashed', linewidth=2)
    plt.axvline(hyp_score, color='green', linestyle='solid', linewidth=2)
    plt.show()

    print(hyp_score)


    # Here we need to calculate the p-value for the comparison between the null distribution and the hypothesized model (the original target model that we ran).
    num_mods = count + 1
    #print(num_mods) # total number of models run including the hypothesized model
    num_obs_btr_than_hyp = 0

    for ii in r_squared_arr: # this for loop will give you the number of r^2 values which are greater than your hypothesized model
        if ii > hyp_score:
            num_obs_btr_than_hyp = num_obs_btr_than_hyp + 1
        print(num_obs_btr_than_hyp)

    num_obs_btr_than_hyp = num_obs_btr_than_hyp + 1 # Adding the original model to the int list to calculate the p-value

    p_value = num_obs_btr_than_hyp/num_mods # this division calculates the probability that hypotheized score is significantly different from the null distribution.

    return(hyp_params)
    return(hyp_score)
    return(hyp_coef)

    print('Hypothesized model:', str(hyp_score))
    print('p-value for model:', str(p_value))
    print('number of models better than hypothesized model:', str(num_obs_btr_than_hyp))
