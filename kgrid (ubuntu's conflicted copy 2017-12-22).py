from sklearn import model_selection, linear_model, svm, metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter

import kutil
from jsklearn import binary_model


def gs_np(X, Y, alphas_log=(-3, 3, 7), method = "Lasso", n_splits=5, disp = False):
    """
    Return
    ------
    df, pd.DataFrame
    All results are included in df

    df_avg, pd.DataFrame
    Average results are included in df_avg

    df_best, pd.DataFrame
    The best of the average results is df_best

    Usage
    -----
    df, df_avg, df_best = gs_np( X, Y)
    """
    df_l = list()
    df_avg_l = list()
    for (alpha_idx, alpha) in enumerate( np.logspace(*alphas_log)):
        model = getattr(linear_model, method)(alpha=alpha)
        kf5_c = model_selection.KFold( n_splits=n_splits, shuffle=True)
        kf5 = kf5_c.split( X)

        r2_l = []
        for train, test in kf5:
            model.fit( X[train,:], Y[train,:])
            r2 = model.score( X[test,:], Y[test,:])
            r2_l.append( r2)
            
        # make a dataframe
        df_i = pd.DataFrame()
        df_i["r2"] = r2_l
        df_i["unit"] = range( len( r2_l))
        df_i["method"] = method
        df_i["n_splits"] = n_splits
        df_i["alpha"] = alpha
        df_i["alpha_idx"] = alpha_idx
        df_l.append( df_i)
        
        df_avg_i = pd.DataFrame()
        df_avg_i["E[r2]"] = [np.mean( r2_l)]
        df_avg_i["std(r2)"] = [np.std( r2_l)]
        df_avg_i["method"] = method
        df_avg_i["n_splits"] = n_splits
        df_avg_i["alpha"] = alpha
        df_avg_i["alpha_idx"] = alpha_idx
        df_avg_l.append( df_avg_i)
        if disp:
            print( "alpha=", alpha)
            print( r2_l)
            print('Average, std=', np.mean(r2_l), np.std(r2_l))
            print('-------------')
            
    df = pd.concat( df_l, ignore_index=True) 
    df_avg = pd.concat( df_avg_l, ignore_index=True)
    
    # dataframe for the best
    idx_best = np.argmax( df_avg["E[r2]"].values)
    df_best = df_avg.loc[[idx_best], :].copy()
    
    return df, df_avg, df_best


def gs_numpy( method, X, Y, alphas_log = (-1, 1, 9), n_splits=5, n_jobs = -1, disp = True):
    """
    Grid search method with numpy array of X and Y
    Previously, np.mat are used for compatible with Matlab notation.    
    """
    if disp:
        print( X.shape, Y.shape)

    clf = getattr( linear_model, method)()
    parmas = {'alpha': np.logspace( *alphas_log)}
    kf5_c = model_selection.KFold( n_splits = n_splits, shuffle=True)
    #kf5 = kf5_c.split( X)
    gs = model_selection.GridSearchCV( clf, parmas, scoring = 'r2', cv = kf5_c, n_jobs = n_jobs)

    gs.fit( X, Y)

    return gs


def gs_Lasso( xM, yV, alphas_log = (-1, 1, 9), n_splits=5, n_jobs = -1):

    print(xM.shape, yV.shape)

    clf = linear_model.Lasso()
    #parmas = {'alpha': np.logspace(1, -1, 9)}
    parmas = {'alpha': np.logspace( *alphas_log)}
    kf5_c = model_selection.KFold( n_splits = n_splits, shuffle=True)
    #kf5 = kf5_c.split( xM)
    gs = model_selection.GridSearchCV( clf, parmas, scoring = 'r2', cv = kf5_c, n_jobs = n_jobs)

    gs.fit( xM, yV)

    return gs

def gs_Lasso_norm( xM, yV, alphas_log = (-1, 1, 9)):

    print(xM.shape, yV.shape)

    clf = linear_model.Lasso( normalize = True)
    #parmas = {'alpha': np.logspace(1, -1, 9)}
    parmas = {'alpha': np.logspace( *alphas_log)}
    kf5_c = model_selection.KFold( n_splits = 5, shuffle=True)
    #kf5 = kf5_c.split( xM)
    gs = model_selection.GridSearchCV( clf, parmas, scoring = 'r2', cv = kf5_c, n_jobs = -1)

    gs.fit( xM, yV)

    return gs


def gs_Lasso_kf( xM, yV, alphas_log_l):

    kf5_ext_c = model_selection.KFold( n_splits = 5, shuffle=True)
    kf5_ext = kf5_ext_c.split( xM)

    score_l = []
    for ix, (tr, te) in enumerate( kf5_ext):

        print('{}th fold external validation stage ============================'.format( ix + 1))
        xM_in = xM[ tr, :]
        yV_in = yV[ tr, 0]

        print('First Lasso Stage')
        gs1 = gs_Lasso( xM_in, yV_in, alphas_log_l[0])
        print('Best score:', gs1.best_score_)
        print('Best param:', gs1.best_params_)
        print(gs1.grid_scores_)

        nz_idx = gs1.best_estimator_.sparse_coef_.indices
        xM_in_nz = xM_in[ :, nz_idx]

        print('Second Lasso Stage')
        gs2 = gs_Lasso( xM_in_nz, yV_in, alphas_log_l[1])
        print('Best score:', gs2.best_score_)
        print('Best param:', gs2.best_params_)
        print(gs2.grid_scores_)

        print('External Validation Stage')
        xM_out = xM[ te, :]
        yV_out = yV[ te, 0]
        xM_out_nz = xM_out[:, nz_idx]
        score = gs2.score( xM_out_nz, yV_out)

        print(score)        
        score_l.append( score)

        print('')

    print('all scores:', score_l)
    print('average scores:', np.mean( score_l))

    return score_l


def gs_Lasso_kf_ext( xM, yV, alphas_log_l):

    kf5_ext_c = model_selection.KFold( n_splits = 5, shuffle=True)
    kf5_ext = kf5_ext_c.split( xM)

    score_l = []
    for ix, (tr, te) in enumerate( kf5_ext):

        print('{}th fold external validation stage ============================'.format( ix + 1))
        xM_in = xM[ tr, :]
        yV_in = yV[ tr, 0]

        print('First Lasso Stage')
        gs1 = gs_Lasso( xM_in, yV_in, alphas_log_l[0])
        print('Best score:', gs1.best_score_)
        print('Best param:', gs1.best_params_)
        print(gs1.grid_scores_)


        nz_idx = gs1.best_estimator_.sparse_coef_.indices
        xM_in_nz = xM_in[ :, nz_idx]

        print('Second Lasso Stage')
        gs2 = gs_Lasso( xM_in_nz, yV_in, alphas_log_l[1])
        print('Best score:', gs2.best_score_)
        print('Best param:', gs2.best_params_)
        print(gs2.grid_scores_)

        print('External Validation Stage')
        # Obtain prediction model by whole data including internal validation data
        alpha = gs2.best_params_['alpha']
        clf = linear_model.Lasso( alpha = alpha)
        clf.fit( xM_in_nz, yV_in)

        xM_out = xM[ te, :]
        yV_out = yV[ te, 0]
        xM_out_nz = xM_out[:, nz_idx]
        score = clf.score( xM_out_nz, yV_out)

        print(score)        
        score_l.append( score)

        print('')

    print('all scores:', score_l)
    print('average scores:', np.mean( score_l))

    return score_l


def gs_Ridge_Asupervising_2fp( xM1, xM2, yV, s_l, alpha_l):
    """
    This 2fp case uses two fingerprints at the same in order to 
    combines their preprocessing versions separately. 
    """
    r2_l2 = list()  
    for alpha in alpha_l:
        print(alpha)
        r2_l = cv_Ridge_Asupervising_2fp( xM1, xM2, yV, s_l, alpha)
        r2_l2.append( r2_l)
    return r2_l2


def _cv_LinearRegression_r0( xM, yV):

    print(xM.shape, yV.shape)

    clf = linear_model.Ridge()
    kf5_c = model_selection.KFold( n_splits = 5, shuffle=True)
    kf5 = kf5_c.split( xM)
    cv_scores = model_selection.cross_val_score( clf, xM, yV, scoring = 'r2', cv = kf5, n_jobs = -1)

    return cv_scores


def _cv_LinearRegression_r1( xM, yV):

    print(xM.shape, yV.shape)

    clf = linear_model.LinearRegression()
    kf5_c = model_selection.KFold( n_splits = 5, shuffle=True)
    kf5 = kf5_c.split( xM)
    cv_scores = model_selection.cross_val_score( clf, xM, yV, scoring = 'r2', cv = kf5, n_jobs = -1)

    print('R^2 mean, std -->', np.mean( cv_scores), np.std( cv_scores))

    return cv_scores


def _cv_LinearRegression_r2( xM, yV, scoring = 'r2'):

    print(xM.shape, yV.shape)

    clf = linear_model.LinearRegression()
    kf5_c = model_selection.KFold( n_splits = 5, shuffle=True)
    kf5 = kf5_c.split( xM)
    cv_scores = model_selection.cross_val_score( clf, xM, yV, scoring = scoring, cv = kf5, n_jobs = -1)

    print('{}: mean, std -->'.format( scoring), np.mean( cv_scores), np.std( cv_scores))

    return cv_scores


def cv_LinearRegression( xM, yV, n_splits = 5, scoring = 'median_absolute_error', disp = False):
    """
    metrics.explained_variance_score(y_true, y_pred)    Explained variance regression score function
    metrics.mean_absolute_error(y_true, y_pred) Mean absolute error regression loss
    metrics.mean_squared_error(y_true, y_pred[, ...])   Mean squared error regression loss
    metrics.median_absolute_error(y_true, y_pred)   Median absolute error regression loss
    metrics.r2_score(y_true, y_pred[, ...]) R^2 (coefficient of determination) regression score function.
    """  
    
    if disp:
        print(xM.shape, yV.shape)

    clf = linear_model.LinearRegression()
    kf5_c = model_selection.KFold( n_splits=n_splits, shuffle=True)
    kf5 = kf5_c.split( xM)  
    cv_score_l = list()
    for train, test in kf5:
        # clf.fit( xM[train,:], yV[train,:])
        # yV is vector but not a metrix here. Hence, it should be treated as a vector
        clf.fit( xM[train,:], yV[train])
        
        yVp_test = clf.predict( xM[test,:])
        if scoring == 'median_absolute_error':
            cv_score_l.append( metrics.median_absolute_error(yV[test], yVp_test))
        else:
            raise ValueError( "{} scoring is not supported.".format( scoring))

    if disp: # Now only this flag is on, the output will be displayed. 
        print('{}: mean, std -->'.format( scoring), np.mean( cv_score_l), np.std( cv_score_l))

    return cv_score_l


def cv_LinearRegression_ci( xM, yV, n_splits = 5, scoring = 'median_absolute_error', disp = False):
    """
    metrics.explained_variance_score(y_true, y_pred)    Explained variance regression score function
    metrics.mean_absolute_error(y_true, y_pred) Mean absolute error regression loss
    metrics.mean_squared_error(y_true, y_pred[, ...])   Mean squared error regression loss
    metrics.median_absolute_error(y_true, y_pred)   Median absolute error regression loss
    metrics.r2_score(y_true, y_pred[, ...]) R^2 (coefficient of determination) regression score function.
    """  
    
    if disp:
        print(xM.shape, yV.shape)

    clf = linear_model.LinearRegression()
    kf5_c = model_selection.KFold( n_splits=n_splits, shuffle=True)
    kf5 = kf5_c.split( xM)  
    cv_score_l = list()
    ci_l = list()
    for train, test in kf5:
        # clf.fit( xM[train,:], yV[train,:])
        # yV is vector but not a metrix here. Hence, it should be treated as a vector
        clf.fit( xM[train,:], yV[train])
        
        yVp_test = clf.predict( xM[test,:])
        
        # Additionally, coef_ and intercept_ are stored. 
        ci_l.append( (clf.coef_, clf.intercept_))
        if scoring == 'median_absolute_error':
            cv_score_l.append( metrics.median_absolute_error(yV[test], yVp_test))
        else:
            raise ValueError( "{} scoring is not supported.".format( scoring))

    if disp: # Now only this flag is on, the output will be displayed. 
        print('{}: mean, std -->'.format( scoring), np.mean( cv_score_l), np.std( cv_score_l))

    return cv_score_l, ci_l


def cv_LinearRegression_ci_pred( xM, yV, n_splits = 5, scoring = 'median_absolute_error', disp = False):
    """
    metrics.explained_variance_score(y_true, y_pred)    Explained variance regression score function
    metrics.mean_absolute_error(y_true, y_pred) Mean absolute error regression loss
    metrics.mean_squared_error(y_true, y_pred[, ...])   Mean squared error regression loss
    metrics.median_absolute_error(y_true, y_pred)   Median absolute error regression loss
    metrics.r2_score(y_true, y_pred[, ...]) R^2 (coefficient of determination) regression score function.
    """  
    
    if disp:
        print(xM.shape, yV.shape)

    clf = linear_model.LinearRegression()
    kf5_c = model_selection.KFold( n_splits=n_splits, shuffle=True)
    kf5 = kf5_c.split( xM)  
    cv_score_l = list()
    ci_l = list()
    yVp = yV.copy() 
    for train, test in kf5:
        # clf.fit( xM[train,:], yV[train,:])
        # yV is vector but not a metrix here. Hence, it should be treated as a vector
        clf.fit( xM[train,:], yV[train])
        
        yVp_test = clf.predict( xM[test,:])
        yVp[test] = yVp_test
        
        # Additionally, coef_ and intercept_ are stored. 
        coef = np.array(clf.coef_).tolist()
        intercept = np.array(clf.intercept_).tolist()
        ci_l.append( (clf.coef_, clf.intercept_))
        if scoring == 'median_absolute_error':
            cv_score_l.append( metrics.median_absolute_error(yV[test], yVp_test))
        else:
            raise ValueError( "{} scoring is not supported.".format( scoring))

    if disp: # Now only this flag is on, the output will be displayed. 
        print('{}: mean, std -->'.format( scoring), np.mean( cv_score_l), np.std( cv_score_l))

    return cv_score_l, ci_l, yVp.A1.tolist()


def cv_LinearRegression_ci_pred_full_Ridge( xM, yV, alpha, n_splits = 5, shuffle=True, disp = False):
    """
    Note - scoring is not used. I may used later. Not it is remained for compatibility purpose.
    metrics.explained_variance_score(y_true, y_pred)    Explained variance regression score function
    metrics.mean_absolute_error(y_true, y_pred) Mean absolute error regression loss
    metrics.mean_squared_error(y_true, y_pred[, ...])   Mean squared error regression loss
    metrics.median_absolute_error(y_true, y_pred)   Median absolute error regression loss
    metrics.r2_score(y_true, y_pred[, ...]) R^2 (coefficient of determination) regression score function.
    """  
    
    if disp:
        print(xM.shape, yV.shape)

    # print( 'alpha of Ridge is', alpha)
    clf = linear_model.Ridge( alpha)
    kf5_c = model_selection.KFold( n_splits=n_splits, shuffle=shuffle)
    kf5 = kf5_c.split( xM)
    
    cv_score_l = list()
    ci_l = list()
    yVp = yV.copy() 
    for train, test in kf5:
        # clf.fit( xM[train,:], yV[train,:])
        # yV is vector but not a metrix here. Hence, it should be treated as a vector
        clf.fit( xM[train,:], yV[train])
        
        yVp_test = clf.predict( xM[test,:])
        yVp[test] = yVp_test
        
        # Additionally, coef_ and intercept_ are stored.        
        ci_l.append( (clf.coef_, clf.intercept_))
        y_a = np.array( yV[test])[:,0]
        yp_a = np.array( yVp_test)[:,0]
        cv_score_l.extend( np.abs(y_a - yp_a).tolist())

    return cv_score_l, ci_l, yVp.A1.tolist()


def cv_LinearRegression_ci_pred_full( xM, yV, n_splits = 5, shuffle=True, disp = False):
    """
    Note - scoring is not used. I may used later. Not it is remained for compatibility purpose.
    metrics.explained_variance_score(y_true, y_pred)    Explained variance regression score function
    metrics.mean_absolute_error(y_true, y_pred) Mean absolute error regression loss
    metrics.mean_squared_error(y_true, y_pred[, ...])   Mean squared error regression loss
    metrics.median_absolute_error(y_true, y_pred)   Median absolute error regression loss
    metrics.r2_score(y_true, y_pred[, ...]) R^2 (coefficient of determination) regression score function.
    """  
    
    if disp:
        print(xM.shape, yV.shape)

    clf = linear_model.LinearRegression()
    kf5_c = model_selection.KFold( n_splits=n_splits, shuffle=shuffle)
    kf5 = kf5_c.split( xM)
    
    cv_score_l = list()
    ci_l = list()
    yVp = yV.copy() 
    for train, test in kf5:
        # clf.fit( xM[train,:], yV[train,:])
        # yV is vector but not a metrix here. Hence, it should be treated as a vector
        clf.fit( xM[train,:], yV[train])
        
        yVp_test = clf.predict( xM[test,:])
        yVp[test] = yVp_test
        
        # Additionally, coef_ and intercept_ are stored.        
        ci_l.append( (clf.coef_, clf.intercept_))
        y_a = np.array( yV[test])[:,0]
        yp_a = np.array( yVp_test)[:,0]
        cv_score_l.extend( np.abs(y_a - yp_a).tolist())

    return cv_score_l, ci_l, yVp.A1.tolist()


def cv_LinearRegression_It( xM, yV, n_splits = 5, scoring = 'median_absolute_error', N_it = 10, disp = False, ldisp = False):
    """
    N_it times iteration is performed for cross_validation in order to make further average effect. 
    The flag of 'disp' is truned off so each iteration will not shown.  
    """
    cv_score_le = list()
    for ni in range( N_it):
        cv_score_l = cv_LinearRegression( xM, yV, n_splits = n_splits, scoring = scoring, disp = disp)
        cv_score_le.extend( cv_score_l)
        
    o_d = {'mean': np.mean( cv_score_le),
           'std': np.std( cv_score_le),
           'list': cv_score_le}
    
    if disp or ldisp:
        print('{0}: mean(+/-std) --> {1}(+/-{2})'.format( scoring, o_d['mean'], o_d['std']))
        
    return o_d


def cv_LinearRegression_ci_It( xM, yV, n_splits = 5, scoring = 'median_absolute_error', N_it = 10, disp = False, ldisp = False):
    """
    N_it times iteration is performed for cross_validation in order to make further average effect. 
    The flag of 'disp' is truned off so each iteration will not shown.  
    """
    cv_score_le = list()
    ci_le = list()
    for ni in range( N_it):
        cv_score_l, ci_l = cv_LinearRegression_ci( xM, yV, n_splits = n_splits, scoring = scoring, disp = disp)
        cv_score_le.extend( cv_score_l)
        ci_le.extend( ci_l)
        
    o_d = {'mean': np.mean( cv_score_le),
           'std': np.std( cv_score_le),
           'list': cv_score_le,
           'ci': ci_le}
    
    if disp or ldisp:
        print('{0}: mean(+/-std) --> {1}(+/-{2})'.format( scoring, o_d['mean'], o_d['std']))
        
    return o_d


def cv_LinearRegression_ci_pred_It( xM, yV, n_splits = 5, scoring = 'median_absolute_error', N_it = 10, disp = False, ldisp = False):
    """
    N_it times iteration is performed for cross_validation in order to make further average effect. 
    The flag of 'disp' is truned off so each iteration will not shown.  
    """
    cv_score_le = list()
    ci_le = list()
    yVp_ltype_l = list() # yVp_ltype is list type of yVp not matrix type
    for ni in range( N_it):
        cv_score_l, ci_l, yVp_ltype = cv_LinearRegression_ci_pred( xM, yV, n_splits = n_splits, scoring = scoring, disp = disp)
        cv_score_le.extend( cv_score_l)
        ci_le.extend( ci_l)
        yVp_ltype_l.append( yVp_ltype)
        
    o_d = {'mean': np.mean( cv_score_le),
           'std': np.std( cv_score_le),
           'list': cv_score_le,
           'ci': ci_le,
           'yVp': yVp_ltype_l}
    
    if disp or ldisp:
        print('{0}: mean(+/-std) --> {1}(+/-{2})'.format( scoring, o_d['mean'], o_d['std']))
        
    return o_d


def cv_LOO(xM, yV, disp = False, ldisp=False):
    """
    This is a specialized function for LOO cross_validation. 
    """
    n_splits = xM.shape[0]  # for LOO CV
    return cv_LinearRegression_ci_pred_full_It(xM, yV, n_splits=n_splits, N_it=1,
                                               shuffle=False, disp=disp, ldisp= disp)

def cv_LOO_mode(mode, xM, yV, disp = False, ldisp = False):
    """
    This is a specialized function for LOO cross_validation. 
    """

    if mode == "Linear":
        # Linear regression
        return cv_LOO( xM = xM, yV = yV, disp = disp, ldisp = ldisp)
    elif mode == "Bias":
        return cv_LinearRegression_Bias( xM, yV)
    elif mode == "None":
        return cv_LinearRegression_None( xM, yV)

    raise ValueError("Mode is not support: mode =", mode)


def cv_LOO_Ridge( xM, yV, alpha, disp = False, ldisp = False):
    """
    This is a specialized function for LOO cross_validation. 
    """
    n_splits = xM.shape[0] # for LOO CV
    return cv_LinearRegression_ci_pred_full_It_Ridge( xM, yV, alpha, n_splits = n_splits, N_it = 1, 
                                    shuffle = False, disp = disp, ldisp = ldisp)



def cv_LinearRegression_ci_pred_full_It_Ridge( xM, yV, alpha, n_splits = 5, N_it = 10, 
                                    shuffle = True, disp = False, ldisp = False):
    """
    N_it times iteration is performed for cross_validation in order to make further average effect. 
    The flag of 'disp' is truned off so each iteration will not shown.  
    """
    cv_score_le = list()
    ci_le = list()
    yVp_ltype_l = list() # yVp_ltype is list type of yVp not matrix type
    for ni in range( N_it):
        cv_score_l, ci_l, yVp_ltype = cv_LinearRegression_ci_pred_full_Ridge( xM, yV, alpha,
                                    n_splits = n_splits, shuffle = shuffle, disp = disp)
        cv_score_le.extend( cv_score_l)
        ci_le.extend( ci_l)
        yVp_ltype_l.append( yVp_ltype)

    # List is not used if N_it is one
    if N_it == 1:
        yVp_ltype_l = yVp_ltype_l[0]
        
    o_d = {'median_abs_err': np.median( cv_score_le),
           'mean_abs_err': np.mean( cv_score_le),
           'std_abs_err': np.std( cv_score_le),
           'list': cv_score_le,
           'ci': ci_le,
           'yVp': yVp_ltype_l}
    
    return o_d


def cv_LinearRegression_ci_pred_full_It( xM, yV, n_splits = 5, N_it = 10, 
                                    shuffle = True, disp = False, ldisp = False):
    """
    N_it times iteration is performed for cross_validation in order to make further average effect. 
    The flag of 'disp' is truned off so each iteration will not shown.  
    """
    cv_score_le = list()
    ci_le = list()
    yVp_ltype_l = list() # yVp_ltype is list type of yVp not matrix type
    for ni in range( N_it):
        cv_score_l, ci_l, yVp_ltype = cv_LinearRegression_ci_pred_full( xM, yV, 
                                    n_splits = n_splits, shuffle = shuffle, disp = disp)
        cv_score_le.extend( cv_score_l)
        ci_le.extend( ci_l)
        yVp_ltype_l.append( yVp_ltype)

    # List is not used if N_it is one
    if N_it == 1:
        yVp_ltype_l = yVp_ltype_l[0]
        
    o_d = {'median_abs_err': np.median( cv_score_le),
           'mean_abs_err': np.mean( cv_score_le),
           'std_abs_err': np.std( cv_score_le),
           'list': cv_score_le,
           'ci': ci_le,
           'yVp': yVp_ltype_l}
    
    return o_d


def cv_LinearRegression_None( xM, yV):
    """
    N_it times iteration is performed for cross_validation in order to make further average effect. 
    The flag of 'disp' is truned off so each iteration will not shown.  
    """
    #print( "cv_LinearRegression_None", xM.shape, yV.shape)
    X, y = np.array( xM)[:,0], np.array( yV)[:,0]

    # only 1-dim is allowed for both X and y
    assert (X.ndim == 1) or (X.shape[2] == 1) and (yV.ndim == 1) or (yV.shape[2] == 1)

    yP = X
    cv_score_le = np.abs( np.array( y - yP)).tolist()
        
    o_d = {'median_abs_err': np.median( cv_score_le),
           'mean_abs_err': np.mean( cv_score_le),
           'std_abs_err': np.std( cv_score_le), # this can be std(err)
           'list': cv_score_le,
           'ci': "t.b.d",
           'yVp': X.tolist()}
    
    return o_d


def cv_LinearRegression_Bias( xM, yV):
    """
    N_it times iteration is performed for cross_validation in order to make further average effect. 
    The flag of 'disp' is truned off so each iteration will not shown.  
    """
    #print( "cv_LinearRegression_None", xM.shape, yV.shape)
    X, y = np.array( xM)[:,0], np.array( yV)[:,0]

    # only 1-dim is allowed for both X and y
    assert (X.ndim == 1) or (X.shape[2] == 1) and (yV.ndim == 1) or (yV.shape[2] == 1)

    loo_c = model_selection.LeaveOneOut()
    loo = loo_c.split( X)

    yP = y.copy()
    for train, test in loo:
        bias = np.mean(y[train] - X[train])
        yP[test] = X[test] + bias

    cv_score_le = np.abs( np.array( y - yP)).tolist()
        
    o_d = {'median_abs_err': np.median( cv_score_le),
           'mean_abs_err': np.mean( cv_score_le),
           'std_abs_err': np.std( cv_score_le), # this can be std(err)
           'list': cv_score_le,
           'ci': "t.b.d",
           'yVp': X.tolist()}
    
    return o_d


def mdae_no_regression( xM, yV, disp = False, ldisp = False):
    """
    Median absloute error (Mdae) is calculated without any (linear) regression.
    """
    xM_a = np.array( xM)
    yV_a = np.array( yV)

    ae_l = [ np.abs(x - y) for x, y in zip(xM_a[:,0], yV_a[:, 0])]

    return np.median( ae_l)

def gs_Ridge_Asupervising_2fp_molw( xM1, xM2, yV, s_l, alpha_l):
    """
    This 2fp case uses two fingerprints at the same in order to 
    combines their preprocessing versions separately. 
    """
    r2_l2 = list()  
    for alpha in alpha_l:
        print(alpha)
        r2_l = cv_Ridge_Asupervising_2fp_molw( xM1, xM2, yV, s_l, alpha)
        r2_l2.append( r2_l)
    return r2_l2

def gs_Ridge_Asupervising_molw( xM, yV, s_l, alpha_l):
    r2_l2 = list()  
    for alpha in alpha_l:
        print(alpha)
        r2_l = cv_Ridge_Asupervising_molw( xM, yV, s_l, alpha)
        r2_l2.append( r2_l)
    return r2_l2

def gs_Ridge_Asupervising( xM, yV, s_l, alpha_l):
    r2_l2 = list()  
    for alpha in alpha_l:
        print(alpha)
        r2_l = cv_Ridge_Asupervising( xM, yV, s_l, alpha)
        r2_l2.append( r2_l)
    return r2_l2

def gs_RidgeByLasso_kf_ext( xM, yV, alphas_log_l):

    kf5_ext_c = model_selection.KFold( n_splits = 5, shuffle=True)
    kf5_ext = kf5_ext_c.split( xM)

    score_l = []
    for ix, (tr, te) in enumerate( kf5_ext):

        print('{}th fold external validation stage ============================'.format( ix + 1))
        xM_in = xM[ tr, :]
        yV_in = yV[ tr, 0]

        print('First Ridge Stage')
        gs1 = gs_Lasso( xM_in, yV_in, alphas_log_l[0])
        print('Best score:', gs1.best_score_)
        print('Best param:', gs1.best_params_)
        print(gs1.grid_scores_)


        nz_idx = gs1.best_estimator_.sparse_coef_.indices
        xM_in_nz = xM_in[ :, nz_idx]

        print('Second Lasso Stage')
        gs2 = gs_Ridge( xM_in_nz, yV_in, alphas_log_l[1])
        print('Best score:', gs2.best_score_)
        print('Best param:', gs2.best_params_)
        print(gs2.grid_scores_)

        print('External Validation Stage')
        # Obtain prediction model by whole data including internal validation data
        alpha = gs2.best_params_['alpha']
        clf = linear_model.Ridge( alpha = alpha)
        clf.fit( xM_in_nz, yV_in)

        xM_out = xM[ te, :]
        yV_out = yV[ te, 0]
        xM_out_nz = xM_out[:, nz_idx]
        score = clf.score( xM_out_nz, yV_out)

        print(score)        
        score_l.append( score)

        print('')

    print('all scores:', score_l)
    print('average scores:', np.mean( score_l))

    return score_l

def gs_SVR( xM, yV, svr_params, n_splits = 5, n_jobs = -1):

    print(xM.shape, yV.shape)

    clf = svm.SVR()
    #parmas = {'alpha': np.logspace(1, -1, 9)}
    kf5_c = model_selection.KFold( n_splits=n_splits, shuffle=True)
    #kf5 = kf5_c.split( xM) 
    gs = model_selection.GridSearchCV( clf, svr_params, scoring = 'r2', cv = kf5_c, n_jobs = n_jobs)

    gs.fit( xM, yV.A1)

    return gs

def cv_SVR( xM, yV, svr_params, n_splits = 5, n_jobs = -1, grid_std = None, graph = True, shuffle = True):
    """
    method can be 'Ridge', 'Lasso'
    cross validation is performed so as to generate prediction output for all input molecules
    """ 
    print(xM.shape, yV.shape)

    clf = svm.SVR( **svr_params)
    kf_n_c = model_selection.KFold( n_splits=n_splits, shuffle=shuffle)
    kf_n = kf5_ext_c.split( xM)
    yV_pred = model_selection.cross_val_predict( clf, xM, yV, cv = kf_n, n_jobs = n_jobs)

    if graph:
        print('The prediction output using cross-validation is given by:')
        kutil.cv_show( yV, yV_pred, grid_std = grid_std)

    return yV_pred

def gs_SVC( X, y, params, n_splits = 5, **kwargs):
    print("Use gs_param for the more general implementation!")
    return gs_param( svm.SVC(), X, y, params, n_splits=n_splits, **kwargs)

def gs_param( model, X, y, param_grid, n_splits=5, shuffle=True, n_jobs=-1, graph=False):
    """
    gs = gs_param( model, X, y, param_grid, n_splits=5, shuffle=True, n_jobs=-1)

    Inputs
    ======
    model = svm.SVC(), or linear_model.LinearRegression(), for example
    param = {"C": np.logspace(-2,2,5)}
    """
    #print(xM.shape, yVc.shape)
    kf5_c = model_selection.KFold( n_splits=n_splits, shuffle=shuffle)
    gs = model_selection.GridSearchCV( model, param_grid, cv=kf5_c, n_jobs=n_jobs)
    gs.fit( X, y)

    if graph:
        plt.plot( gs.cv_results_["mean_train_score"], label='E[Train]')
        plt.plot( gs.cv_results_["mean_test_score"], label='E[Test]')
        plt.legend(loc=0)
        plt.grid()

    return gs

def gs_LinearSVC( xM, yVc, params, n_splits=5, n_jobs=-1):
    return gs_param( svm.LinearSVC(), xM, yVc, params, n_splits=n_splits, n_jobs=n_jobs)

def gs_SVRByLasso_kf_ext( xM, yV, alphas_log, svr_params):

    kf5_ext_c = model_selection.KFold( n_splits = 5, shuffle=True)
    kf5_ext = kf5_ext_c.split( xM)

    score_l = []
    for ix, (tr, te) in enumerate( kf5_ext):

        print('{}th fold external validation stage ============================'.format( ix + 1))
        xM_in = xM[ tr, :]
        yV_in = yV[ tr, 0]

        print('First Ridge Stage')
        gs1 = gs_Lasso( xM_in, yV_in, alphas_log)
        print('Best score:', gs1.best_score_)
        print('Best param:', gs1.best_params_)
        print(gs1.grid_scores_)


        nz_idx = gs1.best_estimator_.sparse_coef_.indices
        xM_in_nz = xM_in[ :, nz_idx]

        print('Second Lasso Stage')
        gs2 = gs_SVR( xM_in_nz, yV_in, svr_params)
        print('Best score:', gs2.best_score_)
        print('Best param:', gs2.best_params_)
        print(gs2.grid_scores_)

        print('External Validation Stage')
        # Obtain prediction model by whole data including internal validation data
        C = gs2.best_params_['C']
        gamma = gs2.best_params_['gamma']
        epsilon = gs2.best_params_['epsilon']

        clf = svm.SVR( C = C, gamma = gamma, epsilon = epsilon)
        clf.fit( xM_in_nz, yV_in.A1)

        xM_out = xM[ te, :]
        yV_out = yV[ te, 0]
        xM_out_nz = xM_out[:, nz_idx]
        score = clf.score( xM_out_nz, yV_out.A1)

        print(score)        
        score_l.append( score)

        print('')

    print('all scores:', score_l)
    print('average scores:', np.mean( score_l))

    return score_l  

def gs_SVRByLasso( xM, yV, alphas_log, svr_params):

    kf5_ext_c = model_selection.KFold( n_splits = 5, shuffle=True)
    kf5_ext = kf5_ext_c.split( xM)

    score1_l = []
    score_l = []
    for ix, (tr, te) in enumerate( kf5_ext):

        print('{}th fold external validation stage ============================'.format( ix + 1))
        xM_in = xM[ tr, :]
        yV_in = yV[ tr, 0]

        print('First Ridge Stage')
        gs1 = gs_Lasso( xM_in, yV_in, alphas_log)
        print('Best score:', gs1.best_score_)
        print('Best param:', gs1.best_params_)
        print(gs1.grid_scores_)
        score1_l.append( gs1.best_score_)


        nz_idx = gs1.best_estimator_.sparse_coef_.indices
        xM_in_nz = xM_in[ :, nz_idx]

        print('Second Lasso Stage')
        gs2 = gs_SVR( xM_in_nz, yV_in, svr_params)
        print('Best score:', gs2.best_score_)
        print('Best param:', gs2.best_params_)
        print(gs2.grid_scores_)

        print('External Validation Stage')
        # Obtain prediction model by whole data including internal validation data
        C = gs2.best_params_['C']
        gamma = gs2.best_params_['gamma']
        epsilon = gs2.best_params_['epsilon']

        clf = svm.SVR( C = C, gamma = gamma, epsilon = epsilon)
        clf.fit( xM_in_nz, yV_in.A1)

        xM_out = xM[ te, :]
        yV_out = yV[ te, 0]
        xM_out_nz = xM_out[:, nz_idx]
        score = clf.score( xM_out_nz, yV_out.A1)

        print(score)        
        score_l.append( score)

        print('')

    print('all scores:', score_l)
    print('average scores:', np.mean( score_l))

    print('First stage scores', score1_l)
    print('Average first stage scores', np.mean( score1_l))

    return score_l, score1_l

def gs_ElasticNet( xM, yV, en_params):

    print(xM.shape, yV.shape)

    clf = linear_model.ElasticNet()
    kf5_c = model_selection.KFold( n_splits = 5, shuffle=True)
    kf5 = kf5_c.split( xM)
    gs = model_selection.GridSearchCV( clf, en_params, scoring = 'r2', cv = kf5_c, n_jobs = -1)

    gs.fit( xM, yV)

    return gs

def gs_SVRByElasticNet( xM, yV, en_params, svr_params):

    kf5_ext_c = model_selection.KFold( n_splits = 5, shuffle=True)
    kf5_ext = kf5_ext_c.split( xM)

    score1_l = []
    score_l = []
    for ix, (tr, te) in enumerate( kf5_ext):

        print('{}th fold external validation stage ============================'.format( ix + 1))
        xM_in = xM[ tr, :]
        yV_in = yV[ tr, 0]

        print('First Ridge Stage')
        gs1 = gs_ElasticNet( xM_in, yV_in, en_params)
        print('Best score:', gs1.best_score_)
        print('Best param:', gs1.best_params_)
        print(gs1.grid_scores_)
        score1_l.append( gs1.best_score_)


        nz_idx = gs1.best_estimator_.sparse_coef_.indices
        xM_in_nz = xM_in[ :, nz_idx]

        print('Second Lasso Stage')
        gs2 = gs_SVR( xM_in_nz, yV_in, svr_params)
        print('Best score:', gs2.best_score_)
        print('Best param:', gs2.best_params_)
        print(gs2.grid_scores_)

        print('External Validation Stage')
        # Obtain prediction model by whole data including internal validation data
        C = gs2.best_params_['C']
        gamma = gs2.best_params_['gamma']
        epsilon = gs2.best_params_['epsilon']

        clf = svm.SVR( C = C, gamma = gamma, epsilon = epsilon)
        clf.fit( xM_in_nz, yV_in.A1)

        xM_out = xM[ te, :]
        yV_out = yV[ te, 0]
        xM_out_nz = xM_out[:, nz_idx]
        score = clf.score( xM_out_nz, yV_out.A1)

        print(score)        
        score_l.append( score)

        print('')

    print('all scores:', score_l)
    print('average scores:', np.mean( score_l))

    print('First stage scores', score1_l)
    print('Average first stage scores', np.mean( score1_l))

    return score_l, score1_l

def gs_GPByLasso( xM, yV, alphas_log):

    kf5_ext_c = model_selection.KFold( n_splits = 5, shuffle=True)
    kf5_ext = kf5_ext_c.split( xM)

    score1_l = []
    score_l = []
    for ix, (tr, te) in enumerate( kf5_ext):

        print('{}th fold external validation stage ============================'.format( ix + 1))
        xM_in = xM[ tr, :]
        yV_in = yV[ tr, 0]

        print('First Ridge Stage')
        gs1 = gs_Lasso( xM_in, yV_in, alphas_log)
        print('Best score:', gs1.best_score_)
        print('Best param:', gs1.best_params_)
        print(gs1.grid_scores_)
        score1_l.append( gs1.best_score_)


        nz_idx = gs1.best_estimator_.sparse_coef_.indices
        xM_in_nz = xM_in[ :, nz_idx]

        print('Second GP Stage')        
        Xa_in_nz = np.array( xM_in_nz)
        ya_in = np.array( yV_in)

        xM_out = xM[ te, :]
        yV_out = yV[ te, 0]
        xM_out_nz = xM_out[:, nz_idx]
        Xa_out_nz = np.array( xM_out_nz)
        ya_out = np.array( yV_out)

        #jgp = gp.GaussianProcess( Xa_in_nz, ya_in, Xa_out_nz, ya_out)
        # the y array should be send as [:,0] form to be sent as vector array
        jgp = gp.GaussianProcess( Xa_in_nz, ya_in[:,0], Xa_out_nz, ya_out[:,0])
        jgp.optimize_noise_and_amp()
        jgp.run_gp()

        #ya_out_pred = np.mat(jgp.predicted_targets)
        ya_out_pred = jgp.predicted_targets
        #print ya_out[:,0].shape, jgp.predicted_targets.shape

        r2, rmse = regress_show( ya_out[:,0], ya_out_pred)

        score = r2
        print(score)        
        score_l.append( score)

        print('')

    print('all scores:', score_l)
    print('average scores:', np.mean( score_l))

    print('First stage scores', score1_l)
    print('Average first stage scores', np.mean( score1_l))

    return score_l, score1_l

def show_gs_alpha( grid_scores):
    alphas = np.array([ x[0]['alpha'] for x in grid_scores])
    r2_mean = np.array([ x[1] for x in grid_scores])
    r2_std = np.array([ np.std(x[2]) for x in grid_scores])
    
    r2_mean_pos = r2_mean + r2_std
    r2_mean_neg = r2_mean - r2_std

    plt.semilogx( alphas, r2_mean, 'x-', label = 'E[$r^2$]')
    plt.semilogx( alphas, r2_mean_pos, ':k', label = 'E[$r^2$]+$\sigma$')
    plt.semilogx( alphas, r2_mean_neg, ':k', label = 'E[$r^2$]-$\sigma$')
    plt.grid()
    plt.legend( loc = 2)
    plt.show()

    best_idx = np.argmax( r2_mean)
    best_r2_mean = r2_mean[ best_idx]
    best_r2_std = r2_std[ best_idx]
    best_alpha = alphas[ best_idx]

    print("Best: r2(alpha = {0}) -> mean:{1}, std:{2}".format( best_alpha, best_r2_mean, best_r2_std))

"""
Specialized code for extract results
"""

def gs_Ridge( xM, yV, alphas_log = (1, -1, 9), n_splits = 5, n_jobs = -1):

    print(xM.shape, yV.shape)

    clf = linear_model.Ridge()
    #parmas = {'alpha': np.logspace(1, -1, 9)}
    parmas = {'alpha': np.logspace( *alphas_log)}
    kf_n_c = model_selection.KFold( n_splits = n_splits, shuffle=True)
    # kf_n = kf5_ext_c.split( xM)
    gs = model_selection.GridSearchCV( clf, parmas, scoring = 'r2', cv = kf_n_c, n_jobs = n_jobs)

    gs.fit( xM, yV)

    return gs

def gsLOO( method, xM, yV, alphas_log = (1, -1, 9), n_jobs = 1, scores = "MedAE", disp = True, graph = False):
    """
    all grid search results 
    input
    ======
    scoring: string 
    'r2', '', 'mean_absolute_error‘, 'mean_squared_error’

    """
    X, y = map( np.array, [xM, yV])
    print(X.shape, y.shape)

    clf = linear_model.Ridge()
    #parmas = {'alpha': np.logspace(1, -1, 9)}
    alpha_l = np.logspace( *alphas_log)
    df_l = list()
    for idx, alpha in enumerate(alpha_l): 
        yp = cvLOO( method, X, y, alpha, n_jobs = n_jobs, graph = False)

        df = pd.DataFrame()
        df["idx(alpha)"] = [idx] * y.shape[0]
        df["alpha"] = [alpha] * y.shape[0]  
        df["y"] = y
        df["yp"] = yp
        df["e"] = y - yp
        df["abs(e)"] = df["e"].abs()
        df_l.append( df)
        if disp:
            print( idx, "Alpha = {0}: MedAE = {1}".format( alpha, df["abs(e)"].median()))
            #print( df.describe())

    all_df = pd.concat( df_l, ignore_index = True)
    g_df = all_df.groupby("idx(alpha)")
    
    if scores == "MedAE":
        best_idx = g_df["abs(e)"].median().argmin()
    elif scores == "std":
        best_idx = g_df["e"].median().argmin()

    all_df["Best"] = [False] * all_df.shape[0]
    all_df.loc[ all_df["idx(alpha)"] == best_idx, "Best"] = True

    best_df = all_df[ all_df["idx(alpha)"] == best_idx].reset_index( drop = True)

    if disp:
        print( "Best idx(alpha) and alpha:", (best_idx, best_df['alpha'][0]))

    if graph:
        kutil.regress_show4( best_df['y'], best_df['yp'])

    return all_df


def gs_Ridge_BIKE( A_list, yV, XX = None, alphas_log = (1, -1, 9), n_splits = 5, n_jobs = -1):
    """
    As is a list of A matrices where A is similarity matrix. 
    X is a concatened linear descriptors. 
    If no X is used, X can be empty
    """

    clf = binary_model.BIKE_Ridge( A_list, XX)
    parmas = {'alpha': np.logspace( *alphas_log)}
    ln = A_list[0].shape[0] # ls is the number of molecules.

    kf_n_c = model_selection.KFold( n_splits = n_splits, shuffle=True)
    #kf_n = kf5_ext_c.split( A_list[0])
    gs = model_selection.GridSearchCV( clf, parmas, scoring = 'r2', cv = kf_n_c, n_jobs = n_jobs)
    
    AX_idx = np.array([list(range( ln))]).T
    gs.fit( AX_idx, yV)

    return gs

def gs_BIKE_Ridge( A_list, yV, alphas_log = (1, -1, 9), X_concat = None, n_splits = 5, n_jobs = -1):
    """
    As is a list of A matrices where A is similarity matrix. 
    X is a concatened linear descriptors. 
    If no X is used, X can be empty
    """

    clf = binary_model.BIKE_Ridge( A_list, X_concat)
    parmas = {'alpha': np.logspace( *alphas_log)}
    ln = A_list[0].shape[0] # ls is the number of molecules.

    kf_n_c = model_selection.KFold( n_splits = n_splits, shuffle=True)
    #kf_n = kf5_ext_c.split( A_list[0])
    gs = model_selection.GridSearchCV( clf, parmas, scoring = 'r2', cv = kf_n_c, n_jobs = n_jobs)
    
    AX_idx = np.array([list(range( ln))]).T
    gs.fit( AX_idx, yV)

    return gs


def _cv_r0( method, xM, yV, alpha, n_splits = 5, n_jobs = -1, grid_std = None, graph = True):
    """
    method can be 'Ridge', 'Lasso'
    cross validation is performed so as to generate prediction output for all input molecules
    """ 
    print(xM.shape, yV.shape)

    clf = getattr( linear_model, method)( alpha = alpha)
    kf_n_c = model_selection.KFold( n_splits = n_splits, shuffle=True)
    kf_n = kf5_ext_c.split( xM)
    yV_pred = model_selection.cross_val_predict( clf, xM, yV, cv = kf_n, n_jobs = n_jobs)

    if graph:
        print('The prediction output using cross-validation is given by:')
        kutil.cv_show( yV, yV_pred, grid_std = grid_std)

    return yV_pred


def cv( method, xM, yV, alpha, n_splits = 5, n_jobs = -1, grid_std = None, graph = True, shuffle = True):
    """
    method can be 'Ridge', 'Lasso'
    cross validation is performed so as to generate prediction output for all input molecules
    """
    print(xM.shape, yV.shape)

    clf = getattr( linear_model, method)( alpha = alpha)
    kf_n_c = model_selection.KFold( n_splits=n_splits, shuffle=shuffle)
    kf_n = kf_n_c.split( xM)
    yV_pred = model_selection.cross_val_predict( clf, xM, yV, cv = kf_n, n_jobs = n_jobs)

    if graph:
        print('The prediction output using cross-validation is given by:')
        kutil.cv_show( yV, yV_pred, grid_std = grid_std)

    return yV_pred

def cvLOO( method, xM, yV, alpha, n_jobs = -1, grid_std = None, graph = True):
    """
    method can be 'Ridge', 'Lasso'
    cross validation is performed so as to generate prediction output for all input molecules
    """ 
    n_splits = xM.shape[0]

    # print(xM.shape, yV.shape)

    clf = getattr( linear_model, method)( alpha = alpha)
    kf_n = model_selection.KFold( xM.shape[0], n_splits=n_splits)
    yV_pred = model_selection.cross_val_predict( clf, xM, yV, cv = kf_n, n_jobs = n_jobs)

    if graph:
        print('The prediction output using cross-validation is given by:')
        kutil.cv_show( yV, yV_pred, grid_std = grid_std)

    return yV_pred  

def cv_Ridge_BIKE( A_list, yV, XX = None, alpha = 0.5, n_splits = 5, n_jobs = -1, grid_std = None):

    clf = binary_model.BIKE_Ridge( A_list, XX, alpha = alpha)
    ln = A_list[0].shape[0] # ls is the number of molecules.
    kf_n_c = model_selection.KFold( n_splits = n_splits, shuffle=True)
    kf_n = kf5_ext_c.split( A_list[0])

    AX_idx = np.array([list(range( ln))]).T
    yV_pred = model_selection.cross_val_predict( clf, AX_idx, yV, cv = kf_n, n_jobs = n_jobs)

    print('The prediction output using cross-validation is given by:')
    kutil.cv_show( yV, yV_pred, grid_std = grid_std)

    return yV_pred

def cv_BIKE_Ridge( A_list, yV, alpha = 0.5, XX = None, n_splits = 5, n_jobs = -1, grid_std = None):

    clf = binary_model.BIKE_Ridge( A_list, XX, alpha = alpha)
    ln = A_list[0].shape[0] # ls is the number of molecules.
    kf_n_c = model_selection.KFold( n_splits = n_splits, shuffle=True)
    kf_n = kf5_ext_c.split( A_list[0])

    AX_idx = np.array([list(range( ln))]).T
    yV_pred = model_selection.cross_val_predict( clf, AX_idx, yV, cv = kf_n, n_jobs = n_jobs)

    print('The prediction output using cross-validation is given by:')
    kutil.cv_show( yV, yV_pred, grid_std = grid_std)

    return yV_pred  


def topscores( gs):
    """
    return only top scores for ridge and lasso with the best parameters
    """
    top_score = sorted(gs.grid_scores_, key=itemgetter(1), reverse=True)[0]
    
    print(top_score.parameters)
    print(top_score.cv_validation_scores)

    return top_score.parameters, top_score.cv_validation_scores

def pd_dataframe( param, scores, descriptor = "Morgan(r=6,nB=4096)", graph = True):

    pdw_score = pd.DataFrame()
    
    k_KF = len(  scores)
    pdw_score["descriptor"] = [ descriptor] * k_KF
    pdw_score["regularization"] = ["Ridge"] * k_KF
    pdw_score["alpha"] = [param['alpha']] * k_KF
    pdw_score["KFold"] = list(range( 1, k_KF + 1))
    pdw_score["r2"] = scores

    if graph:
        pdw_score['r2'].plot( kind = 'box')

    return pdw_score


def ridge( xM, yV, alphas_log, descriptor = "Morgan(r=6,nB=4096)", n_splits = 5, n_jobs = -1):
    gs = gs_Ridge( xM, yV, alphas_log = alphas_log, n_splits = n_splits, n_jobs = n_jobs)

    param, scores = topscores( gs)

    pdw_score = pd_dataframe( param, scores, descriptor = descriptor)
    
    print('Ridge(alpha={0}) = {1}'.format( gs.best_params_['alpha'], gs.best_score_))
    print(pdw_score)

    return pdw_score


def gs( method, xM, yV, alphas_log):
    """
    It simplifies the function name into the two alphabets
    by adding one more argument which is the name of a method.
    """

    if method == "Lasso":
        return gs_Lasso( xM, yV, alphas_log)
    elif method == "Ridge":
        return gs_Ridge( xM, yV, alphas_log)    
    else:
        raise NameError("The method of {} is not supported".format( method))

def gscvLOO( xM, yV, mode = "Lasso", graph = True):
    X, y = np.array( xM), np.array( yV.A1)
    df_reg = gsLOO( mode, X, y, (-1,1,6), graph = graph)
    yp_reg = df_reg[ df_reg.Best == True]["yp"].values
    return yp_reg

def get_r2_mean_std(r2_df):
    r2_df_g = r2_df.groupby(["C"])
    r2_mean = r2_df_g.mean()
    r2_std = r2_df_g.std()
    return r2_mean, r2_std 

class GridSVRPrecomp():
    def __init__(self, Aall, yall, C_a=np.logspace(0,3,4), epsilon=0.1, n_splits=5):
        self.Aall = Aall
        self.yall = yall
        self.C_a = C_a
        self.n_splits = n_splits
        self.epsilon = epsilon
        
    def calc(self):
        """
        Aall <- kernel matrix of X
        yall <- equivalent to y
        """
        Aall = self.Aall
        yall = self.yall
        C_a = self.C_a
        n_splits = self.n_splits  
        epsilon = self.epsilon      

        df_l = list()
        KF = model_selection.KFold(n_splits=n_splits, shuffle=True)
        for C in C_a:
            for it, (tr, te) in enumerate(KF.split(Aall)):
                r, e = np.array(tr).reshape(-1,1), np.array(te).reshape(-1,1)
                Atr = Aall[r,r.T]
                ytr = yall[tr]
                Atetr = Aall[e,r.T]
                yte = yall[te]

                ms = svm.SVR(kernel='precomputed', C=C, epsilon=epsilon)
                ms.fit(Atr, ytr)
                r2_tr = ms.score(Atr, ytr)
                r2_te = ms.score(Atetr, yte)
                print( C, r2_tr, r2_te)
                df_i = pd.DataFrame()
                df_i["C"] = [C]
                df_i["unit"] = [it]
                df_i["r2_tr"] = [r2_tr]
                df_i["r2_te"] = [r2_te]
                df_l.append( df_i)
            print('--------------')
            self.r2_df = pd.concat(df_l,ignore_index=True)
        return self

    def get_r2(self):
        """
        Return
        ------
        r2_mean, mean of r2_df
        r2_std, std of r2_df
        """
        r2_df = self.r2_df
        return get_r2_mean_std(r2_df) 


def avg_cv_SVC(X, y, scoring='accuracy', n_jobs=1): # squared_hinge
    SVC = svm.SVC 
    KFold = model_selection.KFold
    cross_val_score = model_selection.cross_val_score
    
    acc_a_l = []
    for ii in range(10):
        svc = SVC(C=1, gamma=0.001)
        cv = KFold(n_splits=5, shuffle=True)
        acc_a = cross_val_score(svc, X, y, cv=cv, n_jobs=n_jobs, scoring=scoring)
        acc_a_l.append(acc_a)
    print('Mean, STD are', np.mean(acc_a_l), np.std(acc_a_l))
    return acc_a_l