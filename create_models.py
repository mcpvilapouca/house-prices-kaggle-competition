# This script defines the models I think might score better.
# All the parameters are added with scikit-learn default values so I can
# later tune the parameters I want

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import numpy as np


def linmodel(n_jobs=None):
    model=LinearRegression(n_jobs=n_jobs)

    return model

def random_forest(n_estimators=100, criterion='squared_error', max_depth=None,
    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0,
    max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None,
    random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None):

    model=RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
    max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs,
    random_state=random_state, verbose=verbose, warm_start=warm_start, ccp_alpha=ccp_alpha, max_samples=max_samples)

    return model

def knn(n_neighbors=5,n_jobs=None):

    model=KNeighborsRegressor(n_neighbors=n_neighbors,n_jobs=n_jobs)

    return model

def logregressor(n_jobs=None,max_iter=100,solver='lbfgs'):

    model=LogisticRegression(n_jobs=n_jobs,max_iter=max_iter,solver=solver)

    return model

def gradboost(loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse',
                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3,
                min_impurity_decrease=0.0, init=None, random_state=None, max_features=None,
                alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1,
                n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0):


    model=GradientBoostingRegressor(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, criterion=criterion,
                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth,
                min_impurity_decrease=min_impurity_decrease, init=init, random_state=random_state, max_features=max_features,
                alpha=alpha, verbose=verbose, max_leaf_nodes=max_leaf_nodes, warm_start=warm_start, validation_fraction=validation_fraction,
                n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)

    return model

def xgboost(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0,
       importance_type='gain', learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=np.nan, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=0):

    model=xgb.XGBRegressor(base_score=base_score, booster=booster, colsample_bylevel=colsample_bylevel,
       colsample_bynode=colsample_bynode, colsample_bytree=colsample_bytree, gamma=gamma,
       importance_type=importance_type, learning_rate=learning_rate, max_delta_step=max_delta_step,
       max_depth=max_depth, min_child_weight=min_child_weight, missing=missing, n_estimators=n_estimators,
       n_jobs=n_jobs, nthread=nthread, objective=objective, random_state=random_state,
       reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight, seed=seed,
       silent=silent, subsample=subsample, verbosity=verbosity)

    return model

def ridge(alpha=1, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None):

    model=KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, kernel_params=kernel_params)

    return model