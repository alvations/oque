
import io

import numpy as np
from scipy.stats import uniform as sp_rand

from sklearn.linear_model import BayesianRidge
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR

from o import cosine_feature

def load_quest(direction, dataset, which_data, quest_data_path='quest/'):
    '''
    # USAGE:
    baseline_train = load_quest('en-de', 'training', 'baseline17')
    baseline_train = load_quest('en-de', 'test', 'baseline17')
    meteor_train = load_quest('en-de', 'training', 'meteor')
    meteor_test =  load_quest('en-de', 'test', 'meteor')
    '''
    x = np.loadtxt(quest_data_path+direction+'.'+dataset+'.'+which_data)
    x = x / np.linalg.norm(x)
    return x

    
def experiments(direction, to_tune, with_cosine, classifier=None):
    '''
    # USAGE:
    direction = 'en-de'
    to_tune = False
    with_cosine = False
    outfilename, mae, mse = experiments(direction, to_tune, with_cosine)
    print outfilename, mae, mse
    '''
    train, test = 'training', 'test'
    # Load train data
    baseline_train = load_quest(direction, train, 'baseline17') 
    cos_train = cosine_feature(direction, train)
    meteor_train = load_quest(direction, train, 'meteor')
    # Load test data
    baseline_test = load_quest(direction, test, 'baseline17')
    cos_test = cosine_feature(direction, test)
    meteor_test = load_quest(direction, test, 'meteor')
    
    # Create training array and outputs
    X_train = baseline_train
    if with_cosine:
        X_train = np.concatenate((baseline_train, cos_train), axis=1)
    y_train = meteor_train
    # Create testing array and gold answers
    X_test = baseline_test
    if with_cosine:
        X_test = np.concatenate((baseline_test, cos_test), axis=1)
    y_test = meteor_test
    
    
    # Initialize Classifier.
    clf = BayesianRidge()
    #clf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    if classifier:
        clf = classifier
        to_tune = False
    if to_tune:
        # Grid search: find optimal classifier parameters.
        param_grid = {'alpha_1': sp_rand(), 'alpha_2': sp_rand()}
        #param_grid = {'C': sp_rand(), 'gamma': sp_rand()}
        rsearch = RandomizedSearchCV(estimator=clf, 
                                     param_distributions=param_grid, n_iter=5000)
        rsearch.fit(X_train, y_train)
        # Use tuned classifier.
        clf = rsearch.best_estimator_
          
    # Trains Classifier   
    clf.fit(X_train, y_train)
    
    # Outputs to file.
    to_tune_str = 'tuned' if to_tune else 'notune'
    model_name = 'withcosine' if with_cosine else 'baseline'
    outfile_name = ".".join(['oque',model_name,to_tune_str,direction,'output']) 
    with io.open(outfile_name, 'w') as fout:
        answers = []
        for i in clf.predict(X_test):
            answers.append(i)
            fout.write(unicode(i)+'\n')
    
    mse = mean_squared_error(y_test, np.array(answers))
    mae = mean_absolute_error(y_test, np.array(answers))
    
    #for i,j,k in zip(answers, y_test, baseline_train):
    #    print i,j, k
    
    return outfile_name, mae, mse


# 0.0967

direction = 'en-de'
to_tune = False
with_cosine = False
outfilename, mae, mse = experiments(direction, to_tune, with_cosine)
print outfilename, mae, mse

