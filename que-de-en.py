import io, sys

import numpy as np
from scipy.stats import uniform as sp_rand
from itertools import combinations

from sklearn.linear_model import BayesianRidge
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from o import cosine_feature, complexity_feature

train, test = 'training', 'test'

def load_quest(direction, dataset, which_data, quest_data_path='quest/', 
               to_normalize=True):
    '''
    # USAGE:
    baseline_train = load_quest('en-de', 'training', 'baseline17')
    baseline_train = load_quest('en-de', 'test', 'baseline17')
    meteor_train = load_quest('en-de', 'training', 'meteor')
    meteor_test =  load_quest('en-de', 'test', 'meteor')
    '''
    x = np.loadtxt(quest_data_path+direction+'.'+dataset+'.'+which_data)
    if to_normalize:
        x = x / np.linalg.norm(x)
    return x

def load_wmt15_data(direction):    
    # Load train data
    baseline_train = load_quest(direction, train, 'baseline17', to_normalize=False)     
    meteor_train = load_quest(direction, train, 'meteor', to_normalize=False)
    # Load test data
    baseline_test = load_quest(direction, test, 'baseline17', to_normalize=False)
    meteor_test = load_quest(direction, test, 'meteor', to_normalize=False)
    return baseline_train, meteor_train, baseline_test, meteor_test
    

def load_cosine_features(direction):
    cos_train = cosine_feature(direction, train)
    #cos_train = complexity_feature(direction, train)
    cos_test = cosine_feature(direction, test)
    #cos_test = complexity_feature(direction, test)
    return cos_train, cos_test    

def train_classiifer(X_train, y_train, to_tune, classifier):
    # Initialize Classifier.
    clf = BayesianRidge()
    clf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    #clf = RandomForestRegressor()
    if classifier:
        clf = classifier
        to_tune = False
    if to_tune:
        # Grid search: find optimal classifier parameters.
        param_grid = {'alpha_1': sp_rand(), 'alpha_2': sp_rand()}
        param_grid = {'C': sp_rand(), 'gamma': sp_rand()}
        rsearch = RandomizedSearchCV(estimator=clf, 
                                     param_distributions=param_grid, n_iter=5000)
        rsearch.fit(X_train, y_train)
        # Use tuned classifier.
        clf = rsearch.best_estimator_
          
    # Trains Classifier   
    clf.fit(X_train, y_train)
    return clf


def brute_force_feature_selection():
    x = range(1,18)
    for l in range (1,len(x)+1):
        for f in list(combinations(range(0,len(x)),l)):
            yield f

def evaluate_classifier(clf, X_test, direction, with_cosine, 
                        to_tune, to_output=True, to_hack=False):
    answers = list(clf.predict(X_test))
    if to_hack:
        hacked_answers = []
        for i,j in zip(answers, X_test):
            if j[9] > 0.7 and j[0] < 12: i = i - 0.2;
            if j[0] ==1 or j[1]== 1: i = i - 0.15;
            if j[0] > 200: i = i - 0.1;
            if i < 0: i = 0.0;
            hacked_answers.append(i)
        answers = hacked_answers
    outfile_name = ''
    if to_output: # Outputs to file.
        to_tune_str = 'tuned' if to_tune else 'notune'
        model_name = 'withcosine' if with_cosine else 'baseline'
        outfile_name = ".".join(['oque',model_name,
                                 to_tune_str,direction,'output'])
        
        with io.open(outfile_name, 'w') as fout:
            for i in answers:
                fout.write(unicode(i)+'\n')
    return answers, outfile_name

def brute_force_classification(X_train, y_train, X_test, y_test,
                               direction, with_cosine,
                               to_tune, to_output=False, to_hack=False):
    
    #score_fout = io.open('que.'+direction+'.scores', 'w')
    for f in brute_force_feature_selection():
        _X_train = X_train[:, f]
        
        _X_test = X_test[:, f]
        # Train classifier
        clf = train_classiifer(_X_train, y_train, to_tune, classifier=None)
        answers, outfile_name = evaluate_classifier(clf, _X_test, direction, 
                                               with_cosine, to_tune, 
                                               to_output=False, to_hack=False)
    
        mse = mean_squared_error(y_test, np.array(answers))
        mae = mean_absolute_error(y_test, np.array(answers))
        
        outfile_name = "results/oque.baseline." + direction +'.'+str(mae) + '.' 
        outfile_name+= "-".join(map(str, f))+'.output'
        with io.open(outfile_name, 'w') as fout:
            for i in answers:
                fout.write(unicode(i)+'\n')
        print mae, f
        sys.stdout.flush()
    
def experiments(direction, with_cosine, to_tune, to_output=True, to_hack=False, 
                to_debug=False, classifier=None):
    '''
    # USAGE:
    direction = 'en-de'
    to_tune = False
    with_cosine = False
    outfilename, mae, mse = experiments(direction, to_tune, with_cosine)
    print outfilename, mae, mse
    '''
    # Create training and testing array and outputs
    X_train, y_train, X_test, y_test = load_wmt15_data(direction)
    if with_cosine:
        # Create cosine features for training
        cos_train, cos_test = load_cosine_features(direction)
        X_train = np.concatenate((X_train, cos_train), axis=1)
        X_test = np.concatenate((X_test, cos_test), axis=1)
    
    brute_force_classification(X_train, y_train, X_test, y_test, direction, 
                               with_cosine, to_tune, to_output=False, 
                               to_hack=False)
    
    '''
    # Best setup for EN-DE up till now.
    f = (2, 9, 13)
    _X_train = X_train[:, f]
    _X_test = X_test[:, f]
    clf = train_classiifer(_X_train, y_train, to_tune, classifier=None)
    answers, outfile_name = evaluate_classifier(clf, _X_test, direction, 
                                           with_cosine, to_tune, 
                                           to_output=True, to_hack=False)
    '''
    
    mse = mean_squared_error(y_test, np.array(answers))
    mae = mean_absolute_error(y_test, np.array(answers))
    
    if to_debug:
        srcfile = io.open('quest/en-de_source.test', 'r')
        trgfile = io.open('quest/en-de_target.test', 'r')
        cos_train, cos_test = load_cosine_features(direction)
        for i,j,k,s,t, c in zip(answers, y_test, X_test, 
                                srcfile, trgfile, cos_test):
            if i - j > 0.095 or j -1 > 0.095 or c == 9.99990000e-11: 
                print i, j, k[0], k[9], k, c
                print s, t
        
    return outfile_name, mae, mse

direction = 'de-en'
with_cosine = False
to_tune = False
to_output = False
outfilename, mae, mse = experiments(direction, with_cosine,to_tune, to_output, to_debug=False)
print outfilename, mae, mse

# DE-EN
# no-hack at all
# oque.baseline.notune.de-en.output 0.0692666454858 0.011038250617
# no-hack, with cosine
# oque.withcosine.notune.de-en.output 0.0692590476386 0.0110349222335

# Super default + hack
# oque.baseline.notune.de-en.output 0.0685437539196 0.0106677292505
# hacked
# if j[0] ==1 or j[1]== 1: i = i - 0.15
# oque.withcosine.notune.de-en.output 0.0685361560723 0.0106643693054



# EN-DE
# oque.baseline.notune.en-de.output 0.0980804849285 0.0184924281565

# if j[9] > 0.7 and j[0] < 12: i = i -0.2
# oque.baseline.notune.en-de.output 0.097544087243 0.0208756823852 
# oque.withcosine.notune.en-de.output 0.0975427119756 0.0208755274686

# if j[9] > 0.7 and j[0] < 12: i = i -0.2
# if j[0] ==1 or j[1]== 1: i = i - 0.1
# oque.withcosine.notune.en-de.output 0.0973017481202 0.0207602928984

# if j[9] > 0.7 and j[0] < 12: i = i -0.2
# if j[0] ==1 or j[1]== 1: i = i - 0.15
# oque.withcosine.notune.en-de.output 0.0972310140807 0.0207568924808

# if j[9] > 0.7 and j[0] < 12: i = i -0.2
# if j[0] ==1 or j[1]== 1: i = i - 0.15
# if j[0] > 200: i = i - 0.1
# oque.withcosine.notune.en-de.output 0.0968903228194 0.0206775825255

# if j[9] > 0.7 and j[0] < 12: i = i -0.2
# if j[0] ==1 or j[1]== 1: i = i - 0.15
# if j[0] > 200: i = i - 0.1
# if i < 0: i = 0.0
# oque.withcosine.notune.en-de.output 0.0968359771138 0.0206633629455