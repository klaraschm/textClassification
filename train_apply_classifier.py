import multiprocessing
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
import pickle

# my own modules
import loadData
import parentCategories
import loadCategories
# preprocessing
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# classifiers
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
# evaluation
import time
from scipy.stats import scoreatpercentile
import collections
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import recall_score, confusion_matrix, classification_report, f1_score, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from scipy import interp







def train_apply_classifier( \
    classifier = 'logreg', \
    qfile_train = 'question_train.csv', \
    qcatfile_train = 'question_category_train.csv', \
    catfile = 'category.csv', \
    qfile_test = 'question_test.csv', \
    subcats = False):
    
    
    print('classifier ', classifier) 
    ## LOAD DATA 
    subcatids, data = loadData.loadAndCleanData(qfile_train, stemming =False)
    
    
    ## remove near duplicate questions
    remove_duplicates = False
    if remove_duplicates:
        X = CountVectorizer().fit_transform(data)
        _, indices = np.unique(X.toarray(), axis=0, return_index=True)
        indices = list(map(int, indices))
        nof_samples = X.shape[0]
        nof_dups = nof_samples - len(indices)
        print('nof dups: ', nof_dups)
        data = [data[i] for i in indices] 
        subcatids = [subcatids[i] for i in indices] 
    
    # map child categories to parent categories
    parCatids = parentCategories.parentCats(catfile, subcatids)
    # removing class 0, because that is just a root class.
    cat0_idx = parCatids.index(0)
    parCatids.pop(cat0_idx)
    data.pop(cat0_idx)
    
        
    categories, category_hierarchy, categories_all = loadCategories.extractCategories(catfile, subcatids)
    parcat_names_d = {k: categories_all[k] for k in (categories_all.keys() and parCatids)}
    parcat_names = list(parcat_names_d.values())
  
    ## load test data
    _, X_test = loadData.loadAndCleanData(qfile_test, test_file = True)
    
   
    ## split data into training and cross-validation set (if needed)
    if subcats== True:
        X_train, X_cv, y_train, y_cv = train_test_split(data, subCatids, test_size=0.3, random_state=0)
        y = subCatids
    else: 
        X_train, X_cv, y_train, y_cv = train_test_split(data, parCatids, test_size=0.3, random_state=0)
        y = parCatids
    X = data
    
    classes = sorted(set(y))
    n_classes = len(classes)
    # print(classes, n_classes)
    counter=collections.Counter(y)
    n_samples_per_class = [x[1] for x in sorted(counter.items())]
   
    
    german_stops = set(stopwords.words('german'))
    german_stops.update(('was', 'wie', 'warum', 'wann', 'wieso', 'weshalb', 'wer', 'wieviel', 'wie viel', 'gibt', 'bedeutet', 'viel', 'kommt', 'heißt', 'heisst', 'welch', 'fuer'))

    
    
    
    # BUILD CLASSIFIERS
    
    if classifier == 'nbc':
        parameters_vect = {'tfidf__stop_words': [german_stops, None], 'tfidf__min_df': [1,2,3], 'tfidf__max_df': [0.3, 0.5, 0.8, 1.0], 'tfidf__analyzer': ['word', 'char'], 'tfidf__ngram_range': [(3,4),(3,5),(3,6), (3,7),(3,8)] }
        parameters_sel = {'sel__percentile':[10,20,30,40,50,60,70,80,90]}
        parameters_clf = {'clf__alpha': [ 0.1, 0.01,0.001], 'clf__fit_prior':[True, False]}                  
        clf = Pipeline([('tfidf', TfidfVectorizer(use_idf = False, norm=None, stop_words = german_stops, min_df = 1, max_df = 0.5, analyzer = 'char', ngram_range=(3,5))), \
                         ('sel', SelectPercentile(chi2, percentile=40)), \
                         ('clf', BernoulliNB(fit_prior=True, alpha=0.01))])                      
   
    ##  scale data ? 
    elif classifier == 'svm':
        parameters_vect = {'tfidf__use_idf':[True, False], 'tfidf__max_df': [0.3, 0.5, 0.8, 1.0], 'tfidf__analyzer': ['word', 'char'], 'tfidf__ngram_range': [(3,4),(3,6), (3,8)], 'tfidf__stop_words': [german_stops, None]}
        parameters_sel = {'sel__percentile':[10,20,30,40,50,60,70,80,90]} 
        parameters_clf = {'clf__penalty' : ['l2', 'l1' ], 'clf__alpha': [0.0001, 0.001, 0.00001], 'clf__class_weight': [None, 'balanced']}
        clf = Pipeline([('tfidf', TfidfVectorizer(use_idf= True, min_df=1, analyzer='char', max_df =0.8, ngram_range=(3,6), stop_words=None)), \
                        ('sel', SelectPercentile(chi2, percentile=30)), \
                        ('clf', SGDClassifier(loss= 'hinge', max_iter =50, penalty ='l2', alpha=0.0001, class_weight=None)),])

    ##  scale data ? 
    elif classifier == 'logreg':
        parameters_vect = {'tfidf__max_df': [0.3, 0.5, 0.8, 1.0], 'tfidf__analyzer': ['word','char'], 'tfidf__ngram_range': [(3,3),(3,5),(3,7)], \
                            'tfidf__use_idf': (True, False), 'tfidf__stop_words': [german_stops, None]} 
        parameters_sel = {'sel__percentile':[10,20,30,40,50,60,70,80,90]}               
        parameters_clf = {'clf__penalty' : ['l2', 'l1'], 'clf__alpha': [ 0.001, 0.0001, 0.00001], 'clf__class_weight': [None, 'balanced']}                         
        clf = Pipeline([('tfidf', TfidfVectorizer(stop_words=german_stops, analyzer='char', ngram_range=(3,5), max_df=0.8, use_idf =False)), \
                        ('sel', SelectPercentile(chi2, percentile =30)),\
                         ('clf', SGDClassifier(loss='log', max_iter =50, alpha= 0.00001, class_weight='balanced', penalty='l2')),])

            
    else:
        print('classifier has to be "nbc", "svm" or "logreg"')
        exit()
                    
    print('training classifier')
    gridsearch = False
    # optimize classifier
    if gridsearch:
        clf = GridSearchCV(clf, parameters_clf, scoring='f1_macro')
    # train classifier
    clf.fit(X_train, y_train)
    pred_cv = clf.predict(X_cv)
    if gridsearch:
        print(clf.best_params_)
        print(clf.cv_results_.keys())
        if True:
            print(clf.cv_results_['mean_test_score'])
            print(clf.cv_results_['mean_train_score'])
            print(clf.cv_results_['params'])
            print(clf.cv_results_['rank_test_score'])
        

        clfBest = clf.best_estimator_
        print(clf.best_score_)
    joblib.dump(clf, classifier +'.pkl') 
        
    
    # PERFORMANCE EVALUATION
    if True:
        print('evaluating classifier')

        scoring = ['accuracy', 'f1_micro', 'f1_macro', 'precision_micro', 'precision_macro','recall_micro', 'recall_macro']
        scores = cross_validate(clf, X, y, scoring=scoring, cv=3, return_train_score=True)
        fit_time = scores['fit_time'].mean()
        score_time = scores['score_time'].mean()
        scores_test = [scores['test_accuracy'].mean(), scores['test_f1_macro'].mean(), scores['test_f1_micro'].mean(), scores['test_precision_micro'].mean(), scores['test_precision_macro'].mean(), scores['test_recall_micro'].mean(), scores['test_recall_macro'].mean()]
        scores_train = [scores['train_accuracy'].mean(), scores['train_f1_macro'].mean(), scores['train_f1_micro'].mean() ]
        print('scores_test ', scores_test)
        print('scores_train ', scores_train)
    
        cnf_matrix = confusion_matrix(y_cv, pred_cv, labels = classes)
        print(cnf_matrix)
        report = classification_report(y_cv, pred_cv)
        print(report)


    predicted_test = clf.predict(X_test)
    return np.array(predicted_test).reshape(len(predicted_test),1)

        
    
if __name__ == "__main__":   

    predicted_test = train_apply_classifier()



    