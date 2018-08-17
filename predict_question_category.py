from sklearn.externals import joblib
import numpy as np

# my own modules
import loadData



def predict_question_category(qfile = 'questions.csv'):

    ## load test data
    qfile = 'question_test.csv'
    print(qfile)
    _, X_test = loadData.loadAndCleanData(qfile, test_file = True,  stemming= False)
    print('len Xtest ', len(X_test))
    
    clf_main = joblib.load('finalclassifier_main.pkl')
    clf_sub = joblib.load('finalclassifier_sub.pkl')
    preds_main = clf_main.predict(X_test)
    probas_main = clf_main.predict_proba(X_test) 
    preds_sub = clf_sub.predict(X_test)
    probas_sub = clf_sub.predict_proba(X_test) 
    
 
    # main classes
    classes = clf_main.classes_
    translate_d = dict(list(zip( classes, np.arange(len(classes))))) 
    preds = np.array([translate_d[p] for p in preds_main])
    confidence_main = []
    for i,p in enumerate(preds):
        confidence_main.append(probas_main[i,p])
        
    # sub classes
    classes = clf_sub.classes_
    translate_d = dict(list(zip( classes, np.arange(len(classes))))) 
    preds = np.array([translate_d[p] for p in preds_sub])
    confidence_sub = []
    for i,p in enumerate(preds):
        confidence_sub.append(probas_sub[i,p])
        
    values = zip(preds_main, preds_sub, confidence_main, confidence_sub)
    keys = ['major_category', 'minor_category', 'confidence_major_cat', 'confidence_minor_cat']
    return([dict(zip(keys, v)) for v in values])
   
   
   
    
if __name__ == "__main__":
    
   predict_question_category(qfile = 'questions.csv')

  
    
  
    
    
  