import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
import sys

def main():
    path='suicidality.tsv'
    suicidality=pd.read_table(path, encoding='ISO-8859-1',header=None, names=['label', 'tweets'])
    suicidality['label_num']=suicidality.label.map({'safe to ignore':0,'possibly concerning': 1, 'strongly concerning':2})

    X = suicidality.tweets
    y = suicidality.label_num

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


    #print(X_test.head)
    vect = CountVectorizer()
    X_train_dtm = vect.fit_transform(X_train)

    X_test_dtm = vect.transform(X_test)
    tfidf_transformer = TfidfTransformer(norm="l2")
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_dtm )
    X_test_tfidf = tfidf_transformer.fit_transform( X_test_dtm )
    nb = svm.LinearSVC()
    nb.fit(X_train_tfidf, y_train)
    y_pred_class = nb.predict(X_test_tfidf)
    print(X_test)
    print(y_pred_class)
    print()
    print(metrics.accuracy_score(y_test, y_pred_class))
    print()
    for x in range(0, 20):
        simple_test = [input("Tweet: ")]
        if(len(simple_test)>140):
            print ("Only 140 words are allowed")
            sys.exit()

        simple_test_dtm = vect.transform(simple_test)
        simple_test_tfidf= tfidf_transformer.fit_transform( simple_test_dtm )
        simple_test_tfidf=simple_test_tfidf.toarray()
        level_of_concern={0 : 'safe to ignore', 1 : 'possibly concerning', 2 : 'strongly concerning'}
        print(level_of_concern[nb.predict(simple_test_tfidf)[0]])
if __name__== "__main__":
  main()