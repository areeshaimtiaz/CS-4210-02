from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def train_model(df):
    X = df['cleaned']
    y = df['label']

    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    # Model 1: Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)

    # Model 2: Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    print("=== Naive Bayes ===")
    print("Accuracy:", accuracy_score(y_test, nb_preds))
    print("Precision:", precision_score(y_test, nb_preds, pos_label='spam'))
    print("Recall:", recall_score(y_test, nb_preds, pos_label='spam'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, nb_preds))

    print("\n=== Logistic Regression ===")
    print("Accuracy:", accuracy_score(y_test, lr_preds))
    print("Precision:", precision_score(y_test, lr_preds, pos_label='spam'))
    print("Recall:", recall_score(y_test, lr_preds, pos_label='spam'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, lr_preds))

    return vectorizer, nb_model  # we’ll use NB for prediction