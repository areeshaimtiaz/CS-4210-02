# SMS Spam Detection Using Machine Learning

## Overview

This project builds a machine learning system to classify SMS messages as **spam** or **ham (non-spam)** using the UCI SMS Spam Collection dataset.

The system processes text messages, converts them into numerical features using TF-IDF, and applies machine learning models to make predictions.

---

## Dataset

* **Source:** UCI SMS Spam Collection Dataset
* **Total Messages:** 5,574

  * Ham: 4,827
  * Spam: 747

Each message is labeled as either `ham` or `spam`.

---

## Technologies Used

* Python
* pandas
* scikit-learn
* numpy

---

## Project Structure

```
CS-4210-02/
│── sms-spam-classifiers/
│   ├── data/
│   │   ├── SMSSpamCollection
│   │   └── readme
│   │
│   ├── output/
│   │   └── test-cases.png
│   │
│   ├── src/
│   │   ├── __init__.py
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── __pycache__/
│   │
│   ├── main.py
│   ├── requirements.txt
│   └── .DS_Store
│
│── Final Project Proposal.docx
│── README.md
│── main.py
│── requirements.txt
```

---

## How to Run

1. Navigate into the project folder:

```
cd sms-spam-classifiers
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the program:

```
python main.py
```

---

## Pipeline

1. **Data Loading**

   * Load SMS dataset using pandas

2. **Preprocessing**

   * Convert text to lowercase
   * Remove punctuation

3. **Feature Extraction**

   * Transform text into TF-IDF vectors

4. **Model Training**

   * Naive Bayes
   * Logistic Regression

5. **Evaluation**

   * Accuracy
   * Precision
   * Recall
   * Confusion Matrix

6. **Prediction**

   * User inputs a message
   * Model predicts `spam` or `ham`

---

## Example Usage

```
Enter a message (or 'quit'): WINNER!! You have won a prize!
Prediction: spam

Enter a message (or 'quit'): hey are we still meeting today?
Prediction: ham
```

---

## Results

* Naive Bayes Accuracy: ~95–96%
* Logistic Regression Accuracy: ~96%

Logistic Regression performs slightly better overall.

---

## Main Files

* `main.py` → runs the full pipeline and demo
* `src/preprocess.py` → text cleaning
* `src/train.py` → model training and evaluation
* `src/predict.py` → prediction on new messages

---

## Notes

* The `output/` folder contains example test case results
* The `data/` folder contains the original dataset
* Some system files (e.g., `.DS_Store`, `__pycache__`) are auto-generated

---

## Future Improvements

* Add stopword removal and stemming
* Try additional models (SVM, Random Forest)
* Build a web-based interface

---

## Author

Areesha Imtiaz
CS 4210-02 Final Project
