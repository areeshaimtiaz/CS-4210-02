def predict_message(message, vectorizer, model):
    message = message.lower()
    transformed = vectorizer.transform([message])
    prediction = model.predict(transformed)[0]
    return prediction