from src.preprocess import load_data, preprocess_data
from src.train import train_model
from src.predict import predict_message

def main():
    # Load dataset
    df = load_data("data/SMSSpamCollection")

    # Preprocess
    df = preprocess_data(df)

    # Train model
    vectorizer, model = train_model(df)

    # Demo
    print("\n--- Spam Classifier Demo ---")
    while True:
        msg = input("Enter a message (or 'quit'): ")
        if msg.lower() == 'quit':
            break

        result = predict_message(msg, vectorizer, model)
        print("Prediction:", result)

if __name__ == "__main__":
    main()