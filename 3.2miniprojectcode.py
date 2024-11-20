import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
# Load the IMDb movie review dataset
max_words = 10000
maxlen = 100
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_words)

# Load the IMDb movie review dataset
max_words = 10000
maxlen = 100
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_words)

# Pad sequences to ensure uniform length
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Define the LSTM model
model = Sequential()
model.add(Embedding(max_words, 128))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))
model.save('/content/model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Function to preprocess the user input and make predictions
def predict_sentiment(model, review_text):
    # Tokenize the review text
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts([review_text])
    review_sequence = tokenizer.texts_to_sequences([review_text])
    review_sequence = pad_sequences(review_sequence, maxlen=maxlen)
    # Predict sentiment
    prediction = model.predict(review_sequence)[0]
    if prediction >= 0.5:
        return "Positive"
    else:
        return "Negative"

# Get user input
user_review = input("Enter your movie review: ")
# Predict sentiment
sentiment = predict_sentiment(model, user_review)
print("Predicted Sentiment:", sentiment)