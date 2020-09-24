import pandas as pd
import numpy as np

# Extracting the data
dataset = pd.read_csv("Musical_instruments_reviews.csv")
reviews = np.array(dataset.iloc[:, 4].values, dtype='str')
labels = np.array(dataset.iloc[:, 5].values)

# Lowercasing the reviews
reviews = [review.lower() for review in reviews]

# Removing any punctuation
from string import punctuation

reviews = [review + "\n" for review in reviews]
all_text = ''.join([c for r in reviews for c in r if c not in punctuation])

# Splitting the reviews apart
reviews_split = all_text.split('\n')[:-1]

# Create vocab to int mapping dictionary (Tokenizing)
from collections import Counter

all_text_2 = ' '.join(reviews_split)
words = all_text_2.split()
count_words = Counter(words)
total_words = len(count_words)
sorted_words = count_words.most_common(total_words)

vocab_to_int = {w:i+1 for i, (w, n) in enumerate(sorted_words)}

# Encoding the words
reviews_int = []
for review in reviews_split:
    nreview = [vocab_to_int[w] for w in review.split()]
    reviews_int.append(nreview)

# Encoding the labels
encoded_labels = [1 if label >= 3 else 0 for label in labels]
encoded_labels = np.array(encoded_labels)

# Analyzing the data
import matplotlib.pyplot as plt
%matplotlib inline

reviews_len = [len(x) for x in reviews_int]
pd.Series(reviews_len).hist()
plt.show()

pd.Series(reviews_len).describe()

# Removing too short and too long reviews
reviews_int = [reviews_int[i] for i, l in enumerate(reviews_len)  if l > 0]
encoded_labels = [encoded_labels[i] for i, l in enumerate(reviews_len)  if l > 0]

# Padding and truncating the remaining data
def pad_features(reviews_int, seq_length):
    features = np.zeros((len(reviews_int), seq_length), dtype = int)

    for i, review in enumerate(reviews_int):
        review_len = len(review)
        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len))
            new = zeroes+review
        elif review_len > seq_length:
            new = review[0:seq_length]
        
        features[i,:] = np.array(new)
    return features
        
features = pad_features(reviews_int, 150)

# Formatting the data
features = np.reshape(features, (features.shape[0], features.shape[1], 1))
encoded_labels = np.array(encoded_labels)
encoded_labels = np.reshape(encoded_labels, (encoded_labels.shape[0], 1))

# Train test splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2)

# Making the machine learning model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (150, 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 100))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1, activation='sigmoid'))

regressor.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

regressor.fit(X_train, y_train, epochs = 20, batch_size = 32)

y_pred = regressor.predict(X_test)
y_pred = (y_pred > 0.5)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


