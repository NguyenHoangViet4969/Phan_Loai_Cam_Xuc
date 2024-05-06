import pandas as pd
import numpy as np

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization

# Đọc dữ liệu từ các file Parquet
train_df = pd.read_parquet("train_49000.parquet")
val_df = pd.read_parquet("vali_7000.parquet")
test_df = pd.read_parquet("Test_Chuan.parquet")

# Lấy các câu và nhãn từ DataFrame
train_sentences = train_df["sentence"].tolist()
train_labels = train_df["label"].values
val_sentences = val_df["sentence"].tolist()
val_labels = val_df["label"].values
test_sentences = test_df["sentence"].tolist()
test_labels = test_df["label"].values

# Tokenization and padding
tokenizer = Tokenizer() # init tokenizer

# training to perform vocabulary learning from the training data set
tokenizer.fit_on_texts(train_sentences)

vocab_size = len(tokenizer.word_index) + 1

max_length = 200  # You can adjust this based on your needs

# Converts words to integers, make them the same length
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post")

val_sequences = tokenizer.texts_to_sequences(val_sentences)
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding="post")

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding="post")

# Define the CNN model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=200, input_length=max_length),
    Conv1D(256, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    BatchNormalization(), # giúp ổn định và tăng tốc quá trình huấn luyện
    Dropout(0.5),
    Dense(2, activation='softmax')
], name="CNN_model")

# Compile the CNN model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# start training
model.fit(train_padded, train_labels, validation_data=(val_padded, val_labels),
                     batch_size=64, epochs=5)

# Predictions with Test Set
predictions = model.predict(test_padded)
predicted_labels = np.argmax(predictions, axis=1)  # Chọn lớp có xác suất cao nhất cho mỗi mẫu


model.summary()

# Demo the predicted first 10 lines of test_data
for i in range(10):
    print("Sentence:", test_sentences[i])
    print("Predicted probability for each class (0 and 1):", predictions[i])


test_label = []
for pro in predictions:
  if pro[0] > pro[1] :
    test_label.append(0)
  else:
    test_label.append(1)
test_label = np.array(test_label)

# Đánh giá mô hình
loss, accuracy = model.evaluate(test_padded, test_labels)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)