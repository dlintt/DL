import os
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt




# ==================================================
# ✅ PARAMETERS (reduced data size only)
# ==================================================
max_features = 5000
maxlen = 200
batch_size = 16        # smaller batches use less RAM
epochs = 2             # keep training shorter for testing
history_dict = {}

# ==================================================
# ✅ LOAD AND TRIM DATASET
# ==================================================
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Reduce dataset for low-RAM systems
x_train, y_train = x_train[:3000], y_train[:3000]
x_test, y_test = x_test[:800], y_test[:800]

# Pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# ==================================================
# ✅ BUILD AND TRAIN LSTM (unchanged)
# ==================================================
lstm_model = Sequential()
lstm_model.add(Embedding(max_features, 16))
lstm_model.add(LSTM(16))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

print("\n--- LSTM Training ---")
hist_lstm = lstm_model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
    verbose=1
)
history_dict['LSTM'] = hist_lstm.history

# ✅ Free memory between models
tf.keras.backend.clear_session()
gc.collect()

# ==================================================
# ✅ BUILD AND TRAIN GRU (unchanged)
# ==================================================
gru_model = Sequential()
gru_model.add(Embedding(max_features, 16))
gru_model.add(GRU(16))
gru_model.add(Dense(1, activation='sigmoid'))

gru_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

print("\n--- GRU Training ---")
hist_gru = gru_model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
    verbose=1
)
history_dict['GRU'] = hist_gru.history

# ==================================================
# ✅ EVALUATE MODELS
# ==================================================
loss_lstm, acc_lstm = lstm_model.evaluate(x_test, y_test, verbose=0)
loss_gru, acc_gru = gru_model.evaluate(x_test, y_test, verbose=0)

print("\n--- Test Results ---")
print(f"LSTM Test Loss: {loss_lstm:.4f}")
print(f"LSTM Test Accuracy: {acc_lstm:.4f}")
print(f"GRU Test Loss: {loss_gru:.4f}")
print(f"GRU Test Accuracy: {acc_gru:.4f}")

# ==================================================
# ✅ PLOT RESULTS (lightweight)
# ==================================================
plt.figure(figsize=(5, 3))
plt.plot(history_dict['LSTM']['loss'], label='LSTM Train Loss')
plt.plot(history_dict['LSTM']['val_loss'], label='LSTM Val Loss')
plt.title("LSTM Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 3))
plt.plot(history_dict['GRU']['loss'], label='GRU Train Loss', color='green')
plt.plot(history_dict['GRU']['val_loss'], label='GRU Val Loss', color='red')
plt.title("GRU Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()
