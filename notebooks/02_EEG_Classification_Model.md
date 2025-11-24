# NeuroLab: EEG Mental State Classification Model

This notebook trains deep learning models for EEG mental state classification.

**Author:** NeuroLab Team  
**License:** MIT

---

## 1. Setup and Imports

```python
# Install required packages (uncomment if needed)
# !pip install tensorflow keras scikit-learn numpy pandas matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("✓ Imports successful")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
```

---

## 2. Load Dataset

```python
# Load the dataset (generated from notebook 01)
df = pd.read_csv('eeg_mental_states_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
df.head()
```

```python
# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# Check class distribution
print("\nClass distribution:")
print(df['state'].value_counts().sort_index())
```

---

## 3. Data Preprocessing

```python
# Separate features and target
feature_cols = ['alpha', 'beta', 'theta', 'delta', 'gamma']
X = df[feature_cols].values
y = df['state'].values

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
```

```python
# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")
```

```python
# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Reshape for LSTM (samples, timesteps, features)
X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_val_lstm = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

print(f"LSTM input shape: {X_train_lstm.shape}")
```

---

## 4. Build Models

### 4.1 Simple LSTM Model

```python
def build_lstm_model(input_shape, num_classes=3):
    model = keras.Sequential([
        layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

lstm_model = build_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
lstm_model.summary()
```

### 4.2 Bidirectional LSTM Model

```python
def build_bilstm_model(input_shape, num_classes=3):
    model = keras.Sequential([
        layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=input_shape),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

bilstm_model = build_bilstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
bilstm_model.summary()
```

### 4.3 CNN-LSTM Hybrid Model

```python
def build_cnn_lstm_model(input_shape, num_classes=3):
    model = keras.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, activation='relu'),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

cnn_lstm_model = build_cnn_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
cnn_lstm_model.summary()
```

---

## 5. Train Models

```python
# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

checkpoint = callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

callback_list = [early_stopping, reduce_lr, checkpoint]
```

### 5.1 Train LSTM Model

```python
print("Training LSTM Model...")
history_lstm = lstm_model.fit(
    X_train_lstm, y_train,
    validation_data=(X_val_lstm, y_val),
    epochs=50,
    batch_size=32, #  
     essed']))
 Strcused', ' 'Foaxed',mes=['Rel_nagetar    t                        
    st, y_pred, port(y_ten_reiocatssifi(cla   print")
 eport:ification Rt("\nClass   prinort
 ication Rep Classif #   
   f}")
 accuracy:.4nAccuracy: {t(f"\
    print, y_pred)score(y_tesracy_cy = accuaccura   
 yrac
    # Accu    =1)
roba, axisy_pred_p= np.argmax(pred     y_
_test)(Xpredictoba = model.ed_pr_pr
    ynsedictio   # Pr   
 '*60}")
 f"{'=  print(
  ion")at} Evalunamedel_print(f"{mo60}")
    '*{'=\n print(f":
   name), model_, y_testtest, X_odel_model(m evaluate`python
defdels

``e Mo. Evaluat
## 6

---
1
)
``` verbose=_list,
   backllbacks=call,
    cae=32   batch_siz0,
    epochs=5 y_val),
 X_val_lstm,n_data=(idatio
    valy_train,tm, ain_ls
    X_trodel.fit(lstm_mcnn_m = y_cnn_lst
histor)"del...LSTM MoCNN-ing ainnt("Tr`python
pri Model

``NN-LSTMTrain C3 ``

### 5.)
`
 verbose=1,
   allback_list=clbacks2,
    calch_size=3
    batepochs=50,,
    lstm, y_val)X_val_tion_data=(validain,
    m, y_tra_lstaint(
    X_trstm_model.fibilstm = bil
history_)"Model...ional LSTM idirectTraining Bint("
pr
```pythondel
LSTM Moional Bidirect# 5.2 Train 

##``ose=1
)
`
    verbst,ack_licks=callb
    callba