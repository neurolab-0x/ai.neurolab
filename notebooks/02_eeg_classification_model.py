# %% [markdown]
# # NeuroLab: EEG Mental State Classification Model
# 
# This notebook trains a deep learning model to classify mental states from EEG data.
# 
# **Model Architecture:**
# - LSTM with Attention mechanism
# - Input: 5 EEG frequency bands (alpha, beta, theta, delta, gamma)
# - Output: 3 mental states (relaxed, focused, stressed)
# 
# **Author:** NeuroLab Team  
# **License:** MIT

# %% [markdown]
# ## 1. Setup and Imports

# %%
# Install required packages (uncomment if needed on Kaggle)
# !pip install tensorflow scikit-learn matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings
warnings.filterwarnings('ignore')

print("✓ Imports successful")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# %% [markdown]
# ## 2. Load and Prepare Data

# %%
# Load dataset (assuming it was generated in notebook 01)
try:
    df = pd.read_csv('eeg_mental_states_ation(n_ssifice_cla  X, y = maktion
  classificake_ import maasetsklearn.dat
    from sation demonstrle data forte sampnera  # Geirst.")
  tebook 01 fe run no Pleasund. not fo"⚠️  Dataset    print(ror:
dErunpt FileNotFocees")
examplf)} sen(d{lset loaded: f"✓ Data  print(v')
  dataset.cs