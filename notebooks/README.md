# NeuroLab Jupyter Notebooks

This directory contains Jupyter notebooks for the NeuroLab EEG Analysis project, ready to be uploaded to Kaggle.

## Notebooks

### 1. EEG Data Generation and Exploration
**File:** `01_EEG_Data_Generation_and_Exploration.ipynb`

- Generate synthetic EEG data for 3 mental states
- Explore frequency band distributions
- Visualize patterns and correlations
- Feature engineering
- Export dataset for training

### 2. EEG Mental State Classification Model
**File:** `02_EEG_Mental_State_Classification.ipynb`

- Load and preprocess EEG data
- Build deep learning models (LSTM, CNN-LSTM)
- Train and evaluate models
- Hyperparameter tuning
- Model comparison and selection
- Save best model

### 3. Model Evaluation and Interpretability
**File:** `03_Model_Evaluation_and_Interpretability.ipynb`

- Comprehensive model evaluation
- Confusion matrix and classification reports
- ROC curves and AUC scores
- SHAP values for feature importance
- LIME explanations
- Model calibration

### 4. Real-time EEG Analysis Demo
**File:** `04_Realtime_EEG_Analysis_Demo.ipynb`

- Load trained model
- Real-time prediction examples
- Temporal smoothing
- State duration calculation
- Visualization of predictions
- NLP-based recommendations

### 5. Voice Emotion Detection
**File:** `05_Voice_Emotion_Detection.ipynb`

- Audio processing basics
- Emotion detection from voice
- Feature extraction from audio
- Mental state mapping
- Multimodal analysis (EEG + Voice)

## Usage on Kaggle

### Upload Instructions

1. **Create a new Kaggle notebook**
2. **Upload the notebook file**
3. **Add required datasets:**
   - EEG Mental States Dataset (generated from notebook 1)
   - Pre-trained model (optional)

### Required Packages

Most packages are pre-installed on Kaggle. Additional packages needed:

```python
!pip install antropy mne pyedflib shap lime
```

### Running the Notebooks

1. **Start with Notebook 1** to generate the dataset
2. **Run Notebook 2** to train the model
3. **Use Notebook 3** for evaluation
4. **Try Notebook 4** for real-time demos
5. **Explore Notebook 5** for voice analysis

## Dataset

The notebooks generate and use the **EEG Mental States Dataset** with:
- **15,000+ samples** (5,000 per state)
- **5 frequency bands:** Alpha, Beta, Theta, Delta, Gamma
- **3 mental states:** Relaxed (0), Focused (1), Stressed (2)
- **Engineered features:** Ratios, percentages, total power

## Models

The notebooks train several model architectures:
- **LSTM** - Long Short-Term Memory networks
- **Bidirectional LSTM** - Bidirectional processing
- **CNN-LSTM** - Convolutional + LSTM hybrid
- **LSTM with Attention** - Attention mechanism

## Results

Expected performance:
- **Accuracy:** 85-95%
- **F1 Score:** 0.85-0.93
- **Training time:** 5-15 minutes (on Kaggle GPU)

## Citation

If you use these notebooks or the NeuroLab project, please cite:

```
@misc{neurolab2025,
  title={NeuroLab: EEG Mental State Classification Platform},
  author={NeuroLab Team},
  year={2024},
  url={https://github.com/neurolab-0x/ai.neurolab}
}
```

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
- GitHub: https://github.com/neurolab-0x/ai.neurolab
- Documentation: See project README.md

---

**Happy Learning! 🧠📊**
