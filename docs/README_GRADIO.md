# 🧠 NeuroLab Gradio Interface

A simplified web interface for testing the NeuroLab EEG Analysis Platform using Gradio.

## 🚀 Quick Start

### 1. Install Gradio

```bash
pip install gradio
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Gradio Interface

```bash
python gradio_app.py
```

The interface will be available at: `http://localhost:7860`

## 📋 Features

### 1. **Manual Input Tab** 📝
- Enter EEG frequency band values manually using sliders
- Adjust alpha, beta, theta, delta, and gamma values
- Get instant analysis results

### 2. **Sample Data Tab** 🎲
- Test with pre-generated sample data
- Choose from different mental states:
  - Relaxed
  - Focused
  - Stressed
  - Neutral
- See how the model classifies different patterns

### 3. **CSV Upload Tab** 📁
- Upload your own CSV file with EEG data
- Required columns: `alpha`, `beta`, `theta`, `delta`, `gamma`
- Supports multiple rows (uses mean values)

### 4. **Model Info Tab** ℹ️
- View information about the loaded model
- Check model status and architecture
- Learn about EEG frequency bands

## 📊 Output Information

Each analysis provides:

1. **Main Result**
   - Detected mental state (Relaxed/Focused/Stressed)
   - Confidence score
   - State distribution percentages

2. **Cognitive Metrics**
   - Attention Index
   - Relaxation Index
   - Stress Index
   - Cognitive Load
   - Mental Fatigue
   - Alertness

3. **Recommendations**
   - Personalized suggestions based on detected state
   - Health and wellness tips

4. **Raw JSON Output**
   - Complete analysis data in JSON format
   - Useful for debugging and integration

## 🎯 Use Cases

### Testing
- Quickly test the model with different input values
- Validate model behavior with known patterns
- Debug analysis pipeline

### Demonstration
- Show the platform capabilities to stakeholders
- Interactive demos for presentations
- Educational purposes

### Development
- Rapid prototyping of new features
- Testing data preprocessing
- Validating model predictions

## 📝 CSV File Format

Your CSV file should have the following structure:

```csv
alpha,beta,theta,delta,gamma
10.5,15.2,6.3,2.1,30.5
9.8,14.5,5.9,1.8,28.3
11.2,16.1,6.7,2.4,32.1
```

- **alpha**: Alpha wave power (8-13 Hz)
- **beta**: Beta wave power (13-30 Hz)
- **theta**: Theta wave power (4-8 Hz)
- **delta**: Delta wave power (0.5-4 Hz)
- **gamma**: Gamma wave power (25-100 Hz)

## 🔧 Configuration

### Port Configuration

To change the port, edit `gradio_app.py`:

```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,  # Change this
    share=False
)
```

### Share Publicly

To create a public link (for 72 hours):

```python
demo.launch(share=True)
```

## 🐛 Troubleshooting

### Model Not Loading

If you see "Model Not Loaded":

1. Train the model first:
   ```bash
   python train_model.py
   ```

2. Check that `processed/trained_model.h5` exists

### Port Already in Use

If port 7860 is busy:

```bash
# Use a different port
python gradio_app.py --port 7861
```

Or modify the port in `gradio_app.py`.

### Import Errors

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## 🎨 Customization

### Adding New Tabs

Edit `gradio_app.py` and add a new tab:

```python
with gr.Tab("🆕 New Feature"):
    gr.Markdown("### Your new feature")
    # Add your components here
```

### Changing Theme

Modify the theme in `create_interface()`:

```python
with gr.Blocks(theme=gr.themes.Glass()) as demo:
    # Available themes: Soft, Glass, Monochrome, Base
```

### Custom Styling

Add custom CSS:

```python
demo = gr.Blocks(css="""
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
""")
```

## 📚 API Integration

The Gradio interface uses the same ML processor as the REST API:

```python
from utils.ml_processor import MLProcessor

ml_processor = MLProcessor()
result = ml_processor.process_eeg_data(data)
```

This ensures consistency between the web UI and API endpoints.

## 🔐 Security Notes

- The Gradio interface is for **testing and development only**
- Do not expose it publicly without proper authentication
- For production use, use the FastAPI endpoints with JWT authentication

## 📞 Support

For issues or questions:
- Check the main [README.md](README.md)
- Review [PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)
- Open an issue on GitHub

## 🎉 Examples

### Example 1: Relaxed State
```
Alpha: 12 Hz (high)
Beta: 8 Hz (low)
Theta: 6 Hz (moderate)
Delta: 2 Hz (low)
Gamma: 28 Hz (moderate)
```

### Example 2: Focused State
```
Alpha: 6 Hz (low)
Beta: 22 Hz (high)
Theta: 4 Hz (low)
Delta: 1 Hz (low)
Gamma: 40 Hz (high)
```

### Example 3: Stressed State
```
Alpha: 5 Hz (low)
Beta: 28 Hz (very high)
Theta: 8 Hz (high)
Delta: 3 Hz (moderate)
Gamma: 45 Hz (very high)
```

---

**Happy Testing! 🧠✨**
