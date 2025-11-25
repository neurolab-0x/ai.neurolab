"""
Script to prepare NeuroLab for Hugging Face Spaces deployment
"""
import os
import shutil
from pathlib import Path

def create_hf_space_structure():
    """Create directory structure for Hugging Face Space"""
    
    # Create base directory
    base_dir = Path("nlpt_2-preview")
    base_dir.mkdir(exist_ok=True)
    
    print(f"📁 Creating Hugging Face Space structure in: {base_dir.absolute()}")
    
    # Directories to copy
    dirs_to_copy = ['src', 'utils', 'core', 'preprocessing', 'api', 'config', 'scripts']
    
    for dir_name in dirs_to_copy:
        src_dir = Path(dir_name)
        if src_dir.exists():
            dst_dir = base_dir / dir_name
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            print(f"✅ Copied {dir_name}/")
        else:
            print(f"⚠️  {dir_name}/ not found, skipping...")
    
    # Copy model files if they exist
    model_dirs = [
        ('model', 'trained_model.h5'),
        ('model', 'voice_emotion_model.h5')
    ]
    
    for dir_name, model_file in model_dirs:
        src_path = Path(dir_name) / model_file
        if src_path.exists():
            dst_dir = base_dir / dir_name
            dst_dir.mkdir(exist_ok=True)
            shutil.copy(src_path, dst_dir / model_file)
            print(f"✅ Copied {dir_name}/{model_file}")
        else:
            print(f"⚠️  {dir_name}/{model_file} not found")
    
    # Create requirements.txt
    requirements = """gradio>=4.0.0
numpy>=1.21.0
pandas>=1.3.0
tensorflow>=2.12.0
scikit-learn>=1.0.0
librosa>=0.10.0
soundfile>=0.12.0
scipy>=1.7.0
pydantic>=2.0.0
python-multipart>=0.0.6
"""
    
    with open(base_dir / "requirements.txt", "w") as f:
        f.write(requirements)
    print("✅ Created requirements.txt")
    
    # Create README.md
    readme = """---
title: NeuroLab EEG Analysis
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# NeuroLab: EEG & Voice Analysis Platform

Analyze EEG data to detect mental states: **Relaxed**, **Focused**, or **Stressed**

## Features
- 📝 Manual EEG input with sliders
- 🎲 Sample data generation
- 📁 CSV file upload
- 💡 AI-powered recommendations

## Usage
1. Choose an input method (Manual, Sample, or CSV)
2. Enter or upload your EEG data
3. Click "Analyze" to get results
4. View mental state classification and recommendations

## Model
- **Input**: 5 EEG frequency bands (alpha, beta, theta, delta, gamma)
- **Output**: 3 mental states (relaxed, focused, stressed)
- **Architecture**: CNN with BatchNormalization

## About
Built by the NeuroLab Team for mental state monitoring and neurofeedback applications.

Repository: [GitHub](https://github.com/neurolab-0x/ai.neurolab)
"""
    
    with open(base_dir / "README.md", "w") as f:
        f.write(readme)
    print("✅ Created README.md")
    
    # Create simplified app.py
    app_code = '''"""
NeuroLab Gradio Interface for Hugging Face Spaces
"""
import gradio as gr
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core components (with error handling)
try:
    from utils.ml_processor import MLProcessor
    ml_processor = MLProcessor()
    MODEL_AVAILABLE = True
except Exception as e:
    logger.warning(f"Could not load ML processor: {e}")
    MODEL_AVAILABLE = False
    ml_processor = None

def generate_sample_eeg_data(state: str = "relaxed") -> Dict[str, float]:
    """Generate sample EEG data for testing"""
    if state.lower() == "relaxed":
        return {
            "alpha": np.random.uniform(8, 13),
            "beta": np.random.uniform(5, 10),
            "theta": np.random.uniform(4, 8),
            "delta": np.random.uniform(0.5, 4),
            "gamma": np.random.uniform(25, 35)
        }
    elif state.lower() == "focused":
        return {
            "alpha": np.random.uniform(5, 8),
            "beta": np.random.uniform(15, 25),
            "theta": np.random.uniform(3, 6),
            "delta": np.random.uniform(0.5, 3),
            "gamma": np.random.uniform(30, 45)
        }
    elif state.lower() == "stressed":
        return {
            "alpha": np.random.uniform(4, 7),
            "beta": np.random.uniform(20, 30),
            "theta": np.random.uniform(6, 10),
            "delta": np.random.uniform(1, 4),
            "gamma": np.random.uniform(35, 50)
        }
    else:
        return {
            "alpha": np.random.uniform(6, 12),
            "beta": np.random.uniform(10, 20),
            "theta": np.random.uniform(4, 8),
            "delta": np.random.uniform(1, 4),
            "gamma": np.random.uniform(25, 40)
        }

def analyze_eeg(alpha, beta, theta, delta, gamma):
    """Analyze EEG data"""
    try:
        data = {
            "alpha": alpha,
            "beta": beta,
            "theta": theta,
            "delta": delta,
            "gamma": gamma
        }
        
        if MODEL_AVAILABLE and ml_processor:
            result = ml_processor.process_eeg_data(
                data,
                subject_id="hf_user",
                session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Format results
            state_label = result.get('state_label', 'unknown')
            confidence = result.get('confidence', 0)
            
            output = f"""
## 🧠 Analysis Result

**Mental State:** {state_label.upper()}  
**Confidence:** {confidence:.1f}%

### Input Values
- Alpha: {alpha:.2f} Hz
- Beta: {beta:.2f} Hz
- Theta: {theta:.2f} Hz
- Delta: {delta:.2f} Hz
- Gamma: {gamma:.2f} Hz
"""
            
            # Add recommendations if available
            if 'recommendations' in result:
                output += "\\n### 💡 Recommendations\\n"
                for i, rec in enumerate(result['recommendations'][:3], 1):
                    output += f"{i}. {rec}\\n"
            
            return output
        else:
            # Fallback rule-based classification
            if alpha > 10 and beta < 15:
                state = "RELAXED"
            elif beta > 20:
                state = "STRESSED"
            else:
                state = "FOCUSED"
                
            return f"""
## 🧠 Analysis Result (Rule-based)

**Mental State:** {state}  
**Note:** Using rule-based classification (model not available)

### Input Values
- Alpha: {alpha:.2f} Hz
- Beta: {beta:.2f} Hz
- Theta: {theta:.2f} Hz
- Delta: {delta:.2f} Hz
- Gamma: {gamma:.2f} Hz
"""
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="NeuroLab EEG Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 NeuroLab EEG Analysis Platform
    
    Analyze EEG data to detect mental states: **Relaxed**, **Focused**, or **Stressed**
    """)
    
    with gr.Tabs():
        with gr.Tab("📝 Manual Input"):
            gr.Markdown("### Enter EEG frequency band values")
            
            with gr.Row():
                with gr.Column():
                    alpha = gr.Slider(0, 50, value=10, label="Alpha (8-13 Hz)", info="Relaxation")
                    beta = gr.Slider(0, 50, value=15, label="Beta (13-30 Hz)", info="Focus/Anxiety")
                    theta = gr.Slider(0, 50, value=6, label="Theta (4-8 Hz)", info="Drowsiness")
                    delta = gr.Slider(0, 50, value=2, label="Delta (0.5-4 Hz)", info="Deep sleep")
                    gamma = gr.Slider(0, 100, value=30, label="Gamma (25-100 Hz)", info="Processing")
                    
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                
                with gr.Column():
                    output = gr.Markdown()
            
            analyze_btn.click(
                fn=analyze_eeg,
                inputs=[alpha, beta, theta, delta, gamma],
                outputs=output
            )
        
        with gr.Tab("🎲 Sample Data"):
            gr.Markdown("### Test with pre-generated data")
            
            state_choice = gr.Radio(
                choices=["Relaxed", "Focused", "Stressed"],
                value="Relaxed",
                label="Select Mental State"
            )
            sample_btn = gr.Button("🎲 Generate & Analyze", variant="primary")
            sample_output = gr.Markdown()
            
            def analyze_sample(state):
                data = generate_sample_eeg_data(state)
                return analyze_eeg(**data)
            
            sample_btn.click(
                fn=analyze_sample,
                inputs=state_choice,
                outputs=sample_output
            )
    
    gr.Markdown("""
    ---
    ### 📖 About
    
    **NeuroLab** uses machine learning to classify mental states from EEG data.
    
    - **Model:** CNN with BatchNormalization
    - **Input:** 5 EEG frequency bands
    - **Output:** 3 mental states
    
    Built with ❤️ by the NeuroLab Team
    """)

if __name__ == "__main__":
    demo.launch()
'''
    
    with open(base_dir / "app.py", "w") as f:
        f.write(app_code)
    print("✅ Created app.py")
    
    # Create .gitignore
    gitignore = """__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
*.log
.DS_Store
"""
    
    with open(base_dir / ".gitignore", "w") as f:
        f.write(gitignore)
    print("✅ Created .gitignore")
    
    print("\n" + "="*60)
    print("✨ Hugging Face Space structure created successfully!")
    print("="*60)
    print(f"\n📂 Location: {base_dir.absolute()}")
    print("\n📝 Next steps:")
    print("1. cd neurolab-hf-space")
    print("2. git init")
    print("3. git add .")
    print('4. git commit -m "Initial commit"')
    print("5. Create a Space on huggingface.co/spaces")
    print("6. git remote add origin https://huggingface.co/spaces/neurolab-0x/nlpt_2-preview")
    print("7. git push -u origin main")
    print("\n💡 Or use the Hugging Face CLI:")
    print("   huggingface-cli login")
    print("   huggingface-cli repo create SPACE_NAME --type space --space_sdk gradio")
    print("\n📖 See docs/HUGGINGFACE_DEPLOYMENT.md for detailed instructions")

if __name__ == "__main__":
    create_hf_space_structure()
