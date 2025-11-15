"""
Gradio Interface for NeuroLab EEG Analysis Platform
Provides a simple web UI for testing EEG analysis functionality
"""

import gradio as gr
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
import os

# Import core components
from utils.ml_processor import MLProcessor
from core.ml.model import load_calibrated_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ML Processor
ml_processor = MLProcessor()

# ============================================================
# Helper Functions
# ============================================================

def generate_sample_eeg_data(state: str = "relaxed") -> Dict[str, float]:
    """Generate sample EEG data for testing"""
    if state == "relaxed":
        return {
            "alpha": np.random.uniform(8, 13),
            "beta": np.random.uniform(5, 10),
            "theta": np.random.uniform(4, 8),
            "delta": np.random.uniform(0.5, 4),
            "gamma": np.random.uniform(25, 35)
        }
    elif state == "focused":
        return {
            "alpha": np.random.uniform(5, 8),
            "beta": np.random.uniform(15, 25),
            "theta": np.random.uniform(3, 6),
            "delta": np.random.uniform(0.5, 3),
            "gamma": np.random.uniform(30, 45)
        }
    elif state == "stressed":
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

def format_results(result: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Format analysis results for display"""
    try:
        # Main prediction
        state_label = result.get('state_label', 'unknown')
        confidence = result.get('confidence', 0)
        
        # State percentages
        state_percentages = result.get('state_percentages', {})
        
        # Cognitive metrics
        cognitive_metrics = result.get('cognitive_metrics', {})
        
        # Recommendations
        recommendations = result.get('recommendations', [])
        
        # Format main result
        main_result = f"""
## 🧠 EEG Analysis Result

**Detected State:** {state_label.upper()}  
**Confidence:** {confidence:.1f}%

### State Distribution
- Relaxed: {state_percentages.get(0, 0):.1f}%
- Focused: {state_percentages.get(1, 0):.1f}%
- Stressed: {state_percentages.get(2, 0):.1f}%
"""
        
        # Format cognitive metrics
        metrics_text = "## 📊 Cognitive Metrics\n\n"
        if cognitive_metrics:
            metrics_text += f"- **Attention Index:** {cognitive_metrics.get('attention_index', 0):.3f}\n"
            metrics_text += f"- **Relaxation Index:** {cognitive_metrics.get('relaxation_index', 0):.3f}\n"
            metrics_text += f"- **Stress Index:** {cognitive_metrics.get('stress_index', 0):.3f}\n"
            metrics_text += f"- **Cognitive Load:** {cognitive_metrics.get('cognitive_load', 0):.3f}\n"
            metrics_text += f"- **Mental Fatigue:** {cognitive_metrics.get('mental_fatigue', 0):.3f}\n"
            metrics_text += f"- **Alertness:** {cognitive_metrics.get('alertness', 0):.3f}\n"
        else:
            metrics_text += "No metrics available"
        
        # Format recommendations
        rec_text = "## 💡 Recommendations\n\n"
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                rec_text += f"{i}. {rec}\n"
        else:
            rec_text += "No recommendations available"
        
        # Format raw JSON
        raw_json = json.dumps(result, indent=2, default=str)
        
        return main_result, metrics_text, rec_text, raw_json
        
    except Exception as e:
        logger.error(f"Error formatting results: {str(e)}")
        return f"Error: {str(e)}", "", "", ""

# ============================================================
# Gradio Interface Functions
# ============================================================

def analyze_manual_input(alpha: float, beta: float, theta: float, delta: float, gamma: float) -> Tuple[str, str, str, str]:
    """Analyze manually entered EEG values"""
    try:
        # Create data dictionary
        data = {
            "alpha": alpha,
            "beta": beta,
            "theta": theta,
            "delta": delta,
            "gamma": gamma
        }
        
        # Process data
        result = ml_processor.process_eeg_data(
            data,
            subject_id="gradio_user",
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return format_results(result)
        
    except Exception as e:
        logger.error(f"Error in manual analysis: {str(e)}")
        return f"Error: {str(e)}", "", "", ""

def analyze_sample_data(state: str) -> Tuple[str, str, str, str]:
    """Analyze pre-generated sample data"""
    try:
        # Generate sample data
        data = generate_sample_eeg_data(state.lower())
        
        # Process data
        result = ml_processor.process_eeg_data(
            data,
            subject_id="gradio_user",
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return format_results(result)
        
    except Exception as e:
        logger.error(f"Error in sample analysis: {str(e)}")
        return f"Error: {str(e)}", "", "", ""

def analyze_csv_file(file) -> Tuple[str, str, str, str]:
    """Analyze EEG data from uploaded CSV file"""
    try:
        if file is None:
            return "Please upload a CSV file", "", "", ""
        
        # Read CSV file
        df = pd.read_csv(file.name)
        
        # Check for required columns
        required_cols = ['alpha', 'beta', 'theta', 'delta', 'gamma']
        if not all(col in df.columns for col in required_cols):
            return f"CSV must contain columns: {', '.join(required_cols)}", "", "", ""
        
        # Process first row (or aggregate if multiple rows)
        if len(df) > 1:
            # Use mean values if multiple rows
            data = {
                "alpha": float(df['alpha'].mean()),
                "beta": float(df['beta'].mean()),
                "theta": float(df['theta'].mean()),
                "delta": float(df['delta'].mean()),
                "gamma": float(df['gamma'].mean())
            }
        else:
            data = df.iloc[0][required_cols].to_dict()
        
        # Process data
        result = ml_processor.process_eeg_data(
            data,
            subject_id="gradio_user",
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return format_results(result)
        
    except Exception as e:
        logger.error(f"Error in CSV analysis: {str(e)}")
        return f"Error: {str(e)}", "", "", ""

def get_model_info() -> str:
    """Get information about the loaded model"""
    try:
        status = ml_processor.get_status()
        
        info = f"""
## 🤖 Model Information

**Model Status:** {'✅ Loaded' if status['model_loaded'] else '❌ Not Loaded'}  
**Model Path:** {status['model_path']}  
**Model Exists:** {'Yes' if status['model_exists'] else 'No'}  
**Model Type:** {status['model_type']}

### Model Architecture
- Input: 5 EEG frequency bands (alpha, beta, theta, delta, gamma)
- Output: 3 mental states (relaxed, focused, stressed)
- Architecture: CNN with BatchNormalization and Dense layers

### Frequency Bands
- **Alpha (8-13 Hz):** Relaxation, calmness
- **Beta (13-30 Hz):** Active thinking, focus, anxiety
- **Theta (4-8 Hz):** Drowsiness, meditation
- **Delta (0.5-4 Hz):** Deep sleep
- **Gamma (25-100 Hz):** High-level information processing
"""
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return f"Error: {str(e)}"

# ============================================================
# Create Gradio Interface
# ============================================================

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="NeuroLab EEG Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧠 NeuroLab EEG Analysis Platform
        
        Analyze EEG data to detect mental states: **Relaxed**, **Focused**, or **Stressed**
        
        Choose one of the methods below to analyze EEG data.
        """)
        
        with gr.Tabs():
            # Tab 1: Manual Input
            with gr.Tab("📝 Manual Input"):
                gr.Markdown("### Enter EEG frequency band values manually")
                
                with gr.Row():
                    with gr.Column():
                        alpha_input = gr.Slider(0, 50, value=10, label="Alpha (8-13 Hz)", info="Relaxation")
                        beta_input = gr.Slider(0, 50, value=15, label="Beta (13-30 Hz)", info="Focus/Anxiety")
                        theta_input = gr.Slider(0, 50, value=6, label="Theta (4-8 Hz)", info="Drowsiness")
                        delta_input = gr.Slider(0, 50, value=2, label="Delta (0.5-4 Hz)", info="Deep sleep")
                        gamma_input = gr.Slider(0, 100, value=30, label="Gamma (25-100 Hz)", info="Processing")
                        
                        analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                    
                    with gr.Column():
                        result_output = gr.Markdown(label="Analysis Result")
                
                with gr.Row():
                    metrics_output = gr.Markdown(label="Cognitive Metrics")
                    recommendations_output = gr.Markdown(label="Recommendations")
                
                with gr.Accordion("📄 Raw JSON Output", open=False):
                    json_output = gr.Code(language="json", label="Raw Data")
                
                analyze_btn.click(
                    fn=analyze_manual_input,
                    inputs=[alpha_input, beta_input, theta_input, delta_input, gamma_input],
                    outputs=[result_output, metrics_output, recommendations_output, json_output]
                )
            
            # Tab 2: Sample Data
            with gr.Tab("🎲 Sample Data"):
                gr.Markdown("### Test with pre-generated sample data")
                
                with gr.Row():
                    with gr.Column():
                        state_dropdown = gr.Dropdown(
                            choices=["Relaxed", "Focused", "Stressed", "Neutral"],
                            value="Relaxed",
                            label="Select Mental State"
                        )
                        sample_btn = gr.Button("🎲 Generate & Analyze", variant="primary")
                    
                    with gr.Column():
                        sample_result = gr.Markdown(label="Analysis Result")
                
                with gr.Row():
                    sample_metrics = gr.Markdown(label="Cognitive Metrics")
                    sample_recommendations = gr.Markdown(label="Recommendations")
                
                with gr.Accordion("📄 Raw JSON Output", open=False):
                    sample_json = gr.Code(language="json", label="Raw Data")
                
                sample_btn.click(
                    fn=analyze_sample_data,
                    inputs=[state_dropdown],
                    outputs=[sample_result, sample_metrics, sample_recommendations, sample_json]
                )
            
            # Tab 3: CSV Upload
            with gr.Tab("📁 CSV Upload"):
                gr.Markdown("""
                ### Upload a CSV file with EEG data
                
                CSV file should contain columns: `alpha`, `beta`, `theta`, `delta`, `gamma`
                
                If multiple rows are present, the mean values will be used.
                """)
                
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
                        upload_btn = gr.Button("📤 Upload & Analyze", variant="primary")
                    
                    with gr.Column():
                        csv_result = gr.Markdown(label="Analysis Result")
                
                with gr.Row():
                    csv_metrics = gr.Markdown(label="Cognitive Metrics")
                    csv_recommendations = gr.Markdown(label="Recommendations")
                
                with gr.Accordion("📄 Raw JSON Output", open=False):
                    csv_json = gr.Code(language="json", label="Raw Data")
                
                upload_btn.click(
                    fn=analyze_csv_file,
                    inputs=[file_input],
                    outputs=[csv_result, csv_metrics, csv_recommendations, csv_json]
                )
            
            # Tab 4: Model Info
            with gr.Tab("ℹ️ Model Info"):
                gr.Markdown("### Information about the loaded model")
                
                model_info_btn = gr.Button("🔄 Refresh Model Info")
                model_info_output = gr.Markdown()
                
                # Load model info on tab open
                demo.load(fn=get_model_info, outputs=[model_info_output])
                model_info_btn.click(fn=get_model_info, outputs=[model_info_output])
        
        gr.Markdown("""
        ---
        ### 📖 About
        
        **NeuroLab** is an EEG analysis platform that uses machine learning to classify mental states.
        
        - **Model:** CNN with BatchNormalization
        - **Input:** 5 EEG frequency bands
        - **Output:** 3 mental states (relaxed, focused, stressed)
        - **Confidence:** Prediction confidence score
        
        For more information, visit the [documentation](https://github.com/your-repo).
        """)
    
    return demo

# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Main entry point for Gradio app"""
    try:
        logger.info("Starting NeuroLab Gradio Interface...")
        
        # Create and launch interface
        demo = create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Error starting Gradio interface: {str(e)}")
        raise

if __name__ == "__main__":
    main()
