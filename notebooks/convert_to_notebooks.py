"""
Script to convert Python files to Jupyter notebooks
Run this to generate .ipynb files from the Python scripts

Usage:
    pip install jupytext
    python convert_to_notebooks.py
"""

import os
import subprocess

# List of Python files to convert
python_files = [
    '01_eeg_data_generation.py',
    '02_eeg_classification_model.py',
    '03_model_evaluation.py',
    '04_realtime_analysis.py',
    '05_voice_emotion_detection.py'
]

def convert_to_notebook(py_file):
    """Convert Python file to Jupyter notebook"""
    nb_file = py_file.replace('.py', '.ipynb')
    
    try:
        # Using jupytext
        subprocess.run(['jupytext', '--to', 'notebook', py_file], check=True)
        print(f"✓ Converted {py_file} -> {nb_file}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to convert {py_file}")
        return False
    except FileNotFoundError:
        print("✗ jupytext not found. Install with: pip install jupytext")
        return False

def main():
    print("="*60)
    print("Converting Python scripts to Jupyter notebooks")
    print("="*60)
    
    # Check if jupytext is installed
    try:
        subprocess.run(['jupytext', '--version'], capture_output=True, check=True)
    except FileNotFoundError:
        print("\n⚠️  jupytext is not installed")
        print("Install it with: pip install jupytext")
        print("\nAlternatively, you can:")
        print("1. Open Jupyter Lab/Notebook")
        print("2. Create a new notebook")
        print("3. Copy-paste code from the .py files")
        return
    
    success_count = 0
    not_found = []
    
    for py_file in python_files:
        if os.path.exists(py_file):
            if convert_to_notebook(py_file):
                success_count += 1
        else:
            not_found.append(py_file)
            print(f"⚠️  File not found: {py_file}")
    
    print("="*60)
    print(f"✓ Converted {success_count}/{len(python_files)} files")
    
    if not_found:
        print(f"\n⚠️  Files not found: {len(not_found)}")
        for file in not_found:
            print(f"   - {file}")
    
    print("="*60)

if __name__ == "__main__":
    main()
