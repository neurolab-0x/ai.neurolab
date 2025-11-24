"""
Script to create Jupyter notebooks for NeuroLab project
"""

import json
import os

def create_notebook(cells, filename):
    """Create a Jupyter notebook from cells"""
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    os.makedirs('notebooks', exist_ok=True)
    filepath = os.path.join('notebooks', filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✓ Created: {filepath}")

def markdown_cell(text):
    """Create a markdown cell"""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.split('\n')
    }

def code_cell(code):
    """Create a code cell"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split('\n')
    }

# Notebook 1: EEG Data Generation
notebook1_cells = [
    markdown_cell("""# NeuroLab: EEG Data Generation and Exploration

This notebook demonstrates how to generate synthetic EEG data for mental state classification.

**Mental States:**
- 0: Relaxed (high alpha, low beta)
- 1: Focused (high beta, moderate alpha)
- 2: Stressed (very high beta, low alpha, elevated gamma)

**Author:** NeuroLab Team  
**License:** MIT"""),
    
    markdown_cell("## 1. Setup and Imports"),
    
    code_cell("""# Install required packages (uncomment if needed on Kaggle)
# !pip install numpy pandas matplotlib seaborn scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ Imports successful")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")"""),
    
    markdown_cell("## 2. Data Generation Functions"),
    
    code_cell("""def generate_realistic_band_powers(state, num_samples=1000):
    \"\"\"
    Generate realistic frequency band power values for different mental states.
    
    Args:
        state: Mental state ('relaxed', 'focused', or 'stressed')
        num_samples: Number of samples to generate
    
    Returns:
        List of sample dictionaries with frequency band powers
    \"\"\"
    samples = []
    
    for _ in range(num_samples):
        if state == 'relaxed':
            # Relaxed state: High alpha rhythm
            alpha = np.random.uniform(15, 35)
            beta = np.random.uniform(3, 12)
            theta = np.random.uniform(5, 15)
            delta = np.random.uniform(2, 8)
            gamma = np.random.uniform(1, 5)
            
        elif state == 'focused':
            # Focused state: elevated beta
            alpha = np.random.uniform(8, 20)
            beta = np.random.uniform(15, 35)
            theta = np.random.uniform(2, 8)
            delta = np.random.uniform(1, 5)
            gamma = np.random.uniform(5, 15)
            
        elif state == 'stressed':
            # Stressed state: very high beta
            alpha = np.random.uniform(3, 12)
            beta = np.random.uniform(25, 50)
            theta = np.random.uniform(8, 18)
            delta = np.random.uniform(3, 10)
            gamma = np.random.uniform(12, 30)
        
        # Add natural variation
        alpha += np.random.normal(0, 2)
        beta += np.random.normal(0, 3)
        theta += np.random.normal(0, 2)
        delta += np.random.normal(0, 2)
        gamma += np.random.normal(0, 2)
        
        # Ensure positive values
        alpha = max(0.1, alpha)
        beta = max(0.1, beta)
        theta = max(0.1, theta)
        delta = max(0.1, delta)
        gamma = max(0.1, gamma)
        
        samples.append({
            'alpha': alpha,
            'beta': beta,
            'theta': theta,
            'delta': delta,
            'gamma': gamma,
            'state': 0 if state == 'relaxed' else (1 if state == 'focused' else 2)
        })
    
    return samples

print("✓ Data generation functions defined")"""),
    
    markdown_cell("## 3. Generate Training Data"),
    
    code_cell("""# Generate samples for each state
samples_per_state = 5000

print(f"Generating {samples_per_state} samples per state...")
print("="*60)

all_samples = []

print("Generating 'relaxed' state samples...")
all_samples.extend(generate_realistic_band_powers('relaxed', samples_per_state))

print("Generating 'focused' state samples...")
all_samples.extend(generate_realistic_band_powers('focused', samples_per_state))

print("Generating 'stressed' state samples...")
all_samples.extend(generate_realistic_band_powers('stressed', samples_per_state))

# Convert to DataFrame
df = pd.DataFrame(all_samples)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("="*60)
print(f"✓ Total samples generated: {len(df)}")
print(f"\\nState distribution:")
print(df['state'].value_counts().sort_index())"""),
    
    markdown_cell("## 4. Data Exploration"),
    
    code_cell("""# Display first few rows
print("First 10 samples:")
df.head(10)"""),
    
    code_cell("""# Statistical summary
df.describe()"""),
    
    code_cell("""# Summary by state
state_names = {0: 'Relaxed', 1: 'Focused', 2: 'Stressed'}
summary = df.groupby('state')[['alpha', 'beta', 'theta', 'delta', 'gamma']].mean()
summary.index = summary.index.map(state_names)
summary"""),
    
    markdown_cell("## 5. Visualizations"),
    
    code_cell("""# Distribution of frequency bands by state
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('EEG Frequency Band Distributions by Mental State', fontsize=16, fontweight='bold')

bands = ['alpha', 'beta', 'theta', 'delta', 'gamma']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
state_labels = ['Relaxed', 'Focused', 'Stressed']

for idx, band in enumerate(bands):
    ax = axes[idx // 3, idx % 3]
    for state in range(3):
        data = df[df['state'] == state][band]
        ax.hist(data, bins=50, alpha=0.6, label=state_labels[state], color=colors[state])
    ax.set_title(f'{band.capitalize()} Band', fontsize=12, fontweight='bold')
    ax.set_xlabel('Power (μV²)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.delaxes(axes[1, 2])
plt.tight_layout()
plt.show()"""),
    
    code_cell("""# Box plots
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('EEG Frequency Bands Comparison', fontsize=16, fontweight='bold')

for idx, band in enumerate(bands):
    df_plot = df.copy()
    df_plot['state_name'] = df_plot['state'].map(state_names)
    sns.boxplot(data=df_plot, x='state_name', y=band, ax=axes[idx], palette='Set2')
    axes[idx].set_title(f'{band.capitalize()}', fontweight='bold')
    axes[idx].set_xlabel('Mental State')
    axes[idx].set_ylabel('Power (μV²)')

plt.tight_layout()
plt.show()"""),
    
    code_cell("""# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation = df[['alpha', 'beta', 'theta', 'delta', 'gamma']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix of EEG Frequency Bands', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""),
    
    markdown_cell("## 6. Feature Engineering"),
    
    code_cell("""# Create additional features
df['alpha_beta_ratio'] = df['alpha'] / df['beta']
df['theta_beta_ratio'] = df['theta'] / df['beta']
df['total_power'] = df['alpha'] + df['beta'] + df['theta'] + df['delta'] + df['gamma']
df['alpha_percentage'] = (df['alpha'] / df['total_power']) * 100
df['beta_percentage'] = (df['beta'] / df['total_power']) * 100

print("✓ Additional features created")
df[['alpha_beta_ratio', 'theta_beta_ratio', 'total_power']].head()"""),
    
    markdown_cell("## 7. Save Dataset"),
    
    code_cell("""# Save to CSV
output_file = 'eeg_mental_states_dataset.csv'
df.to_csv(output_file, index=False)

print(f"✓ Dataset saved to: {output_file}")
print(f"  Total samples: {len(df)}")
print(f"  Features: {len(df.columns)}")"""),
]

create_notebook(notebook1_cells, '01_EEG_Data_Generation_and_Exploration.ipynb')

print("\n✓ Notebook 1 created successfully!")
