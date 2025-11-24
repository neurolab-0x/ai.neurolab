# %% [markdown]
# # NeuroLab: EEG Data Generation and Exploration
# 
# This notebook demonstrates how to generate synthetic EEG data for mental state classification.
# 
# **Mental States:**
# - 0: Relaxed (high alpha, low beta)
# - 1: Focused (high beta, moderate alpha)
# - 2: Stressed (very high beta, low alpha, elevated gamma)
# 
# **Author:** NeuroLab Team  
# **License:** MIT

# %% [markdown]
# ## 1. Setup and Imports

# %%
# Install required packages (uncomment if needed on Kaggle)
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
print(f"Pandas version: {pd.__version__}")

# %% [markdown]
# ## 2. Data Generation Functions

# %%
def generate_realistic_band_powers(state, num_samples=1000):
    """
    Generate realistic frequency band power values for different mental states.
    
    Args:
        state: Mental state ('relaxed', 'focused', or 'stressed')
        num_samples: Number of samples to generate
    
    Returns:
        List of sample dictionaries with frequency band powers
    """
    samples = []
    
    for _ in range(num_samples):
        if state == 'relaxed':
            # Relaxed state: High alpha rhythm
            alpha = np.random.uniform(15, 35)  # High alpha (8-13 Hz)
            beta = np.random.uniform(3, 12)    # Low beta (13-30 Hz)
            theta = np.random.uniform(5, 15)   # Moderate theta (4-8 Hz)
            delta = np.random.uniform(2, 8)    # Low delta (0.5-4 Hz)
            gamma = np.random.uniform(1, 5)    # Low gamma (30-45 Hz)
            
        elif state == 'focused':
            # Focused state: elevated beta, moderate alpha
            alpha = np.random.uniform(8, 20)   # Moderate alpha
            beta = np.random.uniform(15, 35)   # High beta - concentration
            theta = np.random.uniform(2, 8)    # Low theta
            delta = np.random.uniform(1, 5)    # Low delta
            gamma = np.random.uniform(5, 15)   # Moderate gamma
            
        elif state == 'stressed':
            # Stressed state: very high beta, low alpha
            alpha = np.random.uniform(3, 12)   # Low alpha - anxiety
            beta = np.random.uniform(25, 50)   # Very high beta - stress
            theta = np.random.uniform(8, 18)   # Elevated theta
            delta = np.random.uniform(3, 10)   # Moderate delta
            gamma = np.random.uniform(12, 30)  # High gamma - arousal
        
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

print("✓ Data generation functions defined")

# %% [markdown]
# ## 3. Generate Training Data

# %%
# Generate samples for each state
samples_per_state = 5000

print(f"Generating {samples_per_state} samples per state...")
print("="*60)

all_samples = []

# Relaxed state
print("Generating 'relaxed' state samples...")
all_samples.extend(generate_realistic_band_powers('relaxed', samples_per_state))

# Focused state
print("Generating 'focused' state samples...")
all_samples.extend(generate_realistic_band_powers('focused', samples_per_state))

# Stressed state
print("Generating 'stressed' state samples...")
all_samples.extend(generate_realistic_band_powers('stressed', samples_per_state))

# Convert to DataFrame
df = pd.DataFrame(all_samples)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("="*60)
print(f"✓ Total samples generated: {len(df)}")
print(f"\nState distribution:")
print(df['state'].value_counts().sort_index())

# %% [markdown]
# ## 4. Data Exploration

# %%
# Display first few rows
print("First 10 samples:")
df.head(10)

# %%
# Statistical summary
print("Statistical Summary:")
df.describe()

# %%
# Summary by state
print("Mean values by mental state:")
state_names = {0: 'Relaxed', 1: 'Focused', 2: 'Stressed'}
summary = df.groupby('state')[['alpha', 'beta', 'theta', 'delta', 'gamma']].mean()
summary.index = summary.index.map(state_names)
summary

# %% [markdown]
# ## 5. Visualizations

# %%
# Distribution of frequency bands by state
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

# Remove empty subplot
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()

# %%
# Box plots for each frequency band
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('EEG Frequency Bands Comparison Across Mental States', fontsize=16, fontweight='bold')

for idx, band in enumerate(bands):
    df_plot = df.copy()
    df_plot['state_name'] = df_plot['state'].map(state_names)
    
    sns.boxplot(data=df_plot, x='state_name', y=band, ax=axes[idx], palette='Set2')
    axes[idx].set_title(f'{band.capitalize()}', fontweight='bold')
    axes[idx].set_xlabel('Mental State')
    axes[idx].set_ylabel('Power (μV²)')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation = df[['alpha', 'beta', 'theta', 'delta', 'gamma']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of EEG Frequency Bands', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
# Radar chart for average band powers
from math import pi

categories = ['Alpha', 'Beta', 'Theta', 'Delta', 'Gamma']
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

for state in range(3):
    values = df[df['state'] == state][bands].mean().values.tolist()
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label=state_labels[state], color=colors[state])
    ax.fill(angles, values, alpha=0.25, color=colors[state])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=12)
ax.set_ylim(0, 30)
ax.set_title('Average EEG Band Powers by Mental State', size=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Feature Engineering

# %%
# Create additional features
df['alpha_beta_ratio'] = df['alpha'] / df['beta']
df['theta_beta_ratio'] = df['theta'] / df['beta']
df['total_power'] = df['alpha'] + df['beta'] + df['theta'] + df['delta'] + df['gamma']
df['alpha_percentage'] = (df['alpha'] / df['total_power']) * 100
df['beta_percentage'] = (df['beta'] / df['total_power']) * 100

print("✓ Additional features created")
print("\nNew features:")
print(df[['alpha_beta_ratio', 'theta_beta_ratio', 'total_power', 
          'alpha_percentage', 'beta_percentage']].head())

# %%
# Visualize engineered features
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Engineered Features by Mental State', fontsize=16, fontweight='bold')

df_plot = df.copy()
df_plot['state_name'] = df_plot['state'].map(state_names)

sns.boxplot(data=df_plot, x='state_name', y='alpha_beta_ratio', ax=axes[0], palette='Set2')
axes[0].set_title('Alpha/Beta Ratio', fontweight='bold')
axes[0].set_ylabel('Ratio')

sns.boxplot(data=df_plot, x='state_name', y='theta_beta_ratio', ax=axes[1], palette='Set2')
axes[1].set_title('Theta/Beta Ratio', fontweight='bold')
axes[1].set_ylabel('Ratio')

sns.boxplot(data=df_plot, x='state_name', y='total_power', ax=axes[2], palette='Set2')
axes[2].set_title('Total Power', fontweight='bold')
axes[2].set_ylabel('Power (μV²)')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Save Dataset

# %%
# Save to CSV
output_file = 'eeg_mental_states_dataset.csv'
df.to_csv(output_file, index=False)

print(f"✓ Dataset saved to: {output_file}")
print(f"  Total samples: {len(df)}")
print(f"  Features: {len(df.columns)}")
print(f"  File size: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

# %% [markdown]
# ## 8. Summary Statistics

# %%
print("="*60)
print("DATASET SUMMARY")
print("="*60)
print(f"\nTotal Samples: {len(df):,}")
print(f"Features: {len(df.columns)}")
print(f"\nClass Distribution:")
for state, count in df['state'].value_counts().sort_index().items():
    percentage = (count / len(df)) * 100
    print(f"  {state_names[state]}: {count:,} ({percentage:.1f}%)")

print(f"\nFeature List:")
for col in df.columns:
    print(f"  - {col}")

print("\n" + "="*60)
print("✓ Data generation and exploration complete!")
print("="*60)
