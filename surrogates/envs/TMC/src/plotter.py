import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Academic Style Configuration
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['savefig.dpi'] = 120

class TMCPlotter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Academic color palette
        self.colors = ['#00008B', '#800080', '#8B0000'] # DarkBlue, Purple, DarkRed

    def plot_parity(self, true_vals, pred_vals, title, filename, focus_range=None):
        """Plot true vs predicted values."""
        plt.figure(figsize=(6, 5))
        
        # Calculate metrics
        r2 = r2_score(true_vals, pred_vals)
        mae = mean_absolute_error(true_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        
        # Scatter plot with hexbin for density if many points
        if len(true_vals) > 10000:
            plt.hexbin(true_vals, pred_vals, gridsize=50, cmap='Purples', mincnt=1, alpha=0.8)
            plt.colorbar(label='Count')
        else:
            plt.scatter(true_vals, pred_vals, alpha=0.5, color=self.colors[0], s=10)
            
        # Identity line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Identity')
        
        plt.xlabel('Ground Truth Polarisability', fontsize=12)
        plt.ylabel('Predicted Polarisability', fontsize=12)
        plt.title(title, fontsize=14)
        
        # Add metrics text box
        stats_text = f'$R^2$: {r2:.3f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}'
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename + '.pdf'), dpi=120)
        plt.close()

    def plot_error_distribution(self, true_vals, pred_vals, title, filename):
        """Plot distribution of residuals."""
        errors = pred_vals - true_vals
        plt.figure(figsize=(6, 5))
        
        sns.histplot(errors, kde=True, color=self.colors[1], bins=100)
        plt.axvline(0, color='black', linestyle='--')
        
        plt.xlabel('Error (Predicted - True)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(title, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename + '.pdf'), dpi=120)
        plt.close()
