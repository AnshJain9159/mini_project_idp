"""Exploratory Data Analysis module."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logger import get_logger

logger = get_logger(__name__)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class EDAAnalyzer:
    """Performs exploratory data analysis."""

    def __init__(self, df: pd.DataFrame, target_col: str = 'Outcome'):
        """
        Initialize EDA analyzer.

        Args:
            df: DataFrame to analyze
            target_col: Name of target column
        """
        self.df = df.copy()
        self.target_col = target_col
        logger.info(f"EDAAnalyzer initialized with {len(df)} samples")

    def generate_summary_statistics(self) -> pd.DataFrame:
        """
        Generate summary statistics.

        Returns:
            DataFrame containing summary statistics
        """
        logger.info("Generating summary statistics")
        summary = self.df.describe().T
        summary['missing'] = self.df.isnull().sum()
        summary['missing_pct'] = (self.df.isnull().sum() / len(self.df) * 100).round(2)
        summary['zeros'] = (self.df == 0).sum()
        summary['zeros_pct'] = ((self.df == 0).sum() / len(self.df) * 100).round(2)

        return summary

    def plot_target_distribution(self, output_path: Optional[str] = None):
        """
        Plot target variable distribution.

        Args:
            output_path: Path to save the plot
        """
        logger.info("Plotting target distribution")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Count plot
        self.df[self.target_col].value_counts().plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'])
        axes[0].set_title('Target Variable Distribution (Count)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Class', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_xticklabels(['No Diabetes', 'Diabetes'], rotation=0)

        # Percentage
        target_pct = self.df[self.target_col].value_counts(normalize=True) * 100
        target_pct.plot(kind='bar', ax=axes[1], color=['#3498db', '#e74c3c'])
        axes[1].set_title('Target Variable Distribution (%)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Class', fontsize=12)
        axes[1].set_ylabel('Percentage', fontsize=12)
        axes[1].set_xticklabels(['No Diabetes', 'Diabetes'], rotation=0)

        # Add value labels
        for ax in axes:
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_path}")

        plt.close()

    def plot_feature_distributions(self, output_path: Optional[str] = None):
        """
        Plot distributions of all features.

        Args:
            output_path: Path to save the plot
        """
        logger.info("Plotting feature distributions")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)

        n_cols = 4
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            self.df[col].hist(bins=30, ax=ax, color='#3498db', edgecolor='black', alpha=0.7)
            ax.set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(axis='y', alpha=0.3)

        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_path}")

        plt.close()

    def plot_correlation_matrix(self, output_path: Optional[str] = None):
        """
        Plot correlation matrix.

        Args:
            output_path: Path to save the plot
        """
        logger.info("Plotting correlation matrix")

        # Calculate correlation
        corr = self.df.corr()

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_path}")

        plt.close()

    def plot_feature_vs_target(self, output_path: Optional[str] = None):
        """
        Plot features vs target variable.

        Args:
            output_path: Path to save the plot
        """
        logger.info("Plotting features vs target")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)

        n_cols = 4
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            self.df.boxplot(column=col, by=self.target_col, ax=ax)
            ax.set_title(f'{col} by {self.target_col}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Diabetes Status', fontsize=10)
            ax.set_ylabel(col, fontsize=10)
            plt.sca(ax)
            plt.xticks([1, 2], ['No Diabetes', 'Diabetes'])

        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('')  # Remove default title
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_path}")

        plt.close()

    def plot_missing_data(self, output_path: Optional[str] = None):
        """
        Plot missing data analysis.

        Args:
            output_path: Path to save the plot
        """
        logger.info("Plotting missing data analysis")

        # Identify columns where 0 might represent missing values
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        zero_cols = [col for col in zero_cols if col in self.df.columns]

        # Count zeros and nulls
        missing_data = pd.DataFrame({
            'Feature': zero_cols,
            'Zero Count': [(self.df[col] == 0).sum() for col in zero_cols],
            'Null Count': [self.df[col].isnull().sum() for col in zero_cols]
        })

        missing_data['Zero %'] = (missing_data['Zero Count'] / len(self.df) * 100).round(2)
        missing_data['Null %'] = (missing_data['Null Count'] / len(self.df) * 100).round(2)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(missing_data))
        width = 0.35

        bars1 = ax.bar(x - width/2, missing_data['Zero %'], width, label='Zeros (%)', color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x + width/2, missing_data['Null %'], width, label='Nulls (%)', color='#3498db', alpha=0.8)

        ax.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage', fontsize=12, fontweight='bold')
        ax.set_title('Missing Data Analysis (Zeros and Nulls)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(missing_data['Feature'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_path}")

        plt.close()

    def generate_full_report(self, output_dir: str = 'visualizations/eda'):
        """
        Generate full EDA report with all visualizations.

        Args:
            output_dir: Directory to save visualizations
        """
        logger.info("Generating full EDA report")

        os.makedirs(output_dir, exist_ok=True)

        # Generate summary statistics
        summary = self.generate_summary_statistics()
        summary_path = os.path.join(output_dir, 'summary_statistics.csv')
        summary.to_csv(summary_path)
        logger.info(f"Summary statistics saved to {summary_path}")

        # Generate plots
        self.plot_target_distribution(os.path.join(output_dir, 'target_distribution.png'))
        self.plot_feature_distributions(os.path.join(output_dir, 'feature_distributions.png'))
        self.plot_correlation_matrix(os.path.join(output_dir, 'correlation_matrix.png'))
        self.plot_feature_vs_target(os.path.join(output_dir, 'features_vs_target.png'))
        self.plot_missing_data(os.path.join(output_dir, 'missing_data_analysis.png'))

        logger.info(f"Full EDA report generated in {output_dir}")

        return summary
