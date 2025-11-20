"""
Model Evaluation Module
Evaluate model performance with various metrics and visualizations
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Comprehensive model evaluation tools
    """
    
    def __init__(self, model, X_test, y_test, class_names=None):
        """
        Initialize evaluator
        
        Args:
            model: Trained model with predict method
            X_test: Test features
            y_test: True labels
            class_names: Names of classes (e.g., ['Negative', 'Neutral', 'Positive'])
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names
        self.y_pred = None
        self.y_pred_proba = None
        
    def predict(self):
        """
        Generate predictions
        """
        self.y_pred = self.model.predict(self.X_test)
        try:
            self.y_pred_proba = self.model.predict_proba(self.X_test)
        except:
            self.y_pred_proba = None
        
        return self.y_pred
    
    def calculate_metrics(self, average='weighted'):
        """
        Calculate all evaluation metrics
        """
        if self.y_pred is None:
            self.predict()
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred, average=average),
            'recall': recall_score(self.y_test, self.y_pred, average=average),
            'f1_score': f1_score(self.y_test, self.y_pred, average=average)
        }
        
        return metrics
    
    def print_metrics(self):
        """
        Print all metrics in a formatted way
        """
        if self.y_pred is None:
            self.predict()
        
        metrics = self.calculate_metrics()
        
        print("\n" + "="*50)
        print("📊 MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print("="*50)
        
        return metrics
    
    def print_classification_report(self):
        """
        Print detailed classification report
        """
        if self.y_pred is None:
            self.predict()
        
        print("\n📋 CLASSIFICATION REPORT:\n")
        print(classification_report(
            self.y_test,
            self.y_pred,
            target_names=self.class_names
        ))
    
    def plot_confusion_matrix(self, save_path=None, figsize=(8, 6)):
        """
        Plot confusion matrix heatmap
        """
        if self.y_pred is None:
            self.predict()
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved to {save_path}")
        
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, save_path=None, figsize=(8, 6)):
        """
        Plot ROC curve for binary or multi-class classification
        """
        if self.y_pred_proba is None:
            print("⚠️ Model does not support probability predictions")
            return
        
        plt.figure(figsize=figsize)
        
        # Binary classification
        if len(np.unique(self.y_test)) == 2:
            fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba[:, 1])
            auc_score = roc_auc_score(self.y_test, self.y_pred_proba[:, 1])
            
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            
        else:
            # Multi-class ROC (one-vs-rest)
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(self.y_test, classes=np.unique(self.y_test))
            
            for i, class_name in enumerate(self.class_names or range(len(np.unique(self.y_test)))):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], self.y_pred_proba[:, i])
                auc_score = roc_auc_score(y_test_bin[:, i], self.y_pred_proba[:, i])
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ ROC curve saved to {save_path}")
        
        plt.show()
    
    def get_misclassified_samples(self, n=10):
        """
        Get samples that were misclassified
        """
        if self.y_pred is None:
            self.predict()
        
        misclassified_idx = np.where(self.y_pred != self.y_test)[0]
        
        print(f"\n❌ Found {len(misclassified_idx)} misclassified samples")
        
        if n > 0:
            return misclassified_idx[:n]
        return misclassified_idx


def compare_models(models_results, metric='test_accuracy'):
    """
    Compare multiple models and visualize results
    
    Args:
        models_results: Dict with model names as keys and results as values
        metric: Metric to compare ('test_accuracy', 'train_accuracy', etc.)
    """
    print("\n" + "="*60)
    print("🏆 MODEL COMPARISON")
    print("="*60)
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        name: {
            'Train Accuracy': results['train_accuracy'],
            'Test Accuracy': results['test_accuracy']
        }
        for name, results in models_results.items()
    }).T
    
    print(comparison.to_string())
    
    # Find best model
    best_model = comparison['Test Accuracy'].idxmax()
    best_score = comparison['Test Accuracy'].max()
    
    print(f"\n🥇 Best Model: {best_model} (Test Accuracy: {best_score:.4f})")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison.plot(kind='bar', ax=ax, rot=0)
    plt.title('Model Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.legend(['Train Accuracy', 'Test Accuracy'])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return comparison


if __name__ == "__main__":
    print("Model Evaluation Module Loaded!")
    print("\nThis module provides comprehensive evaluation tools including:")
    print("  - Accuracy, Precision, Recall, F1-Score")
    print("  - Confusion Matrix visualization")
    print("  - ROC curves")
    print("  - Model comparison utilities")
