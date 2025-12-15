"""
Fonctions de diagnostic et visualisation pour modèles ML
"""
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_diagnostic_report(model, X, y_true, y_pred=None, y_score=None, feature_names=None, model_name="model", output_dir=".", labels=None, pos_label=1, scores_dict=None, param_name=None, param_range=None, cv=5, scoring=None, n_jobs=None):
    """
    Génère tous les graphes de diagnostic et un rapport texte pour un modèle donné.
    """
 
    os.makedirs(output_dir, exist_ok=True)
    if y_pred is None and hasattr(model, "predict"):
        try:
            y_pred = model.predict(X)
        except Exception:
            y_pred = None
    if y_score is None:
        if hasattr(model, "predict_proba"):
            try:
                y_score = model.predict_proba(X)[:, 1]
            except Exception:
                y_score = None
        elif hasattr(model, "decision_function"):
            try:
                y_score = model.decision_function(X)
            except Exception:
                y_score = None
        else:
            y_score = None
    # Classification report
    plot_classification_report(y_true, y_pred, labels=labels, filename=f"{output_dir}/{model_name}_classification_report.png")
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, labels=labels, filename=f"{output_dir}/{model_name}_confusion_matrix.png")
    # ROC curve
    if y_score is not None:
        plot_roc_curve(y_true, y_score, filename=f"{output_dir}/{model_name}_roc_curve.png", pos_label=pos_label)
        plot_precision_recall_curve(y_true, y_score, filename=f"{output_dir}/{model_name}_precision_recall_curve.png", pos_label=pos_label)
    # Feature importances
    if feature_names is not None:
        plot_feature_importances(model, feature_names, filename=f"{output_dir}/{model_name}_feature_importances.png")
    # Learning curve
    try:
        plot_learning_curve(model, X, y_true, filename=f"{output_dir}/{model_name}_learning_curve.png", cv=cv, scoring=scoring, n_jobs=n_jobs)
    except Exception:
        pass
    # Validation curve (si param_name et param_range fournis)
    if param_name and param_range is not None:
        try:
            plot_validation_curve(model, X, y_true, param_name, param_range, filename=f"{output_dir}/{model_name}_validation_curve.png", cv=cv, scoring=scoring, n_jobs=n_jobs)
        except Exception:
            pass
    # Model comparison (si scores_dict fourni)
    if scores_dict is not None:
        plot_model_comparison(scores_dict, filename=f"{output_dir}/model_comparison.png")
    # Rapport texte
    report_path = f"{output_dir}/{model_name}_diagnostic_report.txt"
    with open(report_path, "w") as f:
        if y_pred is not None:
            f.write("Classification Report:\n")
            f.write(classification_report(y_true, y_pred, labels=labels))
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(confusion_matrix(y_true, y_pred, labels=labels)))
        else:
            f.write("Aucune prédiction disponible pour générer le rapport.\n")
        if y_score is not None:
            from sklearn.metrics import roc_auc_score, average_precision_score
            try:
                roc_auc = roc_auc_score(y_true, y_score)
                ap = average_precision_score(y_true, y_score)
                f.write(f"\nROC AUC: {roc_auc:.4f}\n")
                f.write(f"Average Precision: {ap:.4f}\n")
            except Exception:
                pass
        if feature_names is not None and hasattr(model, "feature_importances_"):
            f.write("\nFeature Importances:\n")
            importances = model.feature_importances_
            for name, val in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
                f.write(f"{name}: {val:.4f}\n")
        elif feature_names is not None:
            f.write("\n[INFO] Le modèle ne possède pas d'attribut feature_importances_.\n")
    print(f"✅ Rapport de diagnostic généré dans {output_dir}")


def plot_classification_report(y_true, y_pred, labels=None, filename="classification_report.png"):
    """
    Génère un rapport de classification sous forme de heatmap et sauvegarde en PNG.
    """
    report = classification_report(y_true, y_pred, output_dict=True, labels=labels)
    report_matrix = np.array([list(v.values()) for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")])
    plt.figure(figsize=(8, 4))
    sns.heatmap(report_matrix, annot=True, fmt=".2f", cmap="Blues",
                yticklabels=[k for k in report if k not in ("accuracy", "macro avg", "weighted avg")],
                xticklabels=["precision", "recall", "f1-score", "support"])
    plt.title("Classification Report")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels=None, filename="confusion_matrix.png"):
    """
    Génère une matrice de confusion et sauvegarde en PNG.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_feature_importances(model, feature_names, filename="feature_importances.png", top_n=20):
    """
    Affiche et sauvegarde les importances des features d'un modèle (ex: RandomForest).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], orient="h")
        plt.title("Feature Importances")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    else:
        print("[INFO] Le modèle ne possède pas d'attribut feature_importances_.")

def plot_model_comparison(scores_dict, filename="model_comparison.png"):
    """
    Compare plusieurs modèles (dict: nom -> score) via un barplot.
    """
    names = list(scores_dict.keys())
    scores = list(scores_dict.values())
    plt.figure(figsize=(8, 4))
    sns.barplot(x=scores, y=names, orient="h")
    plt.xlabel("Score")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_roc_curve(y_true, y_score, filename="roc_curve.png", pos_label=1):
    """
    Génère la courbe ROC et sauvegarde en PNG.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_precision_recall_curve(y_true, y_score, filename="precision_recall_curve.png", pos_label=1):
    """
    Génère la courbe Precision-Recall et sauvegarde en PNG.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
    ap = average_precision_score(y_true, y_score, pos_label=pos_label)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="blue", lw=2, label=f"AP = {ap:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_learning_curve(estimator, X, y, filename="learning_curve.png", cv=5, scoring=None, n_jobs=None):
    """
    Génère une courbe d'apprentissage (train/test score vs. taille d'échantillon).
    """
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_validation_curve(estimator, X, y, param_name, param_range, filename="validation_curve.png", cv=5, scoring=None, n_jobs=None):
    """
    Génère une courbe de validation (score vs. valeur d'un hyperparamètre).
    """
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.figure(figsize=(7, 5))
    plt.plot(param_range, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(param_range, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.title("Validation Curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
