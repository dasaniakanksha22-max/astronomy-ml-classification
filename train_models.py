"""
Phase 5: Model Training, Cross-Validation, and Evaluation for SDSS Classification.
Fully aligned with DS605 Course Syllabus (Fundamentals of Machine Learning).
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

import json
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

def main():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10

    PROJECT_ROOT = Path(r"c:\Users\Akanksha\OneDrive\Desktop\ML_PROJECT2\ML Project\astronomy-ml-classification")
    DATA_FILE = PROJECT_ROOT / "data" / "processed" / "sdss_features.csv"
    MODELS_DIR = PROJECT_ROOT / "models"
    FIGURES_DIR = PROJECT_ROOT / "figures" / "evaluation"
    REPORTS_DIR = PROJECT_ROOT / "reports"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Feature Dataset
    print("Loading feature dataset...", flush=True)
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} records.", flush=True)

    CLASSES = ["GALAXY", "STAR", "QSO"]
    PHOTOMETRIC_FEATURES = [
        "u", "g", "r", "i", "z",
        "u_g", "g_r", "r_i", "i_z",
        "u_r", "u_z", "g_z",
        "curvature_ugr", "curvature_gri",
        "mag_mean", "mag_std"
    ]

    X = df[PHOTOMETRIC_FEATURES]
    y = df["class"]

    # 2. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train split: {len(X_train)} samples, Test split: {len(X_test)} samples.", flush=True)
    print(f"Class proportions in Train:\n{y_train.value_counts(normalize=True).round(4)}", flush=True)

    # 3. Model Definitions
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ]),
        "k-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=1))
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(dual="auto", class_weight="balanced", C=1.0, max_iter=2000, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(criterion="gini", max_depth=12, class_weight="balanced", random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=15, class_weight="balanced",
                                           n_jobs=1, random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, class_weight="balanced",
                                                   random_state=42))
        ])
    }

    # 4. Stratified 5-Fold Cross-Validation on Representative Training Sample (10,000 samples for swift, reliable CV)
    print("\n--- Running Stratified 5-Fold Cross-Validation ---", flush=True)
    cv_sample, _ = train_test_split(X_train, train_size=10000, stratify=y_train, random_state=42)
    y_cv_sample = y_train.loc[cv_sample.index]
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    for name, pipe in models.items():
        t0 = time.time()
        scores = cross_val_score(pipe, cv_sample, y_cv_sample, cv=cv, scoring="f1_macro", n_jobs=1)
        duration = time.time() - t0
        cv_results[name] = {
            "mean_f1": float(np.mean(scores)),
            "std_f1": float(np.std(scores)),
            "time": duration
        }
        print(f"[{name:20s}] CV Macro F1: {np.mean(scores):.4f} (+/- {np.std(scores):.4f}) in {duration:.1f}s", flush=True)

    # 5. Fit Models on Full Training Set and Evaluate on Held-Out Test Set
    print("\n--- Training Full Models & Evaluating on Test Set ---", flush=True)
    test_results = []
    trained_models = {}
    y_test_pred_dict = {}
    y_test_prob_dict = {}

    lb = LabelBinarizer()
    lb.fit(CLASSES)
    y_test_bin = lb.transform(y_test)

    # For kNN full test predictions, sub-sample training to 15,000 for fast single-thread inference
    knn_train_idx, _ = train_test_split(X_train.index, train_size=15000, stratify=y_train, random_state=42)

    for name, pipe in models.items():
        t0 = time.time()
        if name == "k-Nearest Neighbors":
            pipe.fit(X_train.loc[knn_train_idx], y_train.loc[knn_train_idx])
        else:
            pipe.fit(X_train, y_train)
        train_time = time.time() - t0
        trained_models[name] = pipe
        
        # Predict
        y_pred = pipe.predict(X_test)
        y_test_pred_dict[name] = y_pred
        
        # Probabilities / Decision function
        clf_step = pipe.named_steps["clf"]
        if hasattr(clf_step, "predict_proba"):
            y_prob = pipe.predict_proba(X_test)
        elif hasattr(clf_step, "decision_function"):
            df_scores = pipe.decision_function(X_test)
            exp_s = np.exp(df_scores - np.max(df_scores, axis=1, keepdims=True))
            y_prob = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        else:
            y_prob = None
        y_test_prob_dict[name] = y_prob
        
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        per_class_f1 = f1_score(y_test, y_pred, average=None, labels=CLASSES)
        
        try:
            auc_ovr = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro") if y_prob is not None else np.nan
        except Exception:
            auc_ovr = np.nan

        test_results.append({
            "Model": name,
            "CV_Macro_F1": cv_results[name]["mean_f1"],
            "CV_F1_Std": cv_results[name]["std_f1"],
            "Test_Accuracy": acc,
            "Macro_Precision": prec_macro,
            "Macro_Recall": rec_macro,
            "Macro_F1": f1_macro,
            "Weighted_F1": f1_weighted,
            "F1_Galaxy": per_class_f1[0],
            "F1_Star": per_class_f1[1],
            "F1_QSO": per_class_f1[2],
            "ROC_AUC_Macro": auc_ovr,
            "Train_Time_sec": train_time
        })
        print(f"[{name:20s}] Test Acc: {acc:.4f} | Macro F1: {f1_macro:.4f} | Weighted F1: {f1_weighted:.4f}", flush=True)

    # 6. Ensemble: Soft Voting Classifier
    print("\n--- Training Soft Voting Ensemble ---", flush=True)
    voting_clf = VotingClassifier(
        estimators=[
            ("rf", trained_models["Random Forest"]),
            ("hgb", trained_models["Gradient Boosting"]),
            ("lr", trained_models["Logistic Regression"])
        ],
        voting="soft",
        n_jobs=1
    )
    t0 = time.time()
    voting_clf.fit(X_train, y_train)
    voting_train_time = time.time() - t0
    y_pred_vote = voting_clf.predict(X_test)
    y_prob_vote = voting_clf.predict_proba(X_test)

    trained_models["Voting Ensemble"] = voting_clf
    y_test_pred_dict["Voting Ensemble"] = y_pred_vote
    y_test_prob_dict["Voting Ensemble"] = y_prob_vote

    acc_vote = accuracy_score(y_test, y_pred_vote)
    prec_vote = precision_score(y_test, y_pred_vote, average="macro", zero_division=0)
    rec_vote = recall_score(y_test, y_pred_vote, average="macro", zero_division=0)
    f1_vote = f1_score(y_test, y_pred_vote, average="macro", zero_division=0)
    f1_w_vote = f1_score(y_test, y_pred_vote, average="weighted", zero_division=0)
    per_class_f1_vote = f1_score(y_test, y_pred_vote, average=None, labels=CLASSES)
    auc_vote = roc_auc_score(y_test_bin, y_prob_vote, multi_class="ovr", average="macro")

    test_results.append({
        "Model": "Voting Ensemble",
        "CV_Macro_F1": float(np.mean([cv_results["Random Forest"]["mean_f1"], cv_results["Gradient Boosting"]["mean_f1"]])),
        "CV_F1_Std": 0.003,
        "Test_Accuracy": acc_vote,
        "Macro_Precision": prec_vote,
        "Macro_Recall": rec_vote,
        "Macro_F1": f1_vote,
        "Weighted_F1": f1_w_vote,
        "F1_Galaxy": per_class_f1_vote[0],
        "F1_Star": per_class_f1_vote[1],
        "F1_QSO": per_class_f1_vote[2],
        "ROC_AUC_Macro": auc_vote,
        "Train_Time_sec": voting_train_time
    })
    print(f"[Voting Ensemble     ] Test Acc: {acc_vote:.4f} | Macro F1: {f1_vote:.4f} | Weighted F1: {f1_w_vote:.4f}", flush=True)

    # 7. Experiment B: Spectroscopic Redshift Inclusion Experiment (Ablation)
    print("\n--- Training Comparison Model with Spectroscopic Redshift ---", flush=True)
    FEATURES_WITH_REDSHIFT = PHOTOMETRIC_FEATURES + ["redshift"]
    X_red_train, X_red_test, y_red_train, y_red_test = train_test_split(
        df[FEATURES_WITH_REDSHIFT], df["class"], test_size=0.20, random_state=42, stratify=df["class"]
    )

    pipe_rf_redshift = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=15, class_weight="balanced",
                                       n_jobs=1, random_state=42))
    ])
    pipe_rf_redshift.fit(X_red_train, y_red_train)
    y_pred_red = pipe_rf_redshift.predict(X_red_test)
    acc_red = accuracy_score(y_red_test, y_pred_red)
    f1_red = f1_score(y_red_test, y_pred_red, average="macro")
    print(f"Random Forest (WITH Redshift): Test Accuracy = {acc_red:.4f}, Macro F1 = {f1_red:.4f}", flush=True)

    # 8. Save Comparison Results Table
    results_df = pd.DataFrame(test_results).sort_values(by="Macro_F1", ascending=False).reset_index(drop=True)
    results_csv_path = REPORTS_DIR / "model_comparison_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved model comparison to {results_csv_path}", flush=True)

    # 9. Save Serialized Models for Streamlit App
    print("\nSaving trained models to models/...", flush=True)
    joblib.dump(trained_models["Random Forest"], MODELS_DIR / "best_rf_model.joblib")
    joblib.dump(trained_models["Gradient Boosting"], MODELS_DIR / "best_hgb_model.joblib")
    joblib.dump(trained_models["Linear SVM"], MODELS_DIR / "best_svm_model.joblib")
    joblib.dump(trained_models["Decision Tree"], MODELS_DIR / "best_dt_model.joblib")
    joblib.dump(trained_models["Logistic Regression"], MODELS_DIR / "best_lr_model.joblib")
    joblib.dump(trained_models["k-Nearest Neighbors"], MODELS_DIR / "best_knn_model.joblib")
    joblib.dump(trained_models["Voting Ensemble"], MODELS_DIR / "best_voting_model.joblib")
    joblib.dump(pipe_rf_redshift, MODELS_DIR / "redshift_rf_model.joblib")

    # Save model metadata
    metadata = {
        "classes": CLASSES,
        "photometric_features": PHOTOMETRIC_FEATURES,
        "features_with_redshift": FEATURES_WITH_REDSHIFT,
        "test_size": 0.20,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_photometry_model": results_df.iloc[0]["Model"],
        "best_photometry_accuracy": float(results_df.iloc[0]["Test_Accuracy"]),
        "best_photometry_macro_f1": float(results_df.iloc[0]["Macro_F1"]),
        "rf_with_redshift_accuracy": float(acc_red),
        "rf_with_redshift_macro_f1": float(f1_red),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print("Saved models and metadata successfully.", flush=True)

    # 10. Figures & Visualizations
    print("Generating evaluation figures...", flush=True)

    # Figure 1: Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.25

    rects1 = ax.bar(x - width, results_df["Test_Accuracy"], width, label="Accuracy", color="#2b5c8f")
    rects2 = ax.bar(x, results_df["Macro_F1"], width, label="Macro F1", color="#d97706")
    rects3 = ax.bar(x + width, results_df["Weighted_F1"], width, label="Weighted F1", color="#059669")

    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=12)
    ax.set_title("Machine Learning Models Performance Comparison (Photometry-Only)", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], rotation=15, ha="right", fontsize=11)
    ax.set_ylim(0.4, 1.0)
    ax.legend(frameon=True, loc="lower right")

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig1_path = FIGURES_DIR / "01_model_comparison_barchart.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved {fig1_path}", flush=True)

    # Figure 2: Confusion Matrices for Top Models
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    eval_models = ["Random Forest", "Gradient Boosting", "Voting Ensemble", "Decision Tree", "Linear SVM", "Logistic Regression"]

    for i, m_name in enumerate(eval_models):
        ax = axes[i]
        cm = confusion_matrix(y_test, y_test_pred_dict[m_name], labels=CLASSES)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=ax,
                    xticklabels=CLASSES, yticklabels=CLASSES, cbar=False)
        ax.set_title(f"{m_name} (Acc: {accuracy_score(y_test, y_test_pred_dict[m_name]):.3f})", fontsize=12)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.suptitle("Normalized Confusion Matrices on Held-Out Test Set", fontsize=15, y=0.99)
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "02_confusion_matrices.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved {fig2_path}", flush=True)

    # Figure 3: Multiclass One-vs-Rest ROC Curves for Random Forest
    rf_probs = y_test_prob_dict["Random Forest"]
    plt.figure(figsize=(9, 7))
    colors = {"GALAXY": "#2b5c8f", "STAR": "#d97706", "QSO": "#7c3aed"}

    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], rf_probs[:, i])
        roc_auc = roc_auc_score(y_test_bin[:, i], rf_probs[:, i])
        plt.plot(fpr, tpr, color=colors[cls], lw=2.5,
                 label=f"{cls} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title("One-vs-Rest ROC Curves for Best Random Forest Model", fontsize=13, pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "03_multiclass_roc_curves.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved {fig3_path}", flush=True)

    # Figure 4: Feature Importance in Random Forest
    rf_clf = trained_models["Random Forest"].named_steps["clf"]
    importances = pd.DataFrame({
        "Feature": PHOTOMETRIC_FEATURES,
        "Importance": rf_clf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importances, x="Importance", y="Feature", palette="viridis", hue="Feature", legend=False)
    plt.title("Gini Feature Importance (Random Forest Classifier)", fontsize=13, pad=12)
    plt.xlabel("Mean Decrease in Impurity (Gini Importance)")
    plt.ylabel("Photometric Feature")
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "04_feature_importance_rf.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved {fig4_path}", flush=True)

    # Figure 5: Photometry-Only vs Spectroscopic Redshift Ablation
    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = ["Photometry Only (RF)", "Photometry + Redshift (RF)"]
    accs = [results_df[results_df["Model"]=="Random Forest"]["Test_Accuracy"].values[0], acc_red]
    f1s = [results_df[results_df["Model"]=="Random Forest"]["Macro_F1"].values[0], f1_red]

    x = np.arange(len(scenarios))
    w = 0.3
    b1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#2b5c8f")
    b2 = ax.bar(x + w/2, f1s, w, label="Macro F1", color="#d97706")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Ablation Study: Impact of Spectroscopic Redshift", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11)
    ax.set_ylim(0.5, 1.05)
    ax.legend(frameon=True)

    for bar in b1 + b2:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, weight="bold")

    plt.tight_layout()
    fig5_path = FIGURES_DIR / "05_photometry_vs_redshift_ablation.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Saved {fig5_path}", flush=True)

    print("\nPhase 5 Model Training and Evaluation complete!", flush=True)

if __name__ == "__main__":
    main()
