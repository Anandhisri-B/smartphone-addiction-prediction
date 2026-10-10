"""
train_model.py
==============
Trains Random Forest and Decision Tree classifiers on the extracted
feature CSV. Saves trained models as .pkl files used by the Flask API.

Run:
    python train_model.py
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.ensemble          import RandomForestClassifier
from sklearn.tree              import DecisionTreeClassifier, export_text
from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing     import LabelEncoder
from sklearn.metrics           import (accuracy_score, classification_report,
                                        confusion_matrix, ConfusionMatrixDisplay)
from sklearn.inspection        import permutation_importance

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_CSV    = Path("data/features.csv")
MODELS_DIR  = Path("models")
STATIC_DIR  = Path("static/charts")
MODELS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ── Features used by the model ────────────────────────────────────────────────
FEATURE_COLS = [
    'total_screen_time_min',
    'social_time_min',
    'productivity_time_min',
    'video_time_min',
    'gaming_time_min',
    'others_time_min',
    'night_10pm_2am_min',
    'night_usage_pct',
    'social_prod_ratio',
    'gaming_pct',
    'notification_per_hour',
    'unlock_frequency_per_hour',
    'total_notifications',
    'total_unlocks',
]
TARGET_COL = 'addiction_label'
LABEL_ORDER = ['Low', 'Moderate', 'High']


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    print("\n" + "="*60)
    print("  SMARTPHONE ADDICTION MODEL TRAINING")
    print("="*60)

    # ── Load data ──────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_CSV)
    print(f"\n📊 Dataset: {len(df)} samples")
    print(f"   Label distribution:\n{df[TARGET_COL].value_counts().to_string()}\n")

    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL]

    # Encode labels: Low=0, Moderate=1, High=2
    le = LabelEncoder()
    le.classes_ = np.array(LABEL_ORDER)
    y_enc = le.transform(y)

    # ── Train/test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )

    # ── Random Forest ──────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc  = accuracy_score(y_test, rf_pred)

    rf_cv = cross_val_score(rf, X, y_enc,
                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                            scoring='accuracy')

    print("🌲 RANDOM FOREST")
    print(f"   Test Accuracy  : {rf_acc*100:.2f}%")
    print(f"   CV Accuracy    : {rf_cv.mean()*100:.2f}% ± {rf_cv.std()*100:.2f}%")
    print(f"\n{classification_report(y_test, rf_pred, target_names=LABEL_ORDER)}")

    # ── Decision Tree ──────────────────────────────────────────────────────
    dt = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42
    )
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_acc  = accuracy_score(y_test, dt_pred)

    dt_cv = cross_val_score(dt, X, y_enc,
                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                            scoring='accuracy')

    print("🌳 DECISION TREE")
    print(f"   Test Accuracy  : {dt_acc*100:.2f}%")
    print(f"   CV Accuracy    : {dt_cv.mean()*100:.2f}% ± {dt_cv.std()*100:.2f}%")
    print(f"\n{classification_report(y_test, dt_pred, target_names=LABEL_ORDER)}")

    # ── Save models ────────────────────────────────────────────────────────
    with open(MODELS_DIR/'random_forest.pkl', 'wb') as f:
        pickle.dump(rf, f)
    with open(MODELS_DIR/'decision_tree.pkl', 'wb') as f:
        pickle.dump(dt, f)
    with open(MODELS_DIR/'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    with open(MODELS_DIR/'feature_cols.json', 'w') as f:
        json.dump(FEATURE_COLS, f)

    # Save accuracy summary
    summary = {
        'rf_test_acc':  round(rf_acc * 100, 2),
        'rf_cv_acc':    round(rf_cv.mean() * 100, 2),
        'rf_cv_std':    round(rf_cv.std() * 100, 2),
        'dt_test_acc':  round(dt_acc * 100, 2),
        'dt_cv_acc':    round(dt_cv.mean() * 100, 2),
        'dt_cv_std':    round(dt_cv.std() * 100, 2),
        'n_samples':    len(df),
        'feature_cols': FEATURE_COLS,
        'label_distribution': df[TARGET_COL].value_counts().to_dict(),
    }
    with open(MODELS_DIR/'accuracy_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅  Models saved to {MODELS_DIR}/")

    # ── Generate training charts ───────────────────────────────────────────
    _plot_confusion_matrix(rf, X_test, y_test, le, 'Random Forest')
    _plot_confusion_matrix(dt, X_test, y_test, le, 'Decision Tree')
    _plot_feature_importance(rf, FEATURE_COLS)
    _plot_accuracy_comparison(rf_acc, dt_acc, rf_cv, dt_cv)

    print(f"✅  Charts saved to {STATIC_DIR}/\n")
    return rf, dt, le, summary


def _plot_confusion_matrix(model, X_test, y_test, le, name: str):
    preds = model.predict(X_test)
    cm    = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=LABEL_ORDER)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix – {name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = name.lower().replace(' ', '_')
    plt.savefig(STATIC_DIR / f'cm_{fname}.png', dpi=120)
    plt.close()


def _plot_feature_importance(rf, feature_cols):
    imp    = rf.feature_importances_
    sorted_idx = np.argsort(imp)[::-1]
    labels = [feature_cols[i].replace('_', ' ').title() for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels[::-1], imp[sorted_idx[::-1]],
                   color='#4f8ef7', edgecolor='white')
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title('Feature Importances (Random Forest)', fontsize=13, fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(STATIC_DIR / 'feature_importance.png', dpi=120)
    plt.close()

    # Save as JSON for frontend charts
    fi_data = {feature_cols[i]: round(float(imp[i]), 4)
               for i in range(len(feature_cols))}
    with open(STATIC_DIR / 'feature_importance.json', 'w') as f:
        json.dump(fi_data, f, indent=2)


def _plot_accuracy_comparison(rf_acc, dt_acc, rf_cv, dt_cv):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar chart
    ax = axes[0]
    models = ['Random Forest', 'Decision Tree']
    accs   = [rf_acc * 100, dt_acc * 100]
    colors = ['#4f8ef7', '#f7764f']
    bars = ax.bar(models, accs, color=colors, width=0.4, edgecolor='white')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax.set_ylim(0, 115)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy Comparison', fontweight='bold')
    ax.spines[['top','right']].set_visible(False)

    # CV box plot
    ax2 = axes[1]
    ax2.boxplot([rf_cv * 100, dt_cv * 100], labels=models, patch_artist=True,
                boxprops=dict(facecolor='#e8f0fe', color='#4f8ef7'))
    ax2.set_ylabel('CV Accuracy (%)')
    ax2.set_title('Cross-Validation Accuracy (5-Fold)', fontweight='bold')
    ax2.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(STATIC_DIR / 'accuracy_comparison.png', dpi=120)
    plt.close()


if __name__ == '__main__':
    train()