"""
============================================================
EMPLOYEE ATTRITION PREDICTION - COMPLETE ANALYSIS PIPELINE
============================================================
Run this script first to generate all models, reports, and artifacts.
Command: python attrition_analysis.py
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# Get script directory for saving files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_path(filename):
    """Return full path in the same directory as this script."""
    return os.path.join(SCRIPT_DIR, filename)


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
print("=" * 70)
print("STEP 1: LOADING DATASET")
print("=" * 70)

df = pd.read_csv(save_path('employee_attrition.csv'))

print(f"Dataset Shape: {df.shape}")
print(f"Total Employees: {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())


# ============================================================
# STEP 2: DATA QUALITY CHECK
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: DATA QUALITY CHECK")
print("=" * 70)

# Missing values
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing Percentage': missing_pct
})
missing_with_values = missing_df[missing_df['Missing Count'] > 0]

if len(missing_with_values) == 0:
    print("✅ No missing values found!")
else:
    print("⚠️ Missing Values Found:")
    print(missing_with_values)

# Duplicates
duplicates = df.duplicated().sum()
print(f"{'✅' if duplicates == 0 else '⚠️'} Duplicate Rows: {duplicates}")

# Data types
print(f"\nNumerical Columns ({len(df.select_dtypes(include=[np.number]).columns)}):")
print(f"  {df.select_dtypes(include=[np.number]).columns.tolist()}")
print(f"\nCategorical Columns ({len(df.select_dtypes(include=['object']).columns)}):")
print(f"  {df.select_dtypes(include=['object']).columns.tolist()}")


# ============================================================
# STEP 3: TARGET VARIABLE PROCESSING
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: TARGET VARIABLE ANALYSIS")
print("=" * 70)

# Convert if needed
if df['Attrition'].dtype == 'object':
    print("Converting Attrition from Yes/No to 1/0")
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

attrition_counts = df['Attrition'].value_counts()
attrition_pct = df['Attrition'].value_counts(normalize=True) * 100

print(f"Stayed (0): {attrition_counts[0]} ({attrition_pct[0]:.1f}%)")
print(f"Left   (1): {attrition_counts[1]} ({attrition_pct[1]:.1f}%)")
print(f"Imbalance Ratio: 1:{attrition_counts[0] // attrition_counts[1]}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#2ecc71', '#e74c3c']

axes[0].bar(['Stayed (0)', 'Left (1)'], attrition_counts.values, color=colors)
axes[0].set_title('Attrition Count', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Employees')
for i, v in enumerate(attrition_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold', fontsize=12)

axes[1].pie(attrition_counts.values, labels=['Stayed', 'Left'],
            autopct='%1.1f%%', colors=colors, startangle=90,
            explode=(0, 0.1), shadow=True, textprops={'fontsize': 12})
axes[1].set_title('Attrition Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(save_path('target_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: target_distribution.png")


# ============================================================
# STEP 4: REMOVE IRRELEVANT COLUMNS
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: REMOVING IRRELEVANT COLUMNS")
print("=" * 70)

cols_to_drop = []
employee_ids = None

for col in ['EmployeeCount', 'Over18', 'StandardHours']:
    if col in df.columns:
        if df[col].nunique() <= 1:
            cols_to_drop.append(col)
            print(f"  Dropping '{col}' — constant value: {df[col].unique()}")

if 'EmployeeNumber' in df.columns:
    employee_ids = df['EmployeeNumber'].copy()
    cols_to_drop.append('EmployeeNumber')
    print(f"  Dropping 'EmployeeNumber' — unique identifier")

df_clean = df.drop(columns=cols_to_drop, errors='ignore')
print(f"\nShape after dropping: {df_clean.shape}")


# ============================================================
# STEP 5: EDA — NUMERICAL DISTRIBUTIONS
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: EDA — NUMERICAL DISTRIBUTIONS")
print("=" * 70)

numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
if 'Attrition' in numerical_cols:
    numerical_cols.remove('Attrition')

n_cols_plot = 4
n_rows_plot = (len(numerical_cols) + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(20, 4 * n_rows_plot))
axes_flat = axes.flatten()

for idx, col in enumerate(numerical_cols):
    ax = axes_flat[idx]
    df_clean[df_clean['Attrition'] == 0][col].hist(
        ax=ax, bins=30, alpha=0.6, color='#2ecc71', label='Stayed', density=True)
    df_clean[df_clean['Attrition'] == 1][col].hist(
        ax=ax, bins=30, alpha=0.6, color='#e74c3c', label='Left', density=True)
    ax.set_title(col, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

for idx in range(len(numerical_cols), len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle('Numerical Feature Distributions by Attrition',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(save_path('numerical_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: numerical_distributions.png")


# ============================================================
# STEP 6: EDA — CATEGORICAL ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: EDA — CATEGORICAL ANALYSIS")
print("=" * 70)

categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

if len(categorical_cols) > 0:
    fig, axes = plt.subplots(len(categorical_cols), 1,
                             figsize=(14, 5 * len(categorical_cols)))
    if len(categorical_cols) == 1:
        axes = [axes]

    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        ct = pd.crosstab(df_clean[col], df_clean['Attrition'])
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        ct_pct.plot(kind='bar', stacked=True, ax=ax,
                    color=['#2ecc71', '#e74c3c'], alpha=0.8)
        ax.set_title(f'Attrition Rate by {col}', fontsize=13, fontweight='bold')
        ax.set_ylabel('Percentage (%)')
        ax.legend(['Stayed', 'Left'], loc='upper right')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path('categorical_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved: categorical_analysis.png")
else:
    print("⚠️ No categorical columns found to plot.")


# ============================================================
# STEP 7: EDA — CORRELATION MATRIX
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: EDA — CORRELATION ANALYSIS")
print("=" * 70)

fig, ax = plt.subplots(figsize=(20, 16))
corr_cols = numerical_cols + ['Attrition']
corr_matrix = df_clean[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlBu_r', center=0, square=True,
            linewidths=0.5, ax=ax, annot_kws={'size': 7},
            vmin=-1, vmax=1)
ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(save_path('correlation_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: correlation_matrix.png")

# Top correlations with attrition
attrition_corr = corr_matrix['Attrition'].drop('Attrition').sort_values(ascending=False)
print("\nTop Positive Correlations with Attrition:")
print(attrition_corr[attrition_corr > 0].head(5).to_string())
print("\nTop Negative Correlations with Attrition:")
print(attrition_corr[attrition_corr < 0].head(5).to_string())


# ============================================================
# STEP 8: EDA — KEY INSIGHTS VISUALIZATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: EDA — KEY INSIGHTS")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Overtime vs Attrition
if 'OverTime' in df_clean.columns:
    ct = pd.crosstab(df_clean['OverTime'], df_clean['Attrition'], normalize='index') * 100
    ct.plot(kind='bar', ax=axes[0, 0], color=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_title('Overtime vs Attrition Rate', fontweight='bold')
    axes[0, 0].set_ylabel('Percentage')
    axes[0, 0].tick_params(axis='x', rotation=0)

# 2. Monthly Income by Attrition
df_clean.boxplot(column='MonthlyIncome', by='Attrition', ax=axes[0, 1])
axes[0, 1].set_title('Monthly Income by Attrition', fontweight='bold')
axes[0, 1].set_xticklabels(['Stayed', 'Left'])
plt.suptitle('')

# 3. Years at Company
df_clean.boxplot(column='YearsAtCompany', by='Attrition', ax=axes[0, 2])
axes[0, 2].set_title('Years at Company by Attrition', fontweight='bold')
axes[0, 2].set_xticklabels(['Stayed', 'Left'])
plt.suptitle('')

# 4. Job Satisfaction
ct = pd.crosstab(df_clean['JobSatisfaction'], df_clean['Attrition'], normalize='index') * 100
ct[1].plot(kind='bar', ax=axes[1, 0], color='#e74c3c', alpha=0.8)
axes[1, 0].set_title('Attrition Rate by Job Satisfaction', fontweight='bold')
axes[1, 0].set_ylabel('Attrition Rate (%)')
axes[1, 0].tick_params(axis='x', rotation=0)

# 5. Age Distribution
axes[1, 1].hist(df_clean[df_clean['Attrition'] == 0]['Age'], bins=20, alpha=0.6,
                color='#2ecc71', label='Stayed', density=True)
axes[1, 1].hist(df_clean[df_clean['Attrition'] == 1]['Age'], bins=20, alpha=0.6,
                color='#e74c3c', label='Left', density=True)
axes[1, 1].set_title('Age Distribution by Attrition', fontweight='bold')
axes[1, 1].legend()

# 6. Department
if 'Department' in df_clean.columns:
    ct = pd.crosstab(df_clean['Department'], df_clean['Attrition'], normalize='index') * 100
    ct[1].plot(kind='bar', ax=axes[1, 2], color='#e74c3c', alpha=0.8)
    axes[1, 2].set_title('Attrition Rate by Department', fontweight='bold')
    axes[1, 2].set_ylabel('Attrition Rate (%)')
    axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(save_path('key_insights.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: key_insights.png")

# Print EDA summary
print("\n📊 ATTRITION RATES BY KEY FACTORS:")
for col in ['OverTime', 'MaritalStatus', 'BusinessTravel', 'Department']:
    if col in df_clean.columns:
        rates = df_clean.groupby(col)['Attrition'].mean() * 100
        print(f"\n  {col}:")
        for cat, rate in rates.sort_values(ascending=False).items():
            print(f"    {cat}: {rate:.1f}%")


# ============================================================
# STEP 9: FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: FEATURE ENGINEERING")
print("=" * 70)

df_fe = df_clean.copy()

# 1. Income to Experience Ratio
df_fe['IncomePerYear'] = np.where(
    df_fe['TotalWorkingYears'] > 0,
    df_fe['MonthlyIncome'] / df_fe['TotalWorkingYears'],
    df_fe['MonthlyIncome']
)
print("✅ Created: IncomePerYear")

# 2. Promotion Delay Indicators
df_fe['PromotionDelay'] = df_fe['YearsAtCompany'] - df_fe['YearsSinceLastPromotion']
df_fe['IsOverduePromotion'] = (df_fe['YearsSinceLastPromotion'] >= 5).astype(int)
print("✅ Created: PromotionDelay, IsOverduePromotion")

# 3. Engagement Composite Score
satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction',
                     'RelationshipSatisfaction', 'WorkLifeBalance']
df_fe['EngagementScore'] = df_fe[satisfaction_cols].mean(axis=1)
print("✅ Created: EngagementScore")

# 4. Workload Stress Flag
if 'OverTime' in df_fe.columns:
    overtime_numeric = (df_fe['OverTime'] == 'Yes').astype(int) if df_fe['OverTime'].dtype == 'object' else df_fe['OverTime']
    df_fe['WorkloadStress'] = (
        (overtime_numeric == 1) & (df_fe['WorkLifeBalance'] <= 2)
    ).astype(int)
else:
    df_fe['WorkloadStress'] = 0
print("✅ Created: WorkloadStress")

# 5. Tenure Ratio
df_fe['TenureRatio'] = np.where(
    df_fe['TotalWorkingYears'] > 0,
    df_fe['YearsAtCompany'] / df_fe['TotalWorkingYears'],
    0
)
print("✅ Created: TenureRatio")

# 6. Role Stagnation
df_fe['RoleStagnation'] = np.where(
    df_fe['YearsAtCompany'] > 0,
    df_fe['YearsInCurrentRole'] / df_fe['YearsAtCompany'],
    0
)
print("✅ Created: RoleStagnation")

# 7. Satisfaction-Income Interaction
df_fe['SatisfactionIncomeRatio'] = df_fe['JobSatisfaction'] * df_fe['MonthlyIncome'] / 10000
print("✅ Created: SatisfactionIncomeRatio")

# 8. Company Hopper Flag
df_fe['CompanyHopper'] = (df_fe['NumCompaniesWorked'] > 4).astype(int)
print("✅ Created: CompanyHopper")

print(f"\nShape after feature engineering: {df_fe.shape}")
print(f"New features added: {df_fe.shape[1] - df_clean.shape[1]}")


# ============================================================
# STEP 10: ENCODE CATEGORICAL VARIABLES
# ============================================================
print("\n" + "=" * 70)
print("STEP 10: ENCODING CATEGORICAL VARIABLES")
print("=" * 70)

df_encoded = df_fe.copy()
label_encoders = {}
cat_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()

print(f"Categorical columns to encode: {cat_cols}")

for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le
    print(f"  ✅ {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Also create one-hot version for Logistic Regression
df_onehot = pd.get_dummies(df_fe, columns=cat_cols, drop_first=True)

print(f"\nLabel Encoded shape: {df_encoded.shape}")
print(f"One-Hot Encoded shape: {df_onehot.shape}")


# ============================================================
# STEP 11: TRAIN-TEST SPLIT & SCALING
# ============================================================
print("\n" + "=" * 70)
print("STEP 11: TRAIN-TEST SPLIT & SCALING")
print("=" * 70)

# Features and target
X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']

X_onehot = df_onehot.drop('Attrition', axis=1)
y_onehot = df_onehot['Attrition']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_oh, X_test_oh, y_train_oh, y_test_oh = train_test_split(
    X_onehot, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")
print(f"Training Attrition Rate: {y_train.mean():.2%}")
print(f"Testing Attrition Rate:  {y_test.mean():.2%}")

# Scale
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X_test.columns, index=X_test.index
)

scaler_oh = StandardScaler()
X_train_oh_scaled = pd.DataFrame(
    scaler_oh.fit_transform(X_train_oh), columns=X_train_oh.columns, index=X_train_oh.index
)
X_test_oh_scaled = pd.DataFrame(
    scaler_oh.transform(X_test_oh), columns=X_test_oh.columns, index=X_test_oh.index
)

print("✅ Feature scaling completed!")


# ============================================================
# STEP 12: SMOTE — HANDLE CLASS IMBALANCE
# ============================================================
print("\n" + "=" * 70)
print("STEP 12: HANDLING CLASS IMBALANCE (SMOTE)")
print("=" * 70)

print(f"Before SMOTE — Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()}")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
X_train_oh_smote, y_train_oh_smote = smote.fit_resample(X_train_oh_scaled, y_train_oh)

print(f"After  SMOTE — Class 0: {(y_train_smote == 0).sum()}, Class 1: {(y_train_smote == 1).sum()}")
print("✅ Class imbalance handled!")


# ============================================================
# STEP 13: MODEL TRAINING
# ============================================================
print("\n" + "=" * 70)
print("STEP 13: MODEL TRAINING")
print("=" * 70)

results = {}
trained_models = {}

# --- Logistic Regression ---
print("\n🔹 Training Logistic Regression...")
start = time.time()
lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0, solver='lbfgs')
lr_model.fit(X_train_oh_smote, y_train_oh_smote)
lr_pred = lr_model.predict(X_test_oh_scaled)
lr_prob = lr_model.predict_proba(X_test_oh_scaled)[:, 1]
results['Logistic Regression'] = {
    'Accuracy': accuracy_score(y_test_oh, lr_pred),
    'Precision': precision_score(y_test_oh, lr_pred),
    'Recall': recall_score(y_test_oh, lr_pred),
    'F1-Score': f1_score(y_test_oh, lr_pred),
    'ROC-AUC': roc_auc_score(y_test_oh, lr_prob),
    'Time (s)': time.time() - start
}
trained_models['Logistic Regression'] = lr_model
print(f"  ✅ Done | AUC: {results['Logistic Regression']['ROC-AUC']:.4f}")

# --- Random Forest ---
print("\n🔹 Training Random Forest...")
start = time.time()
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight='balanced'
)
rf_model.fit(X_train_smote, y_train_smote)
rf_pred = rf_model.predict(X_test_scaled)
rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
results['Random Forest'] = {
    'Accuracy': accuracy_score(y_test, rf_pred),
    'Precision': precision_score(y_test, rf_pred),
    'Recall': recall_score(y_test, rf_pred),
    'F1-Score': f1_score(y_test, rf_pred),
    'ROC-AUC': roc_auc_score(y_test, rf_prob),
    'Time (s)': time.time() - start
}
trained_models['Random Forest'] = rf_model
print(f"  ✅ Done | AUC: {results['Random Forest']['ROC-AUC']:.4f}")

# --- Gradient Boosting ---
print("\n🔹 Training Gradient Boosting...")
start = time.time()
gb_model = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=42
)
gb_model.fit(X_train_smote, y_train_smote)
gb_pred = gb_model.predict(X_test_scaled)
gb_prob = gb_model.predict_proba(X_test_scaled)[:, 1]
results['Gradient Boosting'] = {
    'Accuracy': accuracy_score(y_test, gb_pred),
    'Precision': precision_score(y_test, gb_pred),
    'Recall': recall_score(y_test, gb_pred),
    'F1-Score': f1_score(y_test, gb_pred),
    'ROC-AUC': roc_auc_score(y_test, gb_prob),
    'Time (s)': time.time() - start
}
trained_models['Gradient Boosting'] = gb_model
print(f"  ✅ Done | AUC: {results['Gradient Boosting']['ROC-AUC']:.4f}")

# --- XGBoost ---
print("\n🔹 Training XGBoost...")
start = time.time()
xgb_model = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    use_label_encoder=False, eval_metric='logloss',
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
)
xgb_model.fit(X_train_smote, y_train_smote)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
results['XGBoost'] = {
    'Accuracy': accuracy_score(y_test, xgb_pred),
    'Precision': precision_score(y_test, xgb_pred),
    'Recall': recall_score(y_test, xgb_pred),
    'F1-Score': f1_score(y_test, xgb_pred),
    'ROC-AUC': roc_auc_score(y_test, xgb_prob),
    'Time (s)': time.time() - start
}
trained_models['XGBoost'] = xgb_model
print(f"  ✅ Done | AUC: {results['XGBoost']['ROC-AUC']:.4f}")


# ============================================================
# STEP 14: MODEL COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("STEP 14: MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results).T.sort_values('ROC-AUC', ascending=False)
print("\n" + results_df.to_string())

best_model_name = results_df['ROC-AUC'].idxmax()
print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {results_df.loc[best_model_name, 'ROC-AUC']:.4f})")


# ============================================================
# STEP 15: MODEL EVALUATION VISUALIZATIONS
# ============================================================
print("\n" + "=" * 70)
print("STEP 15: EVALUATION VISUALIZATIONS")
print("=" * 70)

# --- Bar chart + ROC curves ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
results_df[metrics_to_plot].plot(kind='bar', ax=axes[0], colormap='viridis', alpha=0.85)
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Score')
axes[0].set_ylim(0, 1.1)
axes[0].legend(loc='lower right', fontsize=9)
axes[0].tick_params(axis='x', rotation=15)

predictions_dict = {
    'Logistic Regression': (y_test_oh, lr_prob),
    'Random Forest': (y_test, rf_prob),
    'Gradient Boosting': (y_test, gb_prob),
    'XGBoost': (y_test, xgb_prob)
}
colors_roc = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']
for (name, (y_true, y_prob)), color in zip(predictions_dict.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    axes[1].plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})', color=color, linewidth=2)
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[1].set_title('ROC Curves', fontsize=14, fontweight='bold')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc='lower right', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(save_path('model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: model_comparison.png")

# --- Confusion Matrices ---
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
all_preds = {
    'Logistic Regression': (y_test_oh, lr_pred),
    'Random Forest': (y_test, rf_pred),
    'Gradient Boosting': (y_test, gb_pred),
    'XGBoost': (y_test, xgb_pred)
}
for idx, (name, (y_true, y_pred)) in enumerate(all_preds.items()):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])
    axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(save_path('confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: confusion_matrices.png")

# --- Classification Report ---
print(f"\nDetailed Classification Report — {best_model_name}:")
if best_model_name == 'Logistic Regression':
    print(classification_report(y_test_oh, lr_pred, target_names=['Stayed', 'Left']))
elif best_model_name == 'Random Forest':
    print(classification_report(y_test, rf_pred, target_names=['Stayed', 'Left']))
elif best_model_name == 'Gradient Boosting':
    print(classification_report(y_test, gb_pred, target_names=['Stayed', 'Left']))
else:
    print(classification_report(y_test, xgb_pred, target_names=['Stayed', 'Left']))


# ============================================================
# STEP 16: RISK SCORING
# ============================================================
print("\n" + "=" * 70)
print("STEP 16: RISK SCORING FRAMEWORK")
print("=" * 70)

# Use best tree model for risk scoring
tree_models = {k: v for k, v in results.items() if k != 'Logistic Regression'}
best_tree_name = max(tree_models, key=lambda k: tree_models[k]['ROC-AUC'])
best_tree_model = trained_models[best_tree_name]

print(f"Using {best_tree_name} for risk scoring")

# Score ALL employees
X_all_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
all_probabilities = best_tree_model.predict_proba(X_all_scaled)[:, 1]

# Build risk dataframe
risk_df = df_fe.copy()
risk_df['Attrition_Probability'] = all_probabilities
risk_df['Risk_Percentage'] = (all_probabilities * 100).round(2)


def assign_risk_category(prob):
    if prob < 0.30:
        return 'Low Risk'
    elif prob < 0.60:
        return 'Medium Risk'
    else:
        return 'High Risk'


risk_df['Risk_Category'] = risk_df['Attrition_Probability'].apply(assign_risk_category)

if employee_ids is not None:
    risk_df.insert(0, 'EmployeeID', employee_ids.values)
else:
    risk_df.insert(0, 'EmployeeID', range(1, len(risk_df) + 1))

print("\n📊 RISK DISTRIBUTION:")
risk_counts = risk_df['Risk_Category'].value_counts()
for cat in ['High Risk', 'Medium Risk', 'Low Risk']:
    if cat in risk_counts.index:
        count = risk_counts[cat]
        pct = count / len(risk_df) * 100
        emoji = '🔴' if cat == 'High Risk' else ('🟡' if cat == 'Medium Risk' else '🟢')
        print(f"  {emoji} {cat}: {count} employees ({pct:.1f}%)")

print(f"\n  Mean Risk: {risk_df['Risk_Percentage'].mean():.2f}%")
print(f"  Max Risk:  {risk_df['Risk_Percentage'].max():.2f}%")


# ============================================================
# STEP 17: RISK DISTRIBUTION VISUALIZATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 17: RISK VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1. Risk categories
colors_risk = {'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'}
risk_order = ['Low Risk', 'Medium Risk', 'High Risk']
risk_vals = [risk_counts.get(r, 0) for r in risk_order]
bars = axes[0].bar(risk_order, risk_vals,
                   color=[colors_risk[r] for r in risk_order], alpha=0.85)
axes[0].set_title('Risk Category Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Employees')
for bar, val in zip(bars, risk_vals):
    axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 5,
                 f'{val}\n({val / len(risk_df) * 100:.1f}%)',
                 ha='center', fontweight='bold', fontsize=11)

# 2. Histogram
axes[1].hist(risk_df['Risk_Percentage'], bins=50, color='#3498db', alpha=0.7, edgecolor='white')
axes[1].axvline(x=30, color='#f39c12', linestyle='--', linewidth=2, label='Medium (30%)')
axes[1].axvline(x=60, color='#e74c3c', linestyle='--', linewidth=2, label='High (60%)')
axes[1].set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Risk Score (%)')
axes[1].set_ylabel('Count')
axes[1].legend()

# 3. By department
if 'Department' in risk_df.columns:
    dept_risk = risk_df.groupby('Department')['Risk_Percentage'].mean().sort_values(ascending=True)
    dept_risk.plot(kind='barh', ax=axes[2], color='#e74c3c', alpha=0.8)
    axes[2].set_title('Average Risk by Department', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Average Risk Score (%)')

plt.tight_layout()
plt.savefig(save_path('risk_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: risk_distribution.png")


# ============================================================
# STEP 18: FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 70)
print("STEP 18: FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_tree_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 15 Features ({best_tree_name}):")
for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
    bar = '█' * int(row['Importance'] * 200)
    print(f"  {i + 1:2d}. {row['Feature']:30s} {row['Importance']:.4f} {bar}")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

top15 = feature_importance.head(15)
axes[0].barh(range(len(top15)), top15['Importance'].values,
             color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top15))))
axes[0].set_yticks(range(len(top15)))
axes[0].set_yticklabels(top15['Feature'].values)
axes[0].invert_yaxis()
axes[0].set_title(f'Top 15 Feature Importances ({best_tree_name})',
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Importance Score')

feature_importance['Cumulative'] = feature_importance['Importance'].cumsum()
axes[1].plot(range(len(feature_importance)), feature_importance['Cumulative'].values, 'b-o', markersize=3)
axes[1].axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
n_90 = (feature_importance['Cumulative'] <= 0.9).sum() + 1
axes[1].axvline(x=n_90, color='g', linestyle='--', label=f'{n_90} features for 90%')
axes[1].set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Number of Features')
axes[1].set_ylabel('Cumulative Importance')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(save_path('feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: feature_importance.png")


# ============================================================
# STEP 19: SHAP ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("STEP 19: SHAP EXPLAINABILITY ANALYSIS")
print("=" * 70)

try:
    import shap

    explainer = shap.TreeExplainer(best_tree_model)
    shap_values = explainer.shap_values(X_test_scaled)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Summary plot
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(shap_vals, X_test_scaled, feature_names=X.columns.tolist(),
                      show=False, max_display=15)
    plt.title(f'SHAP Feature Impact ({best_tree_name})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path('shap_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved: shap_summary.png")

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_test_scaled, feature_names=X.columns.tolist(),
                      plot_type='bar', show=False, max_display=15)
    plt.title(f'SHAP Mean Absolute Impact ({best_tree_name})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path('shap_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved: shap_bar.png")

    # Individual explanation
    if best_tree_name != 'Logistic Regression':
        probs_test = best_tree_model.predict_proba(X_test_scaled)[:, 1]
        high_risk_mask = probs_test > 0.5
        if high_risk_mask.any():
            sample_pos = np.where(high_risk_mask)[0][0]
        else:
            sample_pos = 0

        fig, ax = plt.subplots(figsize=(14, 6))
        expected_val = explainer.expected_value
        if isinstance(expected_val, list):
            expected_val = expected_val[1]

        shap_explanation = shap.Explanation(
            values=shap_vals[sample_pos],
            base_values=expected_val,
            data=X_test_scaled.iloc[sample_pos].values,
            feature_names=X.columns.tolist()
        )
        shap.waterfall_plot(shap_explanation, max_display=12, show=False)
        plt.title(f'Individual Risk Explanation (Employee #{sample_pos})',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path('shap_individual.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Saved: shap_individual.png")

except Exception as e:
    print(f"⚠️ SHAP analysis skipped: {e}")


# ============================================================
# STEP 20: SAVE ALL ARTIFACTS
# ============================================================
print("\n" + "=" * 70)
print("STEP 20: SAVING ALL ARTIFACTS")
print("=" * 70)

# Save model
joblib.dump(best_tree_model, save_path('best_model.pkl'))
print(f"✅ Saved: best_model.pkl ({best_tree_name})")

# Save scaler
joblib.dump(scaler, save_path('scaler.pkl'))
print("✅ Saved: scaler.pkl")

# Save label encoders
joblib.dump(label_encoders, save_path('label_encoders.pkl'))
print("✅ Saved: label_encoders.pkl")

# Save feature names
joblib.dump(list(X.columns), save_path('feature_names.pkl'))
print("✅ Saved: feature_names.pkl")

# Save risk scores
risk_df.to_csv(save_path('employee_risk_scores.csv'), index=False)
print("✅ Saved: employee_risk_scores.csv")

# Save model comparison
results_df.to_csv(save_path('model_comparison.csv'))
print("✅ Saved: model_comparison.csv")

# Save feature importance (drop Cumulative column before saving)
fi_save = feature_importance.drop(columns=['Cumulative'], errors='ignore')
fi_save.to_csv(save_path('feature_importance.csv'), index=False)
print("✅ Saved: feature_importance.csv")


# ============================================================
# STEP 21: EXECUTIVE SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("EXECUTIVE SUMMARY")
print("=" * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║   EMPLOYEE ATTRITION PREDICTION — EXECUTIVE SUMMARY             ║
║   Palo Alto Networks HR Analytics                                ║
╚══════════════════════════════════════════════════════════════════╝

1. PROJECT OVERVIEW
   • Analyzed {len(risk_df)} employees across multiple departments
   • Built 4 ML models; best: {best_tree_name}
   • Generated individual risk scores for every employee

2. KEY FINDINGS
   • Overall attrition rate: {df_clean['Attrition'].mean() * 100:.1f}%
   • High Risk employees:   {len(risk_df[risk_df['Risk_Category'] == 'High Risk'])}
   • Medium Risk employees: {len(risk_df[risk_df['Risk_Category'] == 'Medium Risk'])}
   • Low Risk employees:    {len(risk_df[risk_df['Risk_Category'] == 'Low Risk'])}

3. TOP ATTRITION DRIVERS
{chr(10).join([f'   {i + 1}. {row["Feature"]} ({row["Importance"]:.4f})' for i, (_, row) in enumerate(feature_importance.head(5).iterrows())])}

4. MODEL PERFORMANCE
   • Best Model:  {best_tree_name}
   • ROC-AUC:     {results[best_tree_name]['ROC-AUC']:.4f}
   • Precision:   {results[best_tree_name]['Precision']:.4f}
   • Recall:      {results[best_tree_name]['Recall']:.4f}
   • F1-Score:    {results[best_tree_name]['F1-Score']:.4f}

5. RECOMMENDATIONS
   a) IMMEDIATE: Stay interviews with top high-risk employees
   b) SHORT-TERM: Targeted retention programs, compensation review
   c) LONG-TERM: Quarterly risk re-assessment, manager dashboards

══════════════════════════════════════════════════════════════════
✅ All analysis complete! Now run: streamlit run streamlit_app.py
══════════════════════════════════════════════════════════════════
""")