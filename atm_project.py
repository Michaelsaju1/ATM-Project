import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    KFold, 
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    average_precision_score, 
    confusion_matrix,
    precision_score
)
from scipy.stats import linregress
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data():
    """Loads all source parquet files into a dictionary of DataFrames."""
    print("Loading data...")
    try:
        data = {
            "advances": pd.read_parquet("advances.parquet", engine="fastparquet"),
            "applications": pd.read_parquet("applications.parquet", engine="fastparquet"),
            "balances": pd.read_parquet("balances.parquet", engine="fastparquet"),
            "labels": pd.read_parquet(
                "labels.parquet", 
                engine="fastparquet", 
                columns=["advance_id", "repaid_full_30d"]
            ),
        }
        
        # --- NEW: Load and combine all transaction files ---
        print("Loading and combining transaction files...")
        txn_files = [
            "transactions_sample.parquet",
            "transactions_sample_2.parquet",
            "transactions_sample_3.parquet",
            "transactions_sample_4.parquet"
        ]
        all_txns = []
        for f in txn_files:
            try:
                all_txns.append(pd.read_parquet(f, engine="fastparquet"))
            except FileNotFoundError:
                print(f"Warning: File {f} not found. Skipping.")
        
        if not all_txns:
            raise FileNotFoundError("No transaction files found.")
            
        data["transactions"] = pd.concat(all_txns, ignore_index=True)
        print(f"All data loaded successfully. Combined {len(all_txns)} transaction files.")
        return data
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure all .parquet files are in the same directory.")
        return None
    except ImportError as e:
        print(f"Error: {e}. You may need to install 'fastparquet' and 'scipy'.")
        return None


def get_base_table(df_labels, df_advances):
    """Creates the base table of the first advance for each user."""
    print("Creating base table...")
    df_advances_renamed = df_advances.rename(columns={'request_id': 'advance_id'})
    df_merged = pd.merge(df_labels, df_advances_renamed, on='advance_id', how='left')
    
    df_merged = df_merged.dropna(subset=['user_id', 'underwritten_at'])
    
    df_merged['underwritten_at'] = pd.to_datetime(df_merged['underwritten_at'])
    df_merged['disbursed_at'] = pd.to_datetime(df_merged['disbursed_at'])
    df_sorted = df_merged.sort_values(by=['user_id', 'underwritten_at'], ascending=True)
    first_advances_df = df_sorted.drop_duplicates(subset=['user_id'], keep='first').copy()
    print(f"Base table created with {len(first_advances_df)} unique users (first advances).")
    return first_advances_df

def create_application_features(base_df, df_applications):
    """Engineers simplified features from applications *before* underwriting."""
    print("Engineering pre-underwriting application features...")
    df_applications['created_at'] = pd.to_datetime(df_applications['created_at'])
    
    apps_merged = df_applications.merge(
        base_df[['user_id', 'underwritten_at']], 
        on='user_id', 
        how='inner'
    )
    
    # --- 🛡️ LEAKAGE CHECK 🛡️ ---
    pre_apps = apps_merged[
        apps_merged['created_at'] < apps_merged['underwritten_at']
    ].copy()
    
    if not pre_apps.empty:
        assert (pre_apps['created_at'] < pre_apps['underwritten_at']).all(), \
            "Data Leakage Detected in Applications!"
    
    print(f"Found {len(pre_apps)} pre-underwriting applications.")
    
    if not pre_apps.empty:
        pre_apps['days_before_underwrite'] = (
            pre_apps['underwritten_at'] - pre_apps['created_at']
        ).dt.days
        
        app_agg_features = pre_apps.groupby('user_id').agg(
            pre_app_count=('created_at', 'count'),
            days_since_first_pre_app=('days_before_underwrite', 'max'),
            days_since_last_pre_app=('days_before_underwrite', 'min')
        )
        base_df = base_df.merge(app_agg_features, on='user_id', how='left')

    return base_df

def create_balance_features(features_df, df_balances):
    """
    Engineers simplified features from balance snapshots *before* underwriting.
    --- SIMPLIFIED: ONLY USES LATEST SNAPSHOT ---
    """
    print("Engineering simplified pre-underwriting balance features...")
    df_balances['updated_at'] = pd.to_datetime(df_balances['updated_at'])
    
    df_balances_merged = df_balances.merge(
        features_df[['user_id', 'underwritten_at']], 
        on='user_id', 
        how='inner'
    )
    
    # --- 🛡️ LEAKAGE CHECK 🛡️ ---
    pre_balances = df_balances_merged[
        df_balances_merged['updated_at'] < df_balances_merged['underwritten_at']
    ].copy()
    
    if not pre_balances.empty:
        assert (pre_balances['updated_at'] < pre_balances['underwritten_at']).all(), \
            "Data Leakage Detected in Balances!"
            
    print(f"Found {len(pre_balances)} pre-underwriting balance snapshots for {pre_balances['user_id'].nunique()} users.")
    
    if not pre_balances.empty:
        pre_balances_sorted = pre_balances.sort_values(by=['user_id', 'updated_at'], ascending=True)
        latest_pre_balances = pre_balances_sorted.drop_duplicates(subset=['user_id'], keep='last').copy()
        
        latest_pre_balances['days_since_last_balance'] = (
            latest_pre_balances['underwritten_at'] - latest_pre_balances['updated_at']
        ).dt.days
        
        balance_features_last = latest_pre_balances[[
            'user_id', 'available_balance', 'current_balance', 'days_since_last_balance'
        ]].rename(columns={
            'available_balance': 'last_available_balance',
            'current_balance': 'last_current_balance'
        })
        
        features_df = features_df.merge(balance_features_last, on='user_id', how='left')

    return features_df

def create_transaction_features(features_df, df_transactions):
    """
    Engineers simplified features from transactions *before* underwriting.
    --- SIMPLIFIED: NO PIVOTS, NO TRENDS, NO STD/MIN/MAX ---
    """
    print("Engineering simplified pre-underwriting transaction features...")
    df_transactions['date'] = pd.to_datetime(df_transactions['date'])
    
    df_txns_merged = df_transactions.merge(
        features_df[['user_id', 'underwritten_at']], 
        on='user_id', 
        how='inner'
    )
    
    # --- 🛡️ LEAKAGE CHECK 🛡️ ---
    pre_txns = df_txns_merged[
        df_txns_merged['date'] < df_txns_merged['underwritten_at']
    ].copy()
    
    if not pre_txns.empty:
        assert (pre_txns['date'] < pre_txns['underwritten_at']).all(), \
            "Data Leakage Detected in Transactions!"
            
    print(f"Found {len(pre_txns)} pre-underwriting transactions from {pre_txns['user_id'].nunique()} users.")
    
    if not pre_txns.empty:
        # 1. Inflow/Outflow amounts
        pre_txns['inflow'] = pre_txns['amount'].apply(lambda x: x if x < 0 else 0).abs()
        pre_txns['outflow'] = pre_txns['amount'].apply(lambda x: x if x > 0 else 0)
        
        # 2. Days from transaction to underwriting
        pre_txns['days_before_underwriting'] = (
            pre_txns['underwritten_at'] - pre_txns['date']
        ).dt.days
        
        # 3. Aggregate simple features by user
        txn_agg_features = pre_txns.groupby('user_id').agg(
            pre_txn_count=('id', 'count'),
            pre_txn_net_sum=('amount', 'sum'), # This is NET flow
            pre_txn_inflow_sum=('inflow', 'sum'),
            pre_txn_outflow_sum=('outflow', 'sum'),
            pre_txn_user_history_days=('days_before_underwriting', 'max'),
            pre_txn_days_since_last=('days_before_underwriting', 'min')
        )
        
        features_df = features_df.merge(txn_agg_features, on='user_id', how='left')

    return features_df

def finalize_features(features_df):
    """Cleans up the final feature set for modeling."""
    print("Finalizing feature set...")
    
    features_df['underwrite_month'] = features_df['underwritten_at'].dt.month
    features_df['underwrite_dayofweek'] = features_df['underwritten_at'].dt.dayofweek
    features_df['underwrite_hour'] = features_df['underwritten_at'].dt.hour
    
    features_df['amount'] = pd.to_numeric(features_df['amount'].replace(r'[^\\d\\.]', '', regex=True), errors='coerce')
    
    # --- New Ratio Features (Post-Merge) ---
    safe_div = 1e-6 # To avoid division by zero
    features_df['balance_ratio'] = features_df.get('last_available_balance', 0) / (features_df.get('last_current_balance', 0) + safe_div)
    
    # Txn Ratios
    features_df['pre_txn_inflow_ratio_sum'] = features_df.get('pre_txn_inflow_sum', 0) / (features_df.get('pre_txn_inflow_sum', 0) + features_df.get('pre_txn_outflow_sum', 0) + safe_div)
    features_df['pre_txn_avg_txns_per_day'] = features_df.get('pre_txn_count', 0) / (features_df.get('pre_txn_user_history_days', 0) + safe_div)
    features_df['pre_txn_avg_net_flow_per_day'] = features_df.get('pre_txn_net_sum', 0) / (features_df.get('pre_txn_user_history_days', 0) + safe_div)

    
    y = features_df['repaid_full_30d'].astype(int)
    
    cols_to_drop = [
        'advance_id',
        'repaid_full_30d',
        'request_id',
        'underwritten_at',
        'disbursed_at',
        'user_id',
        'amount'
    ]
    
    all_cols = set(features_df.columns)
    cols_to_drop = [col for col in cols_to_drop if col in all_cols]
    
    X = features_df.drop(columns=cols_to_drop)
    
    X = X.fillna(0)
    X.columns = X.columns.astype(str)
    
    print(f"Final feature set shape (simplified): {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    
    return X, y

def calculate_precision_at_k(y_true, y_proba, k_values):
    """Calculates precision at different probability cutoffs (top K percent)."""
    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    df = df.sort_values('y_proba', ascending=False)
    
    results = {}
    for k in k_values:
        n_samples = int(len(df) * k)
        if n_samples == 0:
            results[f'Precision @ Top {int(k*100)}%'] = np.nan
            continue
            
        top_k_df = df.head(n_samples)
        precision = top_k_df['y_true'].mean()
        results[f'Precision @ Top {int(k*100)}%'] = precision
        
    return results

def evaluate_model(X, y):
    """Runs cross-validation and test-set evaluation for LightGBM."""
    print("\n--- Model Evaluation (LightGBM Only) ---")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    model = LGBMClassifier(random_state=42, n_estimators=100, verbosity=-1, num_leaves=31)
    
    print("\n--- 5-Fold Cross-Validation (PR AUC) ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='average_precision', n_jobs=-1)
    print(f"LGBM: Mean PR AUC = {cv_scores.mean():.4f} (Std = {cv_scores.std():.4f})")
        
    print("\n--- Test Set Performance ---")
    
    k_values = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    
    print(f"\n--- Training and Evaluating: LightGBM ---")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    pr_auc = average_precision_score(y_test, y_proba)
    print(classification_report(y_test, y_pred))
    print(f"PR AUC (AUPRC): {pr_auc:.4f}")
    
    # --- Display Precision at K Results Table ---
    print("\n\n--- 🎯 FINAL TARGET METRIC: Precision at K Cutoffs ---")
    precision_scores = calculate_precision_at_k(y_test, y_proba, k_values)
    precision_df = pd.DataFrame(precision_scores, index=["LightGBM"]).T
    print(precision_df.to_markdown(floatfmt=".2%"))
    
    # --- Feature Importance ---
    print("\n--- 📊 Feature Importances ---")
    
    importances = model.feature_importances_
    feature_imp = pd.Series(importances, index=X.columns).nlargest(20)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=feature_imp.values, y=feature_imp.index)
    plt.title("LightGBM - Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("lgbm_feature_importance.png")
    plt.show()

def main():
    """Main function to run the entire pipeline."""
    data = load_data()
    if data is None:
        return
    
    base_df = get_base_table(data["labels"], data["advances"])
    features_df = create_application_features(base_df, data["applications"])
    features_df = create_balance_features(features_df, data["balances"])
    
    # --- RE-ENABLED: Using simplified transaction feature function ---
    features_df = create_transaction_features(features_df, data["transactions"])
    
    X, y = finalize_features(features_df)
    
    evaluate_model(X, y)

if __name__ == "__main__":
    main()