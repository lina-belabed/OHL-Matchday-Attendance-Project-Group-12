"""
src/model.py
Run with: python src/model.py

Trains Ridge, QR35, and QR25 models from data/cleaned/engineered_df.csv
and saves them as .pkl artifacts in the models/ directory.

Trained on all 71 matches (production model — evaluation was done separately in notebooks).
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH = os.path.join(ROOT, 'data', 'cleaned', 'engineered_df.csv')
MODELS_DIR = os.path.join(ROOT, 'models')

# ── Feature sets (must match notebook definitions exactly) ─────────────────
NUMERIC_FEATURES = [
    'kickoff_hour', 'is_playoff', 'is_weekend', 'is_school_holiday_flanders',
    'has_promotion', 'weather_score', 'avg_ohl_interest_7d',
    'pre_match_article_count', 'home_win_rate_last3', 'ohl_season_points',
    'opp_ppg_vs_ohl', 'rolling_avg_attendance_last3', 'matchday_normalized',
    'last_h2h_goal_margin', 'ohl_points_per_game', 'has_campaign',
    'pre_match_interest_ratio',
]
CAT_FEATURES = ['opponent_grouped', 'last_h2h_result']  # OHE with drop_first=True

FEAT_14 = [
    'rolling_avg_attendance_last3', 'ohl_points_per_game', 'home_win_rate_last3',
    'matchday_normalized', 'is_weekend', 'is_school_holiday_flanders',
    'academic_week', 'has_promotion', 'weather_score', 'weather_temp_deviation',
    'avg_ohl_interest_7d', 'pre_match_interest_ratio', 'opp_ppg_vs_ohl',
    'last_h2h_goal_margin',
]

TARGET = 'tickets_scanned'


def preprocess(df):
    df = df.copy()

    # Derive kickoff_hour from time string "HH:MM:SS"
    df['kickoff_hour'] = df['kickoff_time_local'].str.split(':').str[0].astype(int)

    # Derive is_playoff from stage
    df['is_playoff'] = (df['stage'] == 'Conference League Play-off Group').astype(int)

    # Fill nulls exactly as in notebooks
    df['last_h2h_result'] = df['last_h2h_result'].fillna('Unknown')
    df['home_win_rate_last3'] = df['home_win_rate_last3'].fillna(0)
    df['pre_match_article_count'] = df['pre_match_article_count'].fillna(
        df['pre_match_article_count'].median()
    )
    df['rolling_avg_attendance_last3'] = df['rolling_avg_attendance_last3'].fillna(
        df['rolling_avg_attendance_last3'].median()
    )

    return df


def build_ridge_matrix(df):
    X_num = df[NUMERIC_FEATURES]
    X_cat = pd.get_dummies(df[CAT_FEATURES], drop_first=True)
    return pd.concat([X_num, X_cat], axis=1)


def build_qr_matrix(df):
    return df[FEAT_14].copy()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = preprocess(df)

    print(f"Dataset: {len(df)} matches")

    # ── Ridge (36 features with OHE) ───────────────────────────────────────
    X_ridge = build_ridge_matrix(df)
    y = df[TARGET]
    feature_cols_ridge = X_ridge.columns.tolist()
    print(f"Ridge feature count: {len(feature_cols_ridge)}")

    ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000])
    ridge.fit(X_ridge, y)
    print(f"Ridge best alpha: {ridge.alpha_:.4f}  |  Train MAE: {abs(y - ridge.predict(X_ridge)).mean():.0f}")

    joblib.dump(ridge, os.path.join(MODELS_DIR, 'ridge_model.pkl'))
    joblib.dump(feature_cols_ridge, os.path.join(MODELS_DIR, 'feature_columns_ridge.pkl'))
    print("Saved: ridge_model.pkl, feature_columns_ridge.pkl")

    # ── Quantile models (14 features, scaled) ──────────────────────────────
    X_qr = build_qr_matrix(df)
    feature_cols_qr = X_qr.columns.tolist()
    print(f"QR feature count: {len(feature_cols_qr)}")

    for q, name in [(0.35, 'qr35'), (0.25, 'qr25')]:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('qr', QuantileRegressor(quantile=q, alpha=0.1, solver='highs')),
        ])
        pipe.fit(X_qr, y)
        mae = abs(y - pipe.predict(X_qr)).mean()
        print(f"{name.upper()} (q={q})  |  Train MAE: {mae:.0f}")
        joblib.dump(pipe, os.path.join(MODELS_DIR, f'{name}_model.pkl'))
        print(f"Saved: {name}_model.pkl")

    joblib.dump(feature_cols_qr, os.path.join(MODELS_DIR, 'feature_columns_qr.pkl'))
    print("Saved: feature_columns_qr.pkl")

    print("\nAll models saved to models/")


if __name__ == '__main__':
    main()
