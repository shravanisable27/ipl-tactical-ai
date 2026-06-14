import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score,
                              brier_score_loss, mean_squared_error, mean_absolute_error)
from sklearn.calibration import CalibrationDisplay
import xgboost as xgb
import os

print('All imports successful')

# --- CELL 3 ---
conn = sqlite3.connect('../database/ipl_data.db')
df_deliveries = pd.read_sql_query('SELECT * FROM deliveries;', conn)
df_matches = pd.read_sql_query('SELECT id AS match_id, winner, venue, toss_winner, toss_decision FROM matches;', conn)
conn.close()
print(f'Deliveries: {len(df_deliveries):,} rows')
print(f'Matches: {len(df_matches):,} rows')

# --- CELL 4 ---
conditions = [
    (df_deliveries['over'] < 6),
    (df_deliveries['over'] >= 6) & (df_deliveries['over'] < 15),
    (df_deliveries['over'] >= 15)
]
choices = ['Powerplay', 'Middle_Overs', 'Death_Overs']
df_deliveries['match_phase'] = np.select(conditions, choices, default='Middle_Overs')
df_deliveries['is_dot'] = np.where(df_deliveries['batsman_runs'] == 0, 1, 0)

player_profiles = df_deliveries.groupby('batter').agg(
    total_runs=('batsman_runs', 'sum'),
    balls_faced=('match_id', 'count'),
    total_dots=('is_dot', 'sum')
).reset_index()
player_profiles['player_strike_rate'] = (player_profiles['total_runs'] / player_profiles['balls_faced']) * 100
player_profiles['player_dot_percent'] = (player_profiles['total_dots'] / player_profiles['balls_faced']) * 100
player_features = player_profiles[['batter', 'player_strike_rate', 'player_dot_percent']]

df_deliveries = pd.merge(df_deliveries, df_matches, on='match_id', how='left')
df_deliveries['is_winner'] = np.where(df_deliveries['batting_team'] == df_deliveries['winner'], 1, 0)

target_df = df_deliveries[df_deliveries['inning'] == 1].groupby('match_id').agg(
    innings_1_score=('total_runs', 'sum')
).reset_index()
target_df['target_score'] = target_df['innings_1_score'] + 1

df_chase = df_deliveries[df_deliveries['inning'] == 2].copy()
df_chase = pd.merge(df_chase, target_df[['match_id', 'target_score']], on='match_id', how='left')
df_chase = df_chase.dropna(subset=['target_score'])

df_chase['current_score'] = df_chase.groupby('match_id')['total_runs'].cumsum().shift(1).fillna(0)
df_chase['wickets_fallen'] = df_chase.groupby('match_id')['is_wicket'].cumsum().shift(1).fillna(0)

df_chase['balls_bowled'] = (df_chase['over'] * 6) + df_chase['ball']
df_chase['balls_left'] = 120 - df_chase['balls_bowled']
df_chase['wickets_left'] = 10 - df_chase['wickets_fallen']
df_chase['runs_needed'] = np.where(df_chase['target_score'] - df_chase['current_score'] < 0, 0,
                                    df_chase['target_score'] - df_chase['current_score'])

df_chase['crr'] = np.where(df_chase['balls_bowled'] > 0,
                            (df_chase['current_score'] * 6) / df_chase['balls_bowled'], 0)
df_chase['rrr'] = np.where(df_chase['balls_left'] > 0,
                            (df_chase['runs_needed'] * 6) / df_chase['balls_left'], 0)

df_chase['pressure_index'] = np.where(df_chase['crr'] > 0, df_chase['rrr'] / df_chase['crr'], df_chase['rrr'])

df_chase['phase_powerplay'] = (df_chase['over'] < 6).astype(int)
df_chase['phase_death'] = (df_chase['over'] >= 15).astype(int)

df_master = pd.merge(df_chase, player_features, on='batter', how='left')
df_master['player_strike_rate'] = df_master['player_strike_rate'].fillna(125.0)
df_master['player_dot_percent'] = df_master['player_dot_percent'].fillna(35.0)

venue_avg = df_deliveries[df_deliveries['inning'] == 2].merge(
    df_matches[['match_id', 'venue']], on='match_id', how='left'
).groupby(['match_id', 'venue'])['total_runs'].sum().reset_index()
venue_avg = venue_avg.groupby('venue')['total_runs'].mean().reset_index().rename(
    columns={'total_runs': 'venue_avg_2nd_score'})
df_master = pd.merge(df_master, df_matches[['match_id', 'venue']], on='match_id', how='left')
df_master = pd.merge(df_master, venue_avg, on='venue', how='left')
df_master['venue_avg_2nd_score'] = df_master['venue_avg_2nd_score'].fillna(df_master['venue_avg_2nd_score'].median())

df_master = df_master[df_master['balls_left'] >= 0]
print(f'Final dataset shape: {df_master.shape}')
print(f'Target balance: {df_master["is_winner"].value_counts().to_dict()}')

# --- CELL 5 ---
FEATURE_COLS = [
    'runs_needed', 'balls_left', 'wickets_left', 'crr', 'rrr',
    'player_strike_rate', 'player_dot_percent',
    'pressure_index', 'phase_powerplay', 'phase_death', 'venue_avg_2nd_score'
]

X = df_master[FEATURE_COLS].fillna(0)
y = df_master['is_winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                   use_label_encoder=False, eval_metric='logloss', random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, clf in models.items():
    X_fit = X_train_scaled if name == 'Logistic Regression' else X_train.values
    cv_scores = cross_validate(clf, X_fit, y_train,
                               cv=cv,
                               scoring=['accuracy', 'roc_auc'],
                               return_train_score=False)
    results[name] = {
        'cv_acc_mean': cv_scores['test_accuracy'].mean(),
        'cv_acc_std': cv_scores['test_accuracy'].std(),
        'cv_auc_mean': cv_scores['test_roc_auc'].mean(),
        'cv_auc_std': cv_scores['test_roc_auc'].std(),
    }
    print(f'{name}: CV Acc={results[name]["cv_acc_mean"]:.4f} +/- {results[name]["cv_acc_std"]:.4f} | AUC={results[name]["cv_auc_mean"]:.4f}')

results_df = pd.DataFrame(results).T
print('\nCross-validation summary:')
print(results_df.round(4))

# --- CELL 6 ---
trained_models = {}
test_metrics = {}

for name, clf in models.items():
    X_tr = X_train_scaled if name == 'Logistic Regression' else X_train.values
    X_te = X_test_scaled if name == 'Logistic Regression' else X_test.values
    clf.fit(X_tr, y_train)
    trained_models[name] = clf
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    test_metrics[name] = {
        'test_accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'brier_score': brier_score_loss(y_test, y_prob),
    }

metrics_df = pd.DataFrame(test_metrics).T
print('Test set performance:')
print(metrics_df.round(4))

best_model_name = metrics_df['roc_auc'].idxmax()
print(f'\nBest model by ROC-AUC: {best_model_name}')

# --- CELL 7 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Feature Importance - Top Models', fontsize=13, fontweight='bold')

for ax, model_name in zip(axes, ['Random Forest', 'XGBoost']):
    clf = trained_models[model_name]
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)
    ax.barh([FEATURE_COLS[i] for i in sorted_idx], importances[sorted_idx], color='#2196F3')
    ax.set_title(model_name)
    ax.set_xlabel('Importance')

plt.tight_layout()
plt.savefig('../database/feature_importance.png', dpi=120, bbox_inches='tight')
plt.close()
print('Feature importance chart saved.')

# --- CELL 8 ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

for name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
    clf = trained_models[name]
    X_te = X_test_scaled if name == 'Logistic Regression' else X_test.values
    y_prob = clf.predict_proba(X_te)[:, 1]
    CalibrationDisplay.from_predictions(y_test, y_prob, n_bins=10, ax=ax, name=name)

ax.set_title('Model Calibration - Reliability Diagram')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.legend()
plt.tight_layout()
plt.savefig('../database/calibration_plot.png', dpi=120, bbox_inches='tight')
plt.close()
print('Calibration chart saved.')

# --- CELL 9 ---
best_clf = trained_models[best_model_name]
best_X_scaler = scaler if best_model_name == 'Logistic Regression' else None

def predict_win_prob(clf, scaler_obj, scenario_values):
    arr = np.array([scenario_values])
    if scaler_obj is not None:
        arr = scaler_obj.transform(arr)
    return clf.predict_proba(arr)[0][1] * 100

base_scenario = {
    'runs_needed': 50, 'balls_left': 36, 'wickets_left': 5,
    'crr': 6.43, 'rrr': 8.33,
    'player_strike_rate': 0, 'player_dot_percent': 0,
    'pressure_index': 8.33/6.43, 'phase_powerplay': 0, 'phase_death': 1,
    'venue_avg_2nd_score': 160
}

scenarios = {
    'Anchor batter (SR 115, dot 42%)': dict(base_scenario, player_strike_rate=115, player_dot_percent=42),
    'Finisher batter (SR 165, dot 28%)': dict(base_scenario, player_strike_rate=165, player_dot_percent=28),
    'Match winner (SR 190, dot 20%)': dict(base_scenario, player_strike_rate=190, player_dot_percent=20),
}

print('Scenario simulation - Chasing Team Win Probability')
print('-' * 55)
for label, scenario in scenarios.items():
    values = [scenario[col] for col in FEATURE_COLS]
    prob = predict_win_prob(best_clf, best_X_scaler if best_model_name == 'Logistic Regression' else None, values)
    print(f'{label}: {prob:.2f}%')

# --- CELL 10 ---
conn = sqlite3.connect('../database/ipl_data.db')

inn1_scores = pd.read_sql_query("""
    SELECT match_id, SUM(total_runs) AS innings_1_score
    FROM deliveries WHERE inning = 1
    GROUP BY match_id
""", conn)

matches_meta = pd.read_sql_query("""
    SELECT id AS match_id, venue, season, batting_team, toss_winner, toss_decision
    FROM (
        SELECT d.match_id, m.venue, m.season, d.batting_team, m.toss_winner, m.toss_decision
        FROM deliveries d
        JOIN matches m ON d.match_id = m.id
        WHERE d.inning = 1
        GROUP BY d.match_id
    )
""", conn)

conn.close()

inn1_df = pd.merge(inn1_scores, matches_meta, on='match_id', how='inner')

venue_avg_scores = inn1_df.groupby('venue')['innings_1_score'].mean().reset_index().rename(
    columns={'innings_1_score': 'venue_avg_score'})
batting_team_avg = inn1_df.groupby('batting_team')['innings_1_score'].mean().reset_index().rename(
    columns={'innings_1_score': 'team_avg_score'})

inn1_df = pd.merge(inn1_df, venue_avg_scores, on='venue', how='left')
inn1_df = pd.merge(inn1_df, batting_team_avg, on='batting_team', how='left')
inn1_df['toss_bats_first'] = ((inn1_df['toss_winner'] == inn1_df['batting_team']) &
                               (inn1_df['toss_decision'] == 'bat')).astype(int)

inn1_df['season_year'] = inn1_df['season'].apply(
    lambda s: int(str(s).split('/')[-1]) if '/' in str(s) else int(str(s)[:4])
)

REG_FEATURES = ['venue_avg_score', 'team_avg_score', 'toss_bats_first', 'season_year']
X_reg = inn1_df[REG_FEATURES].fillna(inn1_df[REG_FEATURES].median())
y_reg = inn1_df['innings_1_score']

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_models = {
    'Ridge': Ridge(alpha=1.0),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'XGBoost Regressor': xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
}

reg_results = {}
for name, reg in reg_models.items():
    reg.fit(X_tr_r, y_tr_r)
    preds = reg.predict(X_te_r)
    rmse = np.sqrt(mean_squared_error(y_te_r, preds))
    mae = mean_absolute_error(y_te_r, preds)
    reg_results[name] = {'RMSE': round(rmse, 2), 'MAE': round(mae, 2)}
    print(f'{name}: RMSE={rmse:.2f}, MAE={mae:.2f}')

best_reg_name = min(reg_results, key=lambda k: reg_results[k]['RMSE'])
best_reg = reg_models[best_reg_name]
print(f'\nBest innings-1 regressor: {best_reg_name}')

# --- CELL 11 ---
os.makedirs('../database', exist_ok=True)

best_clf_final = trained_models[best_model_name]
joblib.dump(best_clf_final, '../database/win_probability_model.joblib')
joblib.dump(scaler, '../database/win_probability_scaler.joblib')
joblib.dump(FEATURE_COLS, '../database/feature_columns.joblib')

joblib.dump(best_reg, '../database/innings1_score_model.joblib')

print(f'Win probability model ({best_model_name}) saved to database/')
print(f'Innings-1 regressor ({best_reg_name}) saved to database/')
print('Feature column list saved to database/feature_columns.joblib')
print('ALL CELLS COMPLETE')
