
# PROJET PM2.5 - PHASE 1


# --- INITIALISATION ET MONTAGE DU DRIVE ---
import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print("Mode local : pas de montage Google Drive.")

path_drive = '/content/drive/MyDrive'
if os.path.exists(path_drive):
    elements = os.listdir(path_drive)
    suggestions = [el for el in elements if 'projet' in el.lower() or 'pm2' in el.lower()]
    nom_du_dossier = suggestions[0] if suggestions else 'Projet_PM25_Ferouh_Lina_Rommaisa '
else:
    nom_du_dossier = 'Projet_PM25_Ferouh_Lina_Rommaisa '

BASE = f'/content/drive/MyDrive/{nom_du_dossier}'
print(f"Dossier de travail : {BASE}")


# PARTIE 1 : BLOC A (FEROUH) - Nettoyage et Folds



print("\n--- DÉBUT DU BLOC A (FEROUH) ---")

chemin_fichier = f'{BASE}/data/raw/combined_gee_pm25.csv'
df = pd.read_csv(chemin_fichier)

# Nettoyage
df_clean = df[(df['pm25'] > 0) & (df['pm25'] < 500)].copy()
print(f"Lignes supprimées : {len(df) - len(df_clean)}")
df_clean.to_csv(f'{BASE}/data/processed/pm25_clean.csv', index=False)

# Filtrage
df_clean['date'] = pd.to_datetime(df_clean['date'])
df_filtre = df_clean.groupby('id').filter(lambda x: len(x) >= 100)
df_filtre.to_csv(f'{BASE}/data/processed/pm25_clean_final.csv', index=False)
print(f"Stations gardées : {df_filtre['id'].nunique()}")

# Folds
df_temp = df_filtre.sort_values('date').reset_index(drop=True)
tscv = TimeSeriesSplit(n_splits=5, gap=1)
folds_temporal = {f"fold_{i}": {"train_indices": tr.tolist(), "test_indices": te.tolist()} for i, (tr, te) in enumerate(tscv.split(df_temp))}
with open(f'{BASE}/folds/folds_temporal.json', 'w') as f: json.dump(folds_temporal, f)

gkf = GroupKFold(n_splits=5)
folds_spatial = {f"fold_{i}": {"train_indices": tr.tolist(), "test_indices": te.tolist()} for i, (tr, te) in enumerate(gkf.split(df_filtre, groups=df_filtre['id']))}
with open(f'{BASE}/folds/folds_spatial.json', 'w') as f: json.dump(folds_spatial, f)

print(" Folds créés.")


# PARTIE 2 : BLOC B

print("\n--- DÉBUT DU BLOC B (LINA) ---")

# 1. Statistiques
stats_list = []
for col in df_filtre.select_dtypes(include=['float64', 'int64']).columns:
    serie = df_filtre[col]
    q1, median, q3 = serie.quantile(0.25), serie.quantile(0.50), serie.quantile(0.75)
    stats_list.append({
        'variable': col, 'n_missing': serie.isnull().sum(), 'mean': serie.mean(), 
        'std': serie.std(), 'min': serie.min(), 'max': serie.max(), 'iqr': q3 - q1
    })
df_stats = pd.DataFrame(stats_list)

# 2. Outliers
def compter_outliers(serie):
    serie_propre = serie.dropna()
    if len(serie_propre) < 10: return 0, 0, 0, 0
    q1, q3 = serie_propre.quantile(0.25), serie_propre.quantile(0.75)
    iqr = q3 - q1
    n_iqr = ((serie_propre < q1 - 1.5*iqr) | (serie_propre > q3 + 1.5*iqr)).sum()
    n_z = (np.abs(stats.zscore(serie_propre)) > 3).sum()
    X = serie_propre.values.reshape(-1, 1)
    n_if = (IsolationForest(random_state=42, n_jobs=-1).fit_predict(X) == -1).sum()
    n_lof = (LocalOutlierFactor(n_neighbors=20, n_jobs=-1).fit_predict(X) == -1).sum()
    return n_iqr, n_z, n_if, n_lof

res_out = df_stats['variable'].apply(lambda x: compter_outliers(df_filtre[x]))
df_stats[['n_out_iqr', 'n_out_z', 'n_out_if', 'n_out_lof']] = pd.DataFrame(res_out.tolist(), index=df_stats.index)

# 3. Preprocessing
train_indices = folds_temporal['fold_0']['train_indices']
test_indices = folds_temporal['fold_0']['test_indices']
df_train = df_temp.iloc[train_indices].copy()
df_test = df_temp.iloc[test_indices].copy()

cols_a_imputer = ['NO2', 'CO', 'O3', 'AOD']
imputer = SimpleImputer(strategy='median')
df_train[cols_a_imputer] = imputer.fit_transform(df_train[cols_a_imputer])
df_test[cols_a_imputer] = imputer.transform(df_test[cols_a_imputer])
print(" Preprocessing Anti-Leakage terminé.")

# 4. GRAPHIQUES :
print("Génération des graphiques de Lina...")
dossier_figures = f'{BASE}/results/B00-00/Lina/figures'
os.makedirs(dossier_figures, exist_ok=True)
sns.set_theme(style="whitegrid")

# Histogrammes
variables_a_tracer = ['pm25', 'AOD', 'temperature_celsius', 'NO2', 'CO', 'O3']
for var in variables_a_tracer:
    if var in df_filtre.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df_filtre[var].dropna(), bins=50, kde=True, color='skyblue')
        plt.title(f'Distribution de {var}')
        plt.tight_layout()
        plt.savefig(f'{dossier_figures}/hist_{var}.png')
        plt.close()

# Boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_filtre[['pm25', 'NO2', 'CO', 'O3']], palette="Set2")
plt.yscale('log')
plt.title('Boxplots des Polluants (détection des outliers)')
plt.tight_layout()
plt.savefig(f'{dossier_figures}/boxplots_polluants.png')
plt.close()

# Heatmap
plt.figure(figsize=(12, 10))
cols_num = df_filtre.select_dtypes(include=['float64', 'int64']).columns
sns.heatmap(df_filtre[cols_num].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Matrice de corrélation')
plt.tight_layout()
plt.savefig(f'{dossier_figures}/matrice_correlation.png')
plt.close()
print(f" Graphiques Lina sauvegardés dans {dossier_figures}")

# PARTIE 3 : BLOC C - EDA Temporelle ET GRAPHIQUES

print("\n--- DÉBUT DU BLOC C (ROMMAISA) ---")

def regularize(series_by_date, max_gap=3):
    s = series_by_date.sort_index().asfreq("D")
    isn = s.isna()
    block = (~isn).cumsum()
    gap_len = isn.groupby(block).transform("sum")
    f = s.interpolate(method="linear", limit_area="inside")
    f[isn & (gap_len > max_gap)] = np.nan
    return f

def adf_test(x):
    stat, p, lag, nobs, crit, _ = adfuller(x.values, regression="c", autolag="AIC")
    return {"adf_stat": stat, "adf_pvalue": p}

def kpss_test(x):
    stat, p, lag, crit = kpss(x.values, regression="c", nlags="auto")
    return {"kpss_stat": stat, "kpss_pvalue": p}

# Test sur la première station
station_id = df_filtre['id'].unique()[0]
g = df_filtre[df_filtre["id"] == station_id].set_index("date")["pm25"]
seg_reg = regularize(g, max_gap=3)
seg_clean = seg_reg.dropna()

print(f"ADF p-value (Station {station_id}) : {adf_test(seg_clean)['adf_pvalue']:.4f}")

# GRAPHIQUES DE ROMMAISA
print("Génération des graphiques de Rommaisa...")
dossier_fig_rom = f'{BASE}/results/B00-00/Rommaisa/figures'
os.makedirs(dossier_fig_rom, exist_ok=True)

# 1. Série temporelle
plt.figure(figsize=(12, 5))
plt.plot(seg_reg.index, seg_reg.values, color='#c0392b')
plt.title(f'Évolution du PM2.5 (Station {station_id})')
plt.ylabel('PM2.5 (µg/m³)')
plt.tight_layout()
plt.savefig(f'{dossier_fig_rom}/01_serie_temporelle_station.png')
plt.close()

# 2. ACF / PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(seg_clean, ax=axes[0], lags=30)
plot_pacf(seg_clean, ax=axes[1], lags=30)
plt.suptitle(f'ACF et PACF (Station {station_id})')
plt.tight_layout()
plt.savefig(f'{dossier_fig_rom}/02_acf_pacf_station.png')
plt.close()

print(f" Graphiques Rommaisa sauvegardés dans {dossier_fig_rom}")
print("\n FIN DU FICHIER PHASE 1 COMPLET (avec graphiques).")