import os
import random
import pandas as pd
from scipy import stats
import numpy as np
from glob import glob
from evaluate_crop_coefficient import generate_Kc

# Folder paths
folder_path = './raw_info'
folder_path_final = './seasonal_weather_data'

# Column mapping
col_map = {
    'Air Temp. Avg. (°C)': 'dAVG_TEMP',
    'Precip. (mm)': 'dRAIN',
    'ET. Std-Grass (mm)': 'dPET'
}
columns_required = list(col_map.keys())

# Get and shuffle CSV files
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
random.shuffle(csv_files)

combined_data = []

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    try:
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')

        if not all(col in df.columns for col in columns_required):
            print(f"Skipping {file} - missing required columns")
            continue

        # Filter and clean
        df_filtered = df[columns_required]
        df_filtered = df_filtered[(df_filtered >= 0).all(axis=1)]

        # Extract year
        if 'Date (Local Standard Time)' in df.columns:
            date_series = pd.to_datetime(df['Date (Local Standard Time)'], errors='coerce')
            year = date_series.dt.year.mode()[0] if not date_series.isna().all() else file[:4]
        else:
            year = file[:4]

        # Save each column + generate Kc from temperature
        for col in columns_required:
            var = col_map[col]
            filename = f"{var}_{year}.txt"
            df_filtered[col].to_csv(os.path.join(folder_path_final, filename), index=False, header=False)

        # Compute Kc from temperature and save
        temps = df_filtered['Air Temp. Avg. (°C)'].values
        kc_vals = generate_Kc(temps)
        kc_filename = os.path.join(folder_path_final, f"Kc_{year}.txt")
        np.savetxt(kc_filename, kc_vals, fmt='%.4f')

        combined_data.append(df_filtered)

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Fit distributions
df_combined = pd.concat(combined_data, ignore_index=True)
distributions = {}
for col in columns_required:
    mu, sigma = stats.norm.fit(df_combined[col])
    distributions[col_map[col]] = {'mean': mu, 'std': sigma}

print("\nFitted Normal Distributions:")
for label, params in distributions.items():
    print(f"{label}: N(mean={params['mean']:.2f}, std={params['std']:.2f})")

# === Combine files across years for each variable including Kc ===
for var in ['dAVG_TEMP', 'dRAIN', 'dPET', 'Kc']:
    files = glob(os.path.join(folder_path_final, f"{var}_*.txt"))
    file_year_pairs = [(f, os.path.basename(f).split('_')[-1].replace('.txt', '')) for f in files]
    random.shuffle(file_year_pairs)

    combined_lines = []
    for file_path, year in file_year_pairs:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            combined_lines.extend(lines)

    combined_filename = os.path.join(folder_path_final, f"{var}_combined.txt")
    with open(combined_filename, 'w') as f_out:
        f_out.writelines(combined_lines)

    print(f"Combined file written for {var} → {combined_filename}")
