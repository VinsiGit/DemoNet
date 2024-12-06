import pandas as pd
import numpy as np

file_path = "source-data/material_inventory_dataset_1.0.1.ods"
sheet_name = "data_building_part"

data = pd.read_excel(file_path, sheet_name=sheet_name, engine="odf", skiprows=2)

columns_to_keep = [
    "b_completionyear", "mi_s_tfa_aconcrete_l",
    "mi_s_tfa_aconcrete_s", "mi_s_tfa_bitumen", "mi_s_tfa_brick", "mi_s_tfa_concrete",
    "mi_s_tfa_expclay_l", "mi_s_tfa_expclay_s", "mi_s_tfa_minwool_h", "mi_s_tfa_minwool_l",
    "mi_s_tfa_mortar", "mi_s_tfa_plasterboard", "mi_s_tfa_polystyrene", "mi_s_tfa_steel",
    "mi_s_tfa_wood_s", "mi_s_tfa_woodchip", "mi_s_tfa_woodfiber_s", "mi_s_tfa_woodprod",
    "mi_c_tfa_aluminium", "mi_c_tfa_glass", "mi_c_tfa_mdf", "mi_c_tfa_minwool_h",
    "mi_c_tfa_steel", "mi_c_tfa_wood_s", "mi_c_tfa_woodprod"
]

data_filtered = data[columns_to_keep]

data_filtered = data_filtered[data_filtered['b_completionyear'] != 'N/A**']

mean_per_year = data_filtered.groupby('b_completionyear', as_index=False).mean()

mean_per_year = mean_per_year.apply(pd.to_numeric, errors='ignore')

mean_per_year['b_completionyear'] = mean_per_year['b_completionyear'].astype(int)

mean_per_year.columns = (
    mean_per_year.columns
    .str.replace('^mi_s_', '', regex=True)
    .str.replace('^mi_c_', '', regex=True)
)

for col in mean_per_year.columns[mean_per_year.columns.duplicated()].unique():
    duplicate_cols = mean_per_year.columns[mean_per_year.columns == col]
    mean_per_year[col] = mean_per_year[duplicate_cols].sum(axis=1)
    mean_per_year = mean_per_year.drop(columns=duplicate_cols[1:])

mean_per_year.columns = mean_per_year.columns.str.replace('^tfa_', '', regex=True)

mean_per_year = mean_per_year.sort_values('b_completionyear')

mean_per_year = mean_per_year.drop(columns='aconcrete_l')

mean_per_year.set_index('b_completionyear', inplace=True)

year_range = np.arange(1963, 2019)
mean_per_year = mean_per_year.reindex(year_range)

interpolated_data = mean_per_year.interpolate(method='cubic')

interpolated_data = interpolated_data.clip(lower=0)

interpolated_data.fillna(method='ffill', inplace=True)
interpolated_data.fillna(method='bfill', inplace=True)

interpolated_data = interpolated_data.apply(pd.to_numeric, errors='coerce')

interpolated_data = interpolated_data.applymap(lambda x: 0 if abs(x) < 1e-7 else round(x, 6))

# Add the current index as a new column named 'year'
interpolated_data['year'] = interpolated_data.index

# Reset the index and assign a new one
interpolated_data = interpolated_data.reset_index(drop=True)

# Print the DataFrame to check changes
print(interpolated_data)
print(interpolated_data.columns)