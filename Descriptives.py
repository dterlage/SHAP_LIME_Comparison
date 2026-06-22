#%%
#imports
import pandas as pd
import numpy as np

import DataLoader as dl


# What does the fraud dataset look like?
print(dl.data_fraud.head())
print(dl.data_fraud.shape) #Before data preperation: (1309, 21), now: (891, 18)
print(dl.data_fraud.isnull().sum()) # missing data is up to 15% --> using Listwise Deletion (Removing all rows with missing data)

#%%
df_corrs_all = pd.DataFrame(index=dl.data_fraud.select_dtypes(include=np.number).columns)

dataset_configs = [
        (dl.data_fraud, 'is_fraud', 0.0, 1),
        (dl.data_fraud, 'is_fraud', 0.1, 1),
        (dl.data_fraud, 'is_fraud', 0.2, 1),
        (dl.data_fraud, 'is_fraud', 0.3, 1),
        (dl.data_fraud, 'is_fraud', 0.4, 1),
        (dl.data_fraud, 'is_fraud', 0.5, 1)
    ]

for df, target, noise, random_state in dataset_configs:
    df = dl.corrupt_labels(df, target, noise, random_state)

    corrs = (
        df.corr(numeric_only=True)[target]
          .sort_values(ascending=False)
    )
    df_corrs_all[f'{int(noise * 100)}\%'] = corrs     # the \ is needed so that the % doesn't cause problems in LateX OverLeaf

print(df_corrs_all)



# %%
# Latex table: https://www.tilburgsciencehub.com/topics/visualization/reporting-tables/reportingtables/pandas-latex-tables/
# \usepackage{booktabs}
latex_table_split = df_corrs_all.to_latex(
    index=True,  # To not include the DataFrame index as a column in the table
    index_names = 'Features',
    caption="Example correlations between features and the target feature for each noise level",  # The caption to appear above the table in the LaTeX document
    label="tab:correlations",  # A label used for referencing the table within the LaTeX document
    position="htbp",  # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
    column_format="lccccccc",  # The format of the columns: left-aligend first column and center-aligned remaining columns as per APA guidelines
    escape=False,  # Disable escaping LaTeX special characters in the DataFrame
    float_format="{:0.4f}".format  # Formats floats to two decimal places
)

print(latex_table_split)