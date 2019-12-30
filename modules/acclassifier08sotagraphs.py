import json

import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.orca.config.executable = '/home/hiltonc/miniconda3/envs/ML01/orca'

from cbh import config
figs_dir = str(config.FIGURES_DIR)
df = pd.read_json(config.SCORES_JSON_SOTA)
df = df.T
print(df)


fig = px.bar(df)

pio.write_html(fig, file=f'{figs_dir}/sota_bar.html')
fig.write_image(f'{figs_dir}/sota_bar.pdf')