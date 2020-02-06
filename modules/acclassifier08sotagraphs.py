import json

import pandas as pd
import seaborn as sns
from cbh import config
import matplotlib.pyplot as plt


target = config.TARGET
json_file = f"{config.TABLES_DIR}/{target}_scores.json"

figs_dir = str(config.FIGURES_DIR)


df = pd.read_json(json_file)
df = df.T
print(df)

for colname in list(df):
    ax = sns.barplot(y=df.index, x=colname, data=df)
    ax.set_yticklabels(labels=df.index)
    plt.setp(ax.patches, linewidth=0)
    sns.despine(left=True, bottom=True, right=True)
    ax.tick_params(axis='y', which='both', length=0)

    rects = ax.patches

    # For each bar: Place a label
    for rect in rects:
        # Get X and Y placement of label from rect.
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2

        # Number of points between bar and label. Change to your liking.
        space = 5
        # Vertical alignment for positive values
        ha = 'left'

        # If value of bar is negative: Place label left of bar
        if x_value < 0:
            # Invert space to place label to the left
            space *= -1
            # Horizontally align label at right
            ha = 'right'

        # Use X value as label and format number with one decimal place
        label = f"{x_value:.2f}"

        # Create annotation
        plt.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(space, 0),          # Horizontally shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            va='center',                # Vertically center label
            ha=ha)                      # Horizontally align label differently for
                                        # positive and negative values.

    
    
    plt.tight_layout()
    plt.savefig(f"{figs_dir}/{target}_sota_bar_{colname}.pdf")
    plt.close()


# import plotly.express as px
# import plotly.io as pio
# pio.orca.config.executable = '/home/hiltonc/miniconda3/envs/ML01/bin/orca'
# fig = px.bar(df)
# pio.write_html(fig, file=f'{figs_dir}/sota_bar.html')
# fig.write_image(f'{figs_dir}/sota_bar.pdf')