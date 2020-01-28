import json
from urllib.request import urlopen

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from pandas_summary import DataFrameSummary


def main():
    df = pd.read_csv('CountyData.tsv', sep='\t')
    dfs = DataFrameSummary(df)

    for col in df.columns:
        print(df[col].describe())

    sns.pairplot(df)
    plt.tight_layout()
    plt.savefig('county_data_pair.png')

    print(dfs.columns_types)
    print(dfs.columns_stats)

    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    print(counties['features'][0])

    df_2 = pd.read_csv(
        "https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
        dtype={"fips": str}
    )
    fig = px.choropleth_mapbox(
        df,
        geojson=counties,
        locations='id',
        color='rank',
        color_continuous_scale="Viridis",
        # range_color=(0, 12),
        mapbox_style="carto-positron",
        zoom=3, center={"lat": 37.0902, "lon": -95.7129},
        opacity=0.5,
        labels={'unemp': 'unemployment rate'}
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


if __name__ == '__main__':
    main()
