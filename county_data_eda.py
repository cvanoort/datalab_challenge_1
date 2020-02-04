import json
from urllib.request import urlopen

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from pandas_summary import DataFrameSummary


def main():
    df = load_and_process_data()

    # for col in df.columns:
    #     print(df[col].describe())

    # dfs = DataFrameSummary(df)
    # print(dfs.columns_types)
    # print(dfs.columns_stats)

    pair_plot(df)

    make_upshot_figure(df)
    make_modified_upshot_figure(df)


def load_and_process_data():
    df = pd.read_csv(
        'CountyData.tsv',
        dtype={'id': str},
        sep='\t',
        na_values=['No Data', '#N/A']
    )
    df.dropna(inplace=True)
    df.sort_values('rank', inplace=True)
    df['id'] = df['id'].str.zfill(5)
    df.rename({'id': 'FIPS'}, axis=1, inplace=True)
    print(df.columns)

    df2 = pd.read_csv('centroids.csv', na_values=['-'])

    # Clean up and type cast a few columns
    for col in ['Population (2010)', 'Land Area (km)', 'Land Area(mi)', 'Water Area (km)', 'Water Area (mi)', 'Total Area (km)', 'Total Area (mi)']:
        df2[col] = df2[col].str.replace(',', '').astype(float)

    df2['FIPS'] = df2['FIPS'].astype(str).str.zfill(5)
    df2.drop('County', axis=1, inplace=True)

    df3 = pd.merge(df, df2, on='FIPS', how='left', left_index=False, right_index=False)
    return df3


def pair_plot(df):
    sns.pairplot(df.drop(['FIPS', 'County'], axis=1), plot_kws={'facecolor': 'None', 'edgecolor': sns.xkcd_rgb["denim blue"]})
    plt.tight_layout()
    plt.savefig('county_data_pair.png')
    plt.close()


def make_upshot_figure(df):
    """
    Replicates the figure from:
        https://www.nytimes.com/2014/06/26/upshot/where-are-the-hardest-places-to-live-in-the-us.html
    """
    df['Quality of Life'] = pd.qcut(df['rank'], 7, labels=False).astype(str)

    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    fig = px.choropleth(
        df,
        geojson=counties,
        locations='FIPS',
        color='Quality of Life',
        hover_name='County',
        hover_data=['rank', 'income', 'education', 'unemployment', 'disability', 'life', 'obesity'],
        color_discrete_map={
            '0': "#367B7F",
            '1': "#76A5A8",
            '2': "#A6C5C6",
            '3': "#EBE3D7",
            '4': "#F9C79E",
            '5': "#F5A361",
            '6': "#F28124"
        },
        labels={
            'rank': 'Overall Rank',
            'income': 'Median Income',
            'education': 'College Education',
            'unemployment': 'Unemployment',
            'disability': 'Disability',
            'life': 'Life Expectancy',
            'obesity': 'Obesity',
        }
    )
    fig.update_geos(
        visible=False,
        scope='usa',
    )
    fig.update_traces(
        marker_line_color="white"
    )
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )
    fig.show()


def make_modified_upshot_figure(df):
    """
    Replicates the figure from:
        https://www.nytimes.com/2014/06/26/upshot/where-are-the-hardest-places-to-live-in-the-us.html
    """
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    fig = px.choropleth(
        df,
        geojson=counties,
        locations='FIPS',
        color='rank',
        hover_name='County',
        hover_data=['rank', 'income', 'education', 'unemployment', 'disability', 'life', 'obesity'],
        color_continuous_scale='Greens',
        range_color=[-1000, df['rank'].max()],
        labels={
            'rank': 'Quality of Life Rank',
            'income': 'Median Income',
            'education': 'College Education',
            'unemployment': 'Unemployment',
            'disability': 'Disability',
            'life': 'Life Expectancy',
            'obesity': 'Obesity',
        }
    )
    fig.update_geos(
        visible=False,
        scope='usa',
    )
    fig.update_traces(
        marker_line_color="white"
    )
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )
    fig.show()


if __name__ == '__main__':
    main()
