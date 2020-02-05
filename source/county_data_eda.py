import json
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns


def main():
    df = load_and_process_data()

    pair_plot(df)

    make_upshot_figure(df)
    make_modified_upshot_figure(df)

    edge_list = load_edge_list()
    make_neighborhood_rank_divergence_plot(df, edge_list)


def load_and_process_data():
    df = pd.read_csv(
        '../data/county_data.tsv',
        dtype={'id': str},
        sep='\t',
        na_values=['No Data', '#N/A']
    )
    df.dropna(inplace=True)
    df.sort_values('rank', inplace=True)
    df['id'] = df['id'].str.zfill(5)
    df.rename({'id': 'FIPS'}, axis=1, inplace=True)

    df2 = pd.read_csv('../data/centroids.csv', na_values=['-'])

    # Clean up and type cast a few columns
    for col in [
        'Population (2010)',
        'Land Area (km)',
        'Land Area(mi)',
        'Water Area (km)',
        'Water Area (mi)',
        'Total Area (km)',
        'Total Area (mi)'
    ]:
        df2[col] = df2[col].str.replace(',', '').astype(float)

    df2['FIPS'] = df2['FIPS'].astype(str).str.zfill(5)
    df2.drop('County', axis=1, inplace=True)

    df3 = pd.merge(df, df2, on='FIPS', how='left', left_index=False, right_index=False)
    return df3


def load_edge_list():
    return pd.read_csv('../data/county_edge_list.csv')


def pair_plot(df):
    sns.pairplot(
        df.drop(['FIPS', 'County'], axis=1),
        plot_kws={
            'facecolor': 'None',
            'edgecolor': sns.xkcd_rgb["denim blue"],
        },
        diag_kws={'color': sns.xkcd_rgb["denim blue"]},
    )
    plt.tight_layout()
    plt.savefig('../output/county_data_pair.png')
    plt.close()


def make_upshot_figure(df):
    """
    Replicates the figure from:
        https://www.nytimes.com/2014/06/26/upshot/where-are-the-hardest-places-to-live-in-the-us.html
    """
    df['Quality of Life'] = pd.qcut(df['rank'], 7, labels=False)

    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    fig = px.choropleth(
        df,
        geojson=counties,
        locations='FIPS',
        color='Quality of Life',
        hover_name='County',
        hover_data=['rank', 'income', 'education', 'unemployment', 'disability', 'life', 'obesity'],
        color_continuous_scale=[
            (0 / 7, "#367B7F"), (1 / 7, "#367B7F"),
            (1 / 7, "#76A5A8"), (2 / 7, "#76A5A8"),
            (2 / 7, "#A6C5C6"), (3 / 7, "#A6C5C6"),
            (3 / 7, "#EBE3D7"), (4 / 7, "#EBE3D7"),
            (4 / 7, "#F9C79E"), (5 / 7, "#F9C79E"),
            (5 / 7, "#F5A361"), (6 / 7, "#F5A361"),
            (6 / 7, "#F28124"), (7 / 7, "#F28124"),
        ],
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
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            tickvals=[0.4, 5.6],
            ticktext=["Worse", "Better"],
        )
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
        range_color=[-1500, df['rank'].max()],
        labels={
            'rank': 'Quality of Life (Rank)',
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
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            tickvals=[-1250, 2750],
            ticktext=["Worse", "Better"],
        )
    )
    fig.show()


def make_neighborhood_rank_divergence_plot(rank_df, adj_df):
    rank_df.sort_values('rank', inplace=True, ascending=True)

    divergences = np.zeros(len(rank_df.index))
    for i, (county, rank) in enumerate(zip(rank_df['County'], rank_df['rank'])):
        neighbors = adj_df.loc[adj_df.source == county, 'destination']
        divergences[i] = (rank - rank_df.loc[rank_df.County.isin(neighbors).values, 'rank']).mean()

    plt.scatter(
        rank_df['rank'].values, divergences,
        facecolor='None',
        edgecolor=sns.xkcd_rgb["denim blue"],
        linewidth=2,
    )
    plt.title('Mean Neighborhood Rank Divergence')
    plt.xlabel('Quality of Life Rank')
    plt.ylabel('Rank Divergence')
    plt.tight_layout()
    plt.savefig('../output/neighborhood_rank_divergence.png')


if __name__ == '__main__':
    main()
