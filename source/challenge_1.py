import json
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import ruptures as rpt
import seaborn as sns


def main():
    df = load_and_process_data()

    pair_plot(df)

    make_upshot_figure(df)
    make_modified_upshot_figure(df)

    edge_list = load_edge_list()
    make_neighborhood_rank_divergence_plot(df, edge_list)


def load_and_process_data():
    # Load The Upshot data set
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

    # Load the population data set from Jane
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

    # Load the migration data set we found
    df3 = pd.read_csv('../data/net_migration.csv', dtype={'FIPS': str})

    # Merge everything together
    df = pd.merge(
        df,
        df2,
        on='FIPS',
        how='left',
        left_index=False,
        right_index=False,
    )
    df = pd.merge(
        df,
        df3,
        on='FIPS',
        how='left',
        left_index=False,
        right_index=False,
    )
    print(df.columns)
    print(df.head())
    print(len(df.index))
    return df


def load_edge_list():
    return pd.read_csv('../data/county_edge_list.csv')


def pair_plot(df):
    sns.pairplot(
        df.drop(['FIPS', 'County'], axis=1),
        plot_kws={
            'facecolor': 'None',
            'edgecolor': sns.xkcd_rgb['denim blue'],
        },
        diag_kws={'color': sns.xkcd_rgb['denim blue']},
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
            (0 / 7, '#367B7F'), (1 / 7, '#367B7F'),
            (1 / 7, '#76A5A8'), (2 / 7, '#76A5A8'),
            (2 / 7, '#A6C5C6'), (3 / 7, '#A6C5C6'),
            (3 / 7, '#EBE3D7'), (4 / 7, '#EBE3D7'),
            (4 / 7, '#F9C79E'), (5 / 7, '#F9C79E'),
            (5 / 7, '#F5A361'), (6 / 7, '#F5A361'),
            (6 / 7, '#F28124'), (7 / 7, '#F28124'),
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
        marker_line_color='white'
    )
    fig.update_layout(
        margin={'r': 0, 't': 30, 'l': 0, 'b': 0},
        coloraxis_colorbar=dict(
            tickvals=[0.4, 5.6],
            ticktext=['Better', 'Worse'],
        )
    )
    plotly.offline.plot(fig, filename='../output/upshot_figure.html', auto_open=False)
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
        marker_line_color='white'
    )
    fig.update_layout(
        margin={'r': 0, 't': 30, 'l': 0, 'b': 0},
        coloraxis_colorbar=dict(
            tickvals=[-1250, 2750],
            ticktext=['Better', 'Worse'],
        )
    )
    plotly.offline.plot(fig, filename='../output/upshot_figure_restyled.html', auto_open=False)
    fig.show()


def make_neighborhood_rank_divergence_plot(rank_df, adj_df):
    rank_df.sort_values('rank', inplace=True, ascending=True)

    divergences = np.zeros(len(rank_df.index))
    for i, (county, rank) in enumerate(zip(rank_df['County'], rank_df['rank'])):
        neighbors = adj_df.loc[adj_df.source == county, 'destination']

        if len(neighbors) == 0:
            neighbors = adj_df.loc[adj_df.destination == county, 'source']

        rank_ind = rank_df.County.isin(neighbors).values
        neighbor_ranks = rank_df.loc[rank_ind, 'rank']
        divergence = np.abs(rank - neighbor_ranks).mean()
        divergences[i] = divergence

        if np.isnan(divergence):
            print(county)
            print(neighbors)
            print(neighbor_ranks)

    rank_df['rank_div'] = divergences

    # Change point detection
    signal = rank_df['rank_div'].rolling(100).mean().dropna().values
    # model = {'l1', 'l2', 'rbf', 'linear', 'normal', 'ar'}
    pelt_bkps = rpt.Pelt(model='rbf').fit(signal).predict(pen=100)
    window_bkps = rpt.Window(width=1000, model='l2').fit(signal).predict(n_bkps=1)
    bin_bkps = rpt.Binseg(model='l2').fit(signal).predict(n_bkps=1)
    ensemble_bkp = np.mean([*pelt_bkps[:-1], *window_bkps[:-1], *bin_bkps[:-1]])

    print(
        'Identified Breakpoints:'
        f'\n\tPelt Breakpoints:    {pelt_bkps[:-1]}'
        f'\n\tWindow Breakpoints:  {window_bkps[:-1]}'
        f'\n\tBinary Breakpoints:  {bin_bkps[:-1]}'
        f'\n\tEnsemble Breakpoint: {ensemble_bkp}')

    plt.scatter(
        rank_df['rank'].values,
        rank_df['rank_div'].values,
        facecolor='None',
        edgecolor=sns.xkcd_rgb['denim blue'],
        linewidth=2,
        label='Data',
    )
    plt.plot(
        rank_df['rank'].values,
        rank_df['rank_div'].rolling(100).mean(),
        color='darkorange',
        label='Rolling Mean',
    )

    y_min, y_max = divergences.min(), divergences.max()
    y_range = y_max - y_min
    plt.plot(
        [ensemble_bkp, ensemble_bkp],
        [y_min - 0.1 * y_range, divergences.max() + 0.1 * y_range],
        'k--',
        label='Estimated Breakpoint'
    )
    plt.legend()
    plt.title('Mean Neighborhood Rank Divergence')
    plt.xlabel('Quality of Life Rank')
    plt.ylabel('Rank Divergence')
    plt.tight_layout()
    ymin, ymax = plt.gca().get_ylim()
    figsize = plt.gcf().get_size_inches()
    plt.savefig('../output/neighborhood_rank_divergence.png')

    # Visualize change points
    bkps = []
    rpt.display(
        signal,
        bkps,
        pelt_bkps,
        figsize=figsize,
    )
    plt.ylim(ymin, ymax)
    plt.gca().get_lines()[0].set_color('darkorange')
    plt.title('Pelt Change Point Detection')
    plt.xlabel('Quality of Life Rank')
    plt.ylabel('Local Rank Divergence')
    plt.tight_layout()
    plt.savefig('../output/rank_div_change_point_pelt.png')

    rpt.show.display(
        signal,
        bkps,
        window_bkps,
        figsize=figsize,
    )
    plt.ylim(ymin, ymax)
    plt.gca().get_lines()[0].set_color('darkorange')
    plt.title('Window Change Point Detection')
    plt.xlabel('Quality of Life Rank')
    plt.ylabel('Local Rank Divergence')
    plt.tight_layout()
    plt.savefig('../output/rank_div_change_point_window.png')

    rpt.show.display(
        signal,
        bkps,
        bin_bkps,
        figsize=figsize,

    )
    plt.ylim(ymin, ymax)
    plt.gca().get_lines()[0].set_color('darkorange')
    plt.title('Binary Change Point Detection')
    plt.xlabel('Quality of Life Rank')
    plt.ylabel('Local Rank Divergence')
    plt.tight_layout()
    plt.savefig('../output/rank_div_change_point_binary.png')


if __name__ == '__main__':
    main()
