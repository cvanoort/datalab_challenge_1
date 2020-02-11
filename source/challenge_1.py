import json
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import ruptures as rpt
import seaborn as sns
from sklearn.decomposition import PCA


def main():
    df = load_and_process_data()

    pair_plot(df)

    make_upshot_figure(df)
    make_modified_upshot_figure(df)

    edge_list = load_edge_list()
    make_neighborhood_rank_divergence_plot(df, edge_list)

    reconstruct_rank(df)
    alt_rank(df)
    pca_analysis(df)

    cols = ['education', 'income', 'unemployment', 'disability', 'life', 'obesity', 'migration rate']
    ascending = [False, False, True, True, False, True, False]
    alt_recon_df = reconstruct_rank(df, cols=cols, ascending=ascending, path='../output/migration_rank.png')
    make_rank_div_map(alt_recon_df)


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
    # There are a few counties with missing data
    # We may want to see if we can fill this in?
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
    plt.savefig('../output/county_data_pair.png', dpi=600)
    plt.close()


def make_upshot_figure(df, renderer=None, save=True):
    """
    Replicates the figure from:
        https://www.nytimes.com/2014/06/26/upshot/where-are-the-hardest-places-to-live-in-the-us.html
    """
    if renderer is None:
        renderer = plotly.io.renderers.default

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

    if save:
        plotly.offline.plot(fig, filename='../output/upshot_figure.html', auto_open=False)

    fig.show(renderer=renderer)


def make_modified_upshot_figure(df, renderer=None, save=True):
    """
    Produces a figure similar to:
        https://www.nytimes.com/2014/06/26/upshot/where-are-the-hardest-places-to-live-in-the-us.html

    Design decisions:
        - Color based on integer rank, not quantiles.
        - Sequential rather than diverging color map.
        -
    """
    if renderer is None:
        renderer = plotly.io.renderers.default

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
            'rank': 'Quality of Life',
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

    if save:
        plotly.offline.plot(fig, filename='../output/upshot_figure_restyled.html', auto_open=False)

    fig.show(renderer=renderer)


def make_rank_div_map(df, renderer=None, save=True):
    """
    Makes a similar plot to make_upshot_figure and make_modified_upshot_figure,
    but the coloring is based on a rank divergence rather than the QoL indicator.
    """
    if renderer is None:
        renderer = plotly.io.renderers.default

    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    fig = px.choropleth(
        df,
        geojson=counties,
        locations='FIPS',
        color='rank_error',
        hover_name='County',
        hover_data=['rank_error', 'rank', 'rank_remake', 'income', 'education', 'unemployment', 'disability', 'life',
                    'obesity'],
        color_continuous_scale='RdBu',
        range_color=[-500, 500],
        labels={
            'rank_error': 'Rank Divergence',
            'rank': 'Quality of Life Rank',
            'rank_remake': 'Alternate QoL Rank',
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
    fig.update_layout(
        margin={'r': 0, 't': 30, 'l': 0, 'b': 0},
    )

    if save:
        plotly.offline.plot(fig, filename='../output/rank_divergence_map.html', auto_open=False)

    fig.show(renderer=renderer)


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
        f'\n\tEnsemble Breakpoint: {ensemble_bkp}'
    )

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
        [y_min - 0.1 * y_range, y_max + 0.1 * y_range],
        'k--',
        label='Estimated Breakpoint'
    )
    plt.legend()
    plt.title('Mean Neighborhood Rank Divergence')
    plt.xlabel('Quality of Life Rank (Lower is better)')
    plt.ylabel('Rank Divergence')
    plt.tight_layout()
    ymin, ymax = plt.gca().get_ylim()
    figsize = plt.gcf().get_size_inches()
    plt.savefig('../output/neighborhood_rank_divergence.png', dpi=600)

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
    plt.savefig('../output/rank_div_change_point_pelt.png', dpi=600)

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
    plt.savefig('../output/rank_div_change_point_window.png', dpi=600)

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
    plt.savefig('../output/rank_div_change_point_binary.png', dpi=600)
    plt.close()


def reconstruct_rank(df, cols=None, ascending=None, path='../output/rank_reconstruction_error.png'):
    if cols is None:
        cols = ['education', 'income', 'unemployment', 'disability', 'life', 'obesity']
    if ascending is None:
        ascending = [False, False, True, True, False, True]

    assert len(cols) == len(ascending)

    ranks = []
    for col, ascend in zip(cols, ascending):
        ranks.append(df[col].rank(ascending=ascend, method='first'))

    df['rank_remake'] = pd.Series(np.mean(ranks, axis=0)).rank(ascending=True, method='first')
    df['rank_error'] = df['rank'] - df['rank_remake']

    sns.distplot(df.rank_error, rug=True)
    plt.title('Rank Reconstruction Error')
    plt.savefig(path, dpi=600)
    plt.close()

    print("\n\nRank Reconstruction Error Details:")
    print(f"Total Rank Deviation: {df['rank_error'].abs().sum()}")
    print('\nRank Deviation Summary')
    print(df['rank_error'].describe())
    print('\nNon-Zero Rank Deviation Summary')
    print(df.loc[df.rank_error != 0, 'rank_error'].describe())

    return df


def alt_rank(df, cols=None, ascending=None, path='../output/alt_rank_deviations.png'):
    if cols is None:
        cols = ['education', 'income', 'unemployment', 'disability', 'life', 'obesity']
    if ascending is None:
        ascending = [False, False, True, True, False, True]

    assert len(cols) == len(ascending)

    alt_df = df.loc[:, cols]
    alt_df -= df.mean(axis=0)
    alt_df /= df.std(axis=0)
    alt_df *= np.array([1 if not x else -1 for x in ascending])
    alt_df['alt_rank'] = alt_df.mean(axis=1).rank(ascending=False, method='dense')
    alt_df['rank_change'] = df['rank'] - alt_df['alt_rank']

    sns.distplot(alt_df.rank_change, rug=True)
    plt.title('Alternate Rank Deviation')
    plt.savefig(path, dpi=600)
    plt.close()

    print("\n\nAlternate Rank Deviation Details:")
    print(f"Total Rank Deviation: {alt_df['rank_change'].abs().sum()}")
    print('\nRank Deviation Summary')
    print(alt_df['rank_change'].describe())
    print('\nNon-Zero Rank Deviation Summary')
    print(alt_df.loc[alt_df.rank_change != 0, 'rank_change'].describe())

    return alt_df


def pca_analysis(df):
    cols = ['education', 'income', 'unemployment', 'disability', 'life', 'obesity']
    df = df.loc[:, cols]
    df -= df.mean(axis=0)
    df /= df.std(axis=0)

    pca = PCA()
    pca.fit(df)

    print("\n\nPCA:")
    print("Explained Variance Ratio by Component")
    print(pca.explained_variance_ratio_)
    print("\nColumns Investigated")
    print(cols)
    print("\nComponent Weights")
    print(pca.components_)


if __name__ == '__main__':
    main()
