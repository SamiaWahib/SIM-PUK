import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _():
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import csv
    data = []
    with open('./data/' + 'frederiksberg' + '.csv', mode='r') as file:
        content = csv.reader(file)
        for line in content:
            lst = []
            lst.append(line[5])
            if not line[10]:
                continue
            for _i in range(10, 30 * 2):
                if _i % 2 == 0:
                    lst.append(line[_i])
                else:  #10-29
                    continue
    # Removed verbose printing to avoid noisy output
            data.append(lst)
    return PCA, StandardScaler, data, np, plt


@app.cell
def _(PCA, StandardScaler, data, np, plt):
    # Filter out any rows where the party value is literally 'party' (case-insensitive)
    filtered = [row for row in data if not (isinstance(row[0], str) and row[0].strip().lower() == 'party')]
    parties = [row[0] for row in filtered]
    # Prepare parties and features from filtered data (each row: party, features...)
    features = []
    # Build feature matrix, converting to float and replacing missing with NaN
    for row in filtered:
        feat = []
        for v in row[1:]:
            try:
                feat.append(float(v))
            except Exception:
                feat.append(np.nan)
        features.append(feat)
    features = np.array(features, dtype=float)
    features = np.nan_to_num(features, nan=0.0)
    # Replace NaNs with 0.0 (or choose a different imputation strategy)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    # Standardize the features
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    party_color_map = {'A': '#AF0D0D', 'B': '#7A1898', 'C': '#729B0D', 'D': '#00505B', 'E': '#564F4E', 'F': '#F74B95', 'I': 'cyan', 'J': '#C7C2C1', 'K': '#53619B', 'M': 'purple', 'O': '#FCD03B', 'V': '#01438E', 'T': '#252322', 'Å': '#00FF00', 'Æ': '#668dd1', 'Ø': '#F7660D'}
    # Apply PCA to 2 components
    unique_parties = sorted(set(parties))
    cmap = plt.get_cmap('tab20')
    colors = {}
    # ---- Color mapping: hardcode party colors here ----
    # Edit `party_color_map` to assign exact colors for parties.
    for _i, p in enumerate(unique_parties):
        if p in party_color_map:
            colors[p] = party_color_map[p]
        else:
            colors[p] = cmap(_i % cmap.N)
    point_colors = [colors[p] for p in parties]

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=point_colors, s=30, zorder=2)

    party_means = {}
    coords = np.array(pca_result)
    party_array = np.array(parties)
    for p in unique_parties:
        idx = np.where(party_array == p)[0]
        if idx.size == 0:
            continue
        pts = coords[idx]
        mean_pt = pts.mean(axis=0)
        party_means[p] = mean_pt
        for pt in pts:
    # Map unique parties to colors, preferring the hardcoded map and
    # falling back to a colormap for unmapped parties.
            plt.plot([mean_pt[0], pt[0]], [mean_pt[1], pt[1]], color=colors[p], linewidth=0.6, alpha=0.7, zorder=1)
        plt.scatter(mean_pt[0], mean_pt[1], color=colors[p], edgecolor='k', s=140, marker='X', zorder=3)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title('PCA Analysis colored by party with party means')
    for p in unique_parties:
        plt.scatter([], [], color=colors[p], label=p)
    plt.legend(title='Party', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # Visualize candidates
    # For each party: compute mean point, draw thin lines to each candidate, and plot mean
    # Build legend entries (one per party)
    plt.show()  # draw lines from mean to each candidate  # plot mean point on top
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
