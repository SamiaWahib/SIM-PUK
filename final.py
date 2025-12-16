import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    all_candidates = pd.read_csv('./data/frederiksberg.csv')
    # all_candidates['A'].sum()
    parties = pd.Series(all_candidates['party'].value_counts())
    letters = parties.index

    let_dict = {letter: [0,0] for letter in letters}


    for i, cand in all_candidates.iterrows():
        num = cand['votes']
        ele = 1 if cand['elected'] else 0
        let_dict[cand['party']][0] += num
        let_dict[cand['party']][1] += ele


    fig, ax = plt.subplots()



    size = 0.3
    # vals = np.array([[60., 32.], [37., 40.], [29., 10.]])
    party_color_map = {'A': '#AF0D0D', 'B': '#7A1898', 'C': '#729B0D', 'D': '#00505B', 'E': '#564F4E', 'F': '#F74B95', 'I': 'cyan', 'J': '#C7C2C1', 'K': '#53619B', 'M': 'purple', 'O': '#FCD03B', 'V': '#01438E', 'T': '#252322', 'Å': '#00FF00', 'Æ': '#668dd1', 'Ø': '#F7660D'}

    def get_color(letter):
        if letter in party_color_map.keys():
            return party_color_map[letter]
        return 'grey'


    lst = []
    color = []

    for letter in letters:
        all_candidates = parties[letter]
        elected = let_dict[letter][1]
        # color.append(get_color(letter))
        lst.append([elected, all_candidates-elected])
    lst = np.array(lst)

    # tab20c = plt.color_sequences["tab20c"]
    # outer_colors = [ for i in range(0,15)]
    # inner_colors = [tab20c[i] for i in [1, 2, 5, 6, 9, 10]]

    inner_labels = ["true", "false"] * len(letters)

    ax.pie(lst.sum(axis=1), radius=1,
           wedgeprops=dict(width=size, edgecolor='w'), labels=letters)

    ax.pie(lst.flatten(), radius=1-size,
           wedgeprops=dict(width=size, edgecolor='w'), labels=inner_labels, labeldistance=0.7)

    ax.set(aspect="equal", title='Pie plot with `ax.pie`')
    plt.legend(loc=(-0.3,0))
    plt.show()

    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
