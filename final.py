import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md('# pick a munipality:')

    muni = mo.ui.dropdown(["koebenhavn", "frederiksberg", 
                      "hvidovre", "taarnby", 
                      "dragoer", "roedovre", 
                      "lyngby-taarbaek", 
                      "broendby", "gentofte", 
                      "gladsaxe", "herlev",
                      "ballerup", "glostrup",
                      "albertslund", "ishoej", "vallensbaek", 
                      "furesoe", "rudersdal", "greve"], value="koebenhavn")

    muni
    return mo, muni


@app.cell
def _(mo):
    mo.md("""
    # visualizing each candidate from the
    """)
    return


@app.cell
def _(muni):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    all_candidates = pd.read_csv(f'./data/{muni.value}.csv')

    parties = pd.Series(all_candidates['party'].value_counts())
    letters = parties.index


    let_dict = {letter: [0,0] for letter in letters}

    # returns a dictionary containing each party's letter, number of votes, and how many were elected
    for i, cand in all_candidates.iterrows():
        num = cand['votes']
        ele = 1 if cand['elected'] else 0
        let_dict[cand['party']][0] += num
        let_dict[cand['party']][1] += ele


    party_color_map = {'A': '#AF0D0D', 'B': '#7A1898', 'C': '#729B0D', 'D': '#00505B', 'E': '#564F4E', 'F': '#F74B95', 'I': 'cyan', 'J': '#C7C2C1', 'K': '#53619B', 'M': 'purple', 'O': '#FCD03B', 'V': '#01438E', 'T': '#252322', 'Å': '#00FF00', 'Æ': '#668dd1', 'Ø': '#F7660D'}


    def get_color(letter):
        if letter in party_color_map.keys():
            return party_color_map[letter]
        return '#330019' # dark maroon color, for those candidates with lesser-known parties

    lst = []
    color_elect = []
    color_not_elec = ['black', '#808080'] * len(letters) * 2

    # for each party, get the their corresponding color
    # return a numpy.array with elected, candidates that weren't elected, and color for each party
    for letter in letters:
        all_candidates = parties[letter]
        elected = let_dict[letter][1]
        color_elect.append(get_color(letter))
        lst.append([elected, all_candidates-elected])
    lst = np.array(lst)

    fig, ax = plt.subplots()
    size = 0.3

    inner_labels = ["true", "false"] * len(letters) # if true then elected, if false not elected

    ax.pie(lst.sum(axis=1), radius=1, colors=color_elect,
           wedgeprops=dict(width=size, edgecolor='w'), labels=letters)

    ax.pie(lst.flatten(), radius=1-size, colors=color_not_elec,
           wedgeprops=dict(width=size, edgecolor='w'), labels=inner_labels, labeldistance=0.7)

    ax.set(aspect="equal", title='Pie plot with `ax.pie`')
    plt.legend(letters, loc=(-0.3,0))
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
