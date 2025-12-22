import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import math
    mo.md('# pick a munipality:')

    municipalities = ["koebenhavn", "frederiksberg", 
                      "hvidovre", "taarnby", 
                      "dragoer", "roedovre", 
                      "lyngby-taarbaek", 
                      "broendby", "gentofte", 
                      "gladsaxe", "herlev",
                      "ballerup", "glostrup",
                      "albertslund", "ishoej", "vallensbaek", 
                      "furesoe", "rudersdal", "greve"]

    mun_dan_to_eng = {"København":"koebenhavn", 
                      "Frederiksberg" : "frederiksberg", 
                      "Hvidovre":"hvidovre", 
                      "Tårnby":"taarnby", 
                      "Dragør":"dragoer", 
                      "Rødovre":"roedovre", 
                      "Lyngby-taarbæk":"lyngby-taarbaek", 
                      "Brøndby":"broendby", 
                      "Gentofte":"gentofte", 
                      "Gladsaxe":"gladsaxe", 
                      "Herlev":"herlev",
                      "Ballerup":"ballerup", 
                      "Glostrup":"glostrup",
                      "Albertslund":"albertslund", 
                      "Ishøj":"ishoej", 
                      "Vallensbæk":"vallensbaek", 
                      "Furesø":"furesoe", 
                      "Rudersdal":"rudersdal", 
                      "Greve":"greve"}

    party_color_map = {'A': '#AF0D0D', # social demokratiet
                       'B': '#7A1898', # radikale
                       'C': '#729B0D', # konservative
                       'D': '#00505B', # nye borgerlige
                       'F': '#F74B95', # SF
                       'I': 'cyan', # liberal alliance
                       'M': 'purple', # moderaterne
                       'O': '#FCD03B', # DF
                       'V': '#01438E', # venstre
                       'Å': '#00FF00', # alternativet
                       'Æ': '#668dd1', # danmarks demokraterne
                       'Ø': '#F7660D' # enhedslisten
                      }

    def get_color(letter):
        if letter in party_color_map.keys():
            return party_color_map[letter]
        return '#330019' # dark maroon color, for those candidates with lesser-known parties
    return get_color, math, mo, mun_dan_to_eng


@app.cell
def _(mo, mun_dan_to_eng):
    muni = mo.ui.dropdown(mun_dan_to_eng.keys(), value="Herlev", label = "See data from municipality:")
    muni
    return (muni,)


@app.cell
def _(mo, muni):
    mo.md(f"""
    # Party sizes in {muni.value}
    """)
    return


@app.cell
def _(mo):
    options = ["candidates", "votes"]
    chosen_size = mo.ui.dropdown(options, value = options[0], label = "See party size based on number of:")
    chosen_size
    return (chosen_size,)


@app.cell
def _(chosen_size, get_color, math, mun_dan_to_eng, muni):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    chosen_file = pd.read_csv(f'./data/{mun_dan_to_eng[muni.value]}.csv')

    parties = pd.Series(chosen_file['party'].value_counts())
    letters = parties.index


    let_dict = {letter: [0,0] for letter in letters}

    # returns a dictionary containing each party's letter, number of votes, and how many were elected
    for i, cand in chosen_file.iterrows():
        num = cand['votes']
        ele = 1 if cand['elected'] else 0
        if not math.isnan(num): let_dict[cand['party']][0] += num
        #if cand['party'] == 'Ø': print(num)
        let_dict[cand['party']][1] += ele

    color_elect = [get_color(letter) for letter in letters]

    def nested_pie_num_cand():
        lst = []
        color_not_elec = ['black', 'grey'] * len(letters) * 2
    
        # for each party, get the their corresponding color
        # return a numpy.array with elected, candidates that weren't elected, and color for each party
        for letter in letters:
            chosen_file = parties[letter] # todo: rename
            elected = let_dict[letter][1]
            lst.append([elected, chosen_file-elected])
        lst = np.array(lst)
    
        fig, ax = plt.subplots()
        size = 0.3
    
        ax.pie(lst.sum(axis=1), radius=1, colors=color_elect,
               wedgeprops=dict(width=size, edgecolor='w'), labels=letters)
    
        ax.pie(lst.flatten(), radius=1-size, colors=color_not_elec,
               wedgeprops=dict(width=size, edgecolor='w')) 
    
        ax.set(aspect="equal", title=f'Party size by number of candidates in {muni.value}. \n {color_not_elec[1]} = not elected, {color_not_elec[0]} = elected')
        plt.legend(letters, loc=(-0.3,0))
        return ax

    def pie_num_votes():
        votes = [let_dict[letter][0] for letter in letters]
        #print(votes,letters)
        size = 0.6
    
        fig, ax = plt.subplots()

        ax.pie(votes, radius=1, colors = color_elect,
               wedgeprops=dict(width=size, edgecolor='w'), labels=letters)

        plt.legend(letters, loc=(-0.3,0))
        return ax

    if chosen_size.value == "candidates":
        ax = nested_pie_num_cand()
    else:
        ax = pie_num_votes()
    ax
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
