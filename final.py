import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    todo
    - [ ] check om rækkefølge af spørgsmål/temaer er samme for alle (albertslund brugt)
    - [ ] gør grafik om priotiterede temaer interaktiv
    - [ ] tilføj til grafik om temaer
    - [ ] lad selv brugeren svare på spørgsmål og blive placeret med pca indenfor valgt tema
    - [ ] tekst om pca - forklar
    - [ ] problem med pca: beredskab og sikkerhed viser intet
        - sidste spørgsmåls svar ikke registreret (ser på albertslund)
    - [ ] omskriv pca? chat
    - [x] giv mindre partier forskellige farver
    - [x] partier i alfabetisk rækkefølge
    """)
    return


@app.cell
def _():
    import marimo as mo
    import math
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import pandas as pd
    import csv
    import random as rand

    mun_dan_to_eng = {
                      "Albertslund":"albertslund", 
                      "Ballerup":"ballerup", 
                      "Brøndby":"broendby", 
                      "Dragør":"dragoer", 
                      "Frederiksberg" : "frederiksberg", 
                      "Furesø":"furesoe", 
                      "Gentofte":"gentofte", 
                      "Gladsaxe":"gladsaxe", 
                      "Glostrup":"glostrup",
                      "Greve":"greve",
                      "Herlev":"herlev",
                      "Hvidovre":"hvidovre", 
                      "Ishøj":"ishoej", 
                      "København":"koebenhavn", 
                      "Lyngby-taarbæk":"lyngby-taarbaek", 
                      "Rødovre":"roedovre", 
                      "Rudersdal":"rudersdal", 
                      "Tårnby":"taarnby", 
                      "Vallensbæk":"vallensbaek"
    }

    topics = [
        "Kommune specifikt", 
        "Trafik", 
        "Miljø og klima", 
        "Kultur,  idræt og fritid", 
        "Social- og integrationsområdet", 
        "Ældre", 
        "Sundhed", 
        "Skole/dagtilbud for børn", 
        "Erhverv/administration", 
        "Skat", 
        "Beredskab og sikkerhed", 
    ]

    party_colors = {'A': '#AF0D0D', # socialdemokratiet
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
        # ensures consistency in giving the same color to all parties in all graphs
        if letter in party_colors.keys():
            return party_colors[letter]
        else: # give smaller party a random color
            val = rand.uniform(0,1)
            randcolor = (val,val,val)
            party_colors[letter] = randcolor
            return randcolor
    return (
        PCA,
        StandardScaler,
        csv,
        get_color,
        math,
        mo,
        mun_dan_to_eng,
        np,
        pd,
        plt,
        topics,
    )


@app.cell
def _(mo, mun_dan_to_eng):
    muni = mo.ui.dropdown(mun_dan_to_eng.keys(), value="Albertslund", label = "See data from municipality:")
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
def _(chosen_size, get_color, math, mun_dan_to_eng, muni, np, pd, plt):
    chosen_file = pd.read_csv(f'./data/{mun_dan_to_eng[muni.value]}.csv')

    parties_srs = pd.Series(chosen_file['party'].value_counts())
    letters = parties_srs.index


    let_dict = {letter: [0,0] for letter in letters}

    # returns a dictionary containing each party's letter, number of votes, and how many were elected
    for i, cand in chosen_file.iterrows():
        num = cand['votes']
        ele = 1 if cand['elected'] else 0
        if not math.isnan(num): let_dict[cand['party']][0] += num
        let_dict[cand['party']][1] += ele

    color_elect = [get_color(letter) for letter in letters]

    def nested_pie_num_cand():
        lst = []
        color_not_elec = ['black', 'grey'] * len(letters) * 2

        # for each party, get the their corresponding color
        # return a numpy.array with elected, candidates that weren't elected, and color for each party
        for letter in letters:
            chosen_file = parties_srs[letter] # todo: rename
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
    return chosen_file, letters


@app.cell
def _(mo, muni, topics):
    topics[0] = f"{muni.value} Kommune"
    chosen_topic = mo.ui.dropdown(["Alle"] + topics, value = "Alle", label = "See PCA based on the topic of:")
    chosen_topic
    return (chosen_topic,)


@app.cell
def _(chosen_topic, csv, mun_dan_to_eng, muni):
    # getting data
    data = []

    with open('./data/' + mun_dan_to_eng[muni.value] + '.csv', mode='r') as file:
        content = csv.reader(file)
        for line in content:
            lst = []
            lst.append(line[5])
            if not line[10]: continue
            for _i in range(10, len(line)-3):
                if _i % 2 == 0: lst.append(line[_i])
                else: continue
            data.append(lst)

    description = f" on the topic of {chosen_topic.value}"

    def get_start_stop(): # albertslund numbers
        match chosen_topic.value:
            case "Trafik": 
                return 8,9
            case "Miljø og klima": 
                return 10,11
            case "Kultur,  idræt og fritid": 
                return 12,13
            case "Social- og integrationsområdet": 
                return 14,15
            case "Ældre": 
                return 16,17
            case "Sundhed": 
                return 18,19
            case "Skole/dagtilbud for børn": 
                return 20,21
            case "Erhverv/administration": 
                return 22,23
            case "Skat": 
                return 24,25
            case "Beredskab og sikkerhed": 
                return 26,27
            case _: # specific topics to the municipality
                return 1,7

    parties = [row[0] for row in data[1:]]
    #print(data[1:])

    if chosen_topic.value == "Alle":
        description = ""
        filtered = data[1:]
    else: 
        start, stop = get_start_stop()
        data = data[1:]
        filtered = [[parties[j]] + data[j][start:stop+1] for j in range(len(parties))] 
    #print(filtered)
    return description, filtered, parties


@app.cell
def _(PCA, StandardScaler, description, filtered, get_color, np, parties, plt):
    # PCA
    features = []
    for row in filtered:
        feat = []
        for v in row[1:]:
            if v: feat.append(float(v))
            else: feat.append(0.0)
        features.append(feat)
    features = np.array(features, dtype=float)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)

    unique_parties = sorted(set(parties))
    point_colors = [get_color(p) for p in parties]

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=point_colors, s=30, zorder=2)

    party_means = {}
    coords = np.array(pca_result)
    party_array = np.array(parties)
    for p in unique_parties:
        idx = np.where(party_array == p)[0]
        if idx.size == 0: continue
        pts = coords[idx]
        mean_pt = pts.mean(axis=0)
        party_means[p] = mean_pt
        for pt in pts:
            plt.plot([mean_pt[0], pt[0]], [mean_pt[1], pt[1]], color=get_color(p), linewidth=0.6, alpha=0.7, zorder=1)
        plt.scatter(mean_pt[0], mean_pt[1], color=get_color(p), edgecolor='k', s=140, marker='X', zorder=3)

    #plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    #plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title('PCA Analysis colored by party with party means' + description)
    for p in unique_parties:
        plt.scatter([], [], color=get_color(p), label=p)
    plt.legend(title='Party', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    forklaring af pca
    - ikke højre-venstre spektrum
    - få spørgsmål for temaer = mange punkter oven i hinanden, ikke så mange muligheder for forskellighed
    - kryds = gennemsnit for parti
    """)
    return


@app.cell
def _(chosen_file, get_color, letters, np, plt, topics):
    # topics
    # problem with / in topics
    topic_data = []
    for _, candrow in chosen_file.iterrows():
        candi = []
        themes = candrow['themes']
        if isinstance(themes, float): # nan / blank case
            themes = ""
        tmp = [candrow['party'], themes]
        topic_data.append(tmp)

    barfig, barax = plt.subplots()
    width = 0.5

    topic_count_chosen_by_party = []

    for party in letters:
        tmp = []
        for topic in topics[1:]:
            sum = 0
            for candi in topic_data:
                prioritized = candi[1]
                if not prioritized or candi[0] != party: 
                    continue
                elif topic.casefold() in prioritized.casefold(): 
                    sum += 1
            tmp.append(sum)
        topic_count_chosen_by_party.append(tmp)

    topic_count_chosen_by_party = np.array(topic_count_chosen_by_party)

    # create bars
    for ite in range(len(topic_count_chosen_by_party)):
        bottom = np.sum(topic_count_chosen_by_party[:ite], axis = 0)
        newbar = barax.bar(topics[1:], topic_count_chosen_by_party[ite], width, label=letters[ite], bottom=bottom, color = get_color(letters[ite]))

    # make lables readable 
    labels = barax.get_xticklabels()
    plt.setp(labels, rotation = 45, horizontalalignment = 'right')

    barax.set_title("Priotized topics seen by party")
    barax.legend(bbox_to_anchor=(1.2, 1.07))
    barax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    be aware of the bias in the graph above
    - more candidates running + answered questionnaire -> greater representation
    - prioritized on paper ≠ action
    - candidates choose the amount of themes they select. selecting more themes -> greater representation
    """)
    return


if __name__ == "__main__":
    app.run()
