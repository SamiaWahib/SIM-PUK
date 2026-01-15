import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


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
    import glob
    import os

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

    letter_name = {'A': "Socialdemokratiet",
                'B': 'Radikale Venstre',
                'C': 'Det konservative folkeparti',
                'D': 'Nye borgerlige',
                'F': 'Socialistisk Folkeparti',
                'I': 'Liberal alliance',
                'M': 'Moderaterne',
                'O': 'Dansk Folkeparti',
                'V': 'Venstre',
                'Å': 'Alternativet',
                'Æ': 'Danmarks demokraterne',
                'Ø': 'Enhedslisten'
    }

    used = []
    def get_color(letter):
        # ensures consistency in giving the same color to all parties in all graphs
        if letter in party_colors.keys():
            return party_colors[letter]
        else: # give smaller party a random color
            val = rand.randint(50,220) # if val=255 then the color is white, val=0 gives black
            while val in used: # check that the color is not already being used for another party
                val = rand.randint(50,220)
            used.append(val)
            val = hex(val)[2:] # ignore '0x'
            randcolor = f'#{val}{val}{val}' # when r=g=b it gives color on black-white scale
            party_colors[letter] = randcolor
            return randcolor

    def mysum(lst):
        sumi = 0
        for e in lst:
            sumi += e
        return sumi
    return (
        PCA,
        StandardScaler,
        csv,
        get_color,
        glob,
        letter_name,
        math,
        mo,
        mun_dan_to_eng,
        mysum,
        np,
        os,
        party_colors,
        pd,
        plt,
        topics,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Visualisation of the municipal election of 2025
    We have scraped TV2's results of the candidate quiz for the municipal election in November of 2025 for a number of municipalities near Copnehagen/København.
    The quiz asks the candidates for some personal information, like age, relationship status and then presents a number of politcal statements, asking them to rate to what extent they agree with the statement on a 5-point Likert scale. The data does not include any elements referring to the actions of the parties or candidates, only what they claim to represent and prioritize.

    We here present a visualization of some of the data, displaying different qualities of the candidates as a group and the parties that they represent.

    All candidates have been invited to answer the quiz, and while most candidates in the relevant municipalities have chosen to take part, still other have not. We have not included anything from candidates that was not present on TV2.
    For every candidate we know: Their name, party, number of votes received and if they were elected.
    Everything else is optional; and candidates may choose not to give any more information.
    """)
    return


@app.cell
def _(mo, mun_dan_to_eng):
    muni = mo.ui.dropdown(mun_dan_to_eng.keys(), value="Albertslund", label="## See data from municipality:")
    muni
    return (muni,)


@app.cell
def _(get_color, letter_name, letters, mo, muni, party_colors):
    md = "\n".join(
        f'<span style="display:inline-block;width:35px;height:10px;background:{party_colors[l] if l in party_colors.keys() else get_color(l)};"></span> {l} - {letter_name[l] if l in letter_name.keys() else "Lokalliste"} \n'
        for l in sorted(letters)
    )

    _start = f"All parties who ran in {muni.value}, in alphabetic order: <BR>"

    mo.md(_start + md)
    return


@app.cell
def _(mo):
    mo.md(rf"""
    # Party sizes
    """)
    return


@app.cell
def _(chosen_file, letters, mo, num_elec):
    mo.md(f"""
    ## {chosen_file.shape[0]} candidates ran for {len(letters)} different parties
    ## {num_elec[True]} candidates were elected
    """)
    return


@app.cell
def _(mo):
    options = ["candidates", "votes"]
    chosen_size = mo.ui.dropdown(options, value = options[0], label = "## See party size based on number of:")
    chosen_size
    return (chosen_size,)


@app.cell(hide_code=True)
def _(math, np, pd, plt):
    def info_on_cand(file):
        parties = pd.Series(file['party'].value_counts())
        letters = parties.index

        let_dict = {letter: [0,0] for letter in letters}

        # returns a dictionary containing each party's letter, number of votes, and how many were elected
        for i, cand in file.iterrows():
            num = cand['votes']
            ele = 1 if cand['elected'] else 0
            if not math.isnan(num): let_dict[cand['party']][0] += num
            #if cand['party'] == 'Ø': print(num)
            let_dict[cand['party']][1] += ele

        return parties, letters, let_dict

    def nested_pie_num_cand(f, l, dic, p, c, s):
        lst = []
        color_not_elec = ['black', 'grey'] * len(l) * 2

        # for each party, get the their corresponding color
        # return a numpy.array with elected, candidates that weren't elected, and color for each party
        for letter in l:
            f = p[letter] # todo: rename
            elected = dic[letter][1]
            lst.append([elected, f-elected])
        lst = np.array(lst)

        fig, ax = plt.subplots()
        size = 0.3

        ax.pie(lst.sum(axis=1), radius=1, colors=c,
               wedgeprops=dict(width=size, edgecolor='w'), labels=l)

        ax.pie(lst.flatten(), radius=1-size, colors=color_not_elec,
               wedgeprops=dict(width=size, edgecolor='w')) 

        ax.set(aspect="equal", title=s + f'\n {color_not_elec[1]} = not elected, {color_not_elec[0]} = elected')
        plt.legend(l, loc=(-0.5,0))
        return ax

    def pie_num_votes(dic, l, col):
        votes = [dic[letter][0] for letter in l]
        size = 0.6

        fig, ax = plt.subplots()

        ax.pie(votes, radius=1, colors = col,
               wedgeprops=dict(width=size, edgecolor='w'), labels=l)
        ax.set(title = "Party sizes by number of votes")

        plt.legend(
            l, loc=(-0.5,0))
        return ax
    return info_on_cand, nested_pie_num_cand, pie_num_votes


@app.cell(hide_code=True)
def _(
    chosen_size,
    get_color,
    info_on_cand,
    mun_dan_to_eng,
    muni,
    nested_pie_num_cand,
    pd,
    pie_num_votes,
):
    chosen_file = pd.read_csv(f'./data/{mun_dan_to_eng[muni.value]}.csv')
    num_elec = pd.Series(chosen_file['elected'].value_counts())

    ps, letters, let_dict = info_on_cand(chosen_file)
    color_elect = [get_color(letter) for letter in letters]
    txt_for_muni = f'Party size by number of candidates in {muni.value}.'

    if chosen_size.value == "candidates":
        ax = nested_pie_num_cand(chosen_file, letters, let_dict, ps, color_elect, txt_for_muni)
    else:
        ax = pie_num_votes(let_dict, letters, color_elect)
    ax
    return chosen_file, letters, num_elec


@app.cell(hide_code=True)
def _(
    chosen_size,
    get_color,
    glob,
    info_on_cand,
    nested_pie_num_cand,
    os,
    pd,
    pie_num_votes,
):
    all_files = os.path.join("./data/", "*.csv")
    all_list = glob.glob(all_files)

    # joining all municipality files 
    all_muni = pd.concat(map(pd.read_csv, all_list), ignore_index=True)

    all_parties, all_letters, all_let_dict = info_on_cand(all_muni)
    all_color_elect = [get_color(letter) for letter in all_letters]
    txt_for_all_muni = f'Party size by number of candidates in all municipalities.'

    if chosen_size.value == "candidates":
        ax2 = nested_pie_num_cand(all_files, all_letters, all_let_dict, all_parties, all_color_elect, txt_for_all_muni)
    else:
        ax2 = pie_num_votes(all_let_dict, all_letters, all_color_elect)
    ax2
    return


@app.cell(hide_code=True)
def _(chosen_file, pd):
    postals = pd.Series(chosen_file['postal number'].value_counts())
    return (postals,)


@app.cell
def _(mo, muni, postals):
    mo.md(rf"""
    # Postal codes of running candidates
    There are candidates running from {len(postals.index)} different postal codes in {muni.value}.
    """)
    return


@app.cell(hide_code=True)
def _(plt, postals):
    _labels = postals.index
    size = 0.6
    _fig, postalax = plt.subplots()
    postalax.pie(postals.to_list(), radius=1,
               wedgeprops=dict(width=size, edgecolor='w'), labels=_labels)
    plt.legend(_labels, loc=(1.5,0.3))
    return


@app.cell
def _(mo, muni):
    mo.md(rf"""
    # Age distributions
    /// details | Additional information
    The box plot below shows the age distribution for all candidates in {muni.value} and all elected candidates in {muni.value}. The data only includes candidates that have chosen to write in their ages to TV2.

    Outliers are represented by small circles.
    The boxes show the 50% of ages closest to the median.
    The line in the middle of each box shows the median, that is the value in the middle of the age distribution.
    ///
    """)
    return


@app.cell
def _(chosen_file, muni, pd, plt):
    # boksplot
    _fig, _ax = plt.subplots()

    _elecages = chosen_file.query('elected == True')
    elecages = pd.Series(_elecages['age']).dropna()
    allages = pd.Series(chosen_file['age']).dropna()

    _ax.boxplot([allages, elecages], tick_labels=["All candidates", "Elected candidates"])
    _ax.set_title(f'Age distribution for candidates in {muni.value}')
    _ax.set_ylabel('Ages')

    plt.tight_layout() 
    _fig
    return allages, elecages


@app.cell
def _(allages, elecages, mo, muni):
    mo.md(rf"""
    ### The age of candidates in {muni.value} range from {int(min(allages))} to {int(max(allages))}.
    ### The age of elected candidates in {muni.value} range from {int(min(elecages))} to {int(max(elecages))}.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # PCA
    /// details | Details about the PCA

    Every point represents a candidate, with their party color, and has a line to an 'X', representing the mean position of the party.
    If an 'X' appears to have no lines going out of it, it is because it is a party where only a single candidate has submitted their answers to the questions.

    Only candidates that have submitted their answers to the candidate quiz on TV2 appear in the graph. This may mean that some parties are over- or underrepresented.

    The principal component analysis is a technique that reduces the dimensionalities and features in the data set while keeping the most important information. So the multiple dimensions from the election data is reduced down to two dimensions to make it more readable.

    However, politics is not just a simple left-right wing spectrum and contains more complex dimensions. We cannot see all of these dimensions in a simple graph, and so some information is lost when scaling down the dimension. And the plot below does not show a typical left-right spectrum, but instead allows us to see the candidates and parties in relation to each other.

    When choosing to see the PCA for a specific topic, there might not be a great variation between the positions of the candidates. This is because not many questions were asked for every topic, and so the spread of possible positions may be small. This also causes many candidates to share a position, meaning the points representing them are directly in top of each other, only allowing us to see the color of the candidate last added with that position. Because of this, the party means may be more telling when it comes to the PCA for a specific topic.


    ///
    """)
    return


@app.cell
def _(mo, muni, topics):
    topics[0] = f"{muni.value} Kommune"
    chosen_topic = mo.ui.dropdown(["Alle"] + topics, value = "Alle", label = "### See PCA based on the topic of:")
    chosen_topic
    return (chosen_topic,)


@app.cell(hide_code=True)
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

    if chosen_topic.value == "Alle":
        description = ""
        filtered = data[1:]
    else: 
        start, stop = get_start_stop()
        data = data[1:]
        filtered = [[parties[j]] + data[j][start:stop+1] for j in range(len(parties))] 
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
    return (party_means,)


@app.cell(hide_code=True)
def _(np, party_means):
    closest = [float('inf'), '', '']
    furthest = [float('-inf'), '', '']

    for _keya,_vala in party_means.items():
        for _keyb, _valb in party_means.items():
            if _keya == _keyb: continue
            dist = np.linalg.norm(_vala - _valb)
            if dist < closest[0]:
                closest = [dist, _keya, _keyb]
            elif dist > furthest[0]:
                furthest = [dist, _keya, _keyb]
    return closest, furthest


@app.cell(hide_code=True)
def _(closest, furthest, mo):
    mo.md(f"""
    /// admonition | Based on the PCA and chosen topic

    Two closest parties are {closest[1]} and {closest[2]}. <BR>
    Two furthest parties are {furthest[1]} and {furthest[2]}.
    ///
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Prioritized topics
    Candidates were asked to choose the topics most important to them from a list of topics. <BR>
    The amount of topics chosen by a candidate differs.
    """)
    return


@app.cell(hide_code=True)
def _(letters, mo):
    #newmuni = mo.ui.dropdown(mun_dan_to_eng.keys(), value=muni.value, label = "See data from municipality:")
    #muni
    topic_party = mo.ui.dropdown(sorted(list(letters)), value="A", label = "### See most popular prioritized topics of: ")
    #newmuni,
    topic_party
    return (topic_party,)


@app.cell(hide_code=True)
def _(chosen_file, letters, mysum, np, topic_party, topics):
    # topics
    topic_data = []
    for _, candrow in chosen_file.iterrows():
        candi = []
        themes = candrow['themes']
        if isinstance(themes, float): # nan / blank case
            themes = ""
        tmp = [candrow['party'], themes]
        topic_data.append(tmp)

    topic_count_by_party = []
    stats_for_chosen_party = []

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
        topic_count_by_party.append(tmp)
        if party == topic_party.value:
            stats_for_chosen_party = tmp

    # handle all topics and parties
    topic_count_by_party = np.array(topic_count_by_party)
    top_sums = topic_count_by_party.sum(0)
    top_sums = sorted([[top_sums[i],topics[1:][i]] for i in range(len(top_sums))])

    # handle the chosen party
    chosen_num_topic = [[stats_for_chosen_party[i],topics[1:][i]] for i in range(len(stats_for_chosen_party))]

    if mysum(stats_for_chosen_party) == 0:
        chosen_num_topic = [['','']] * len(topics[1:])
    chosen_num_topic = sorted(chosen_num_topic)

    if not stats_for_chosen_party: 
        stats_for_chosen_party = ['']*10
    return (
        chosen_num_topic,
        stats_for_chosen_party,
        top_sums,
        topic_count_by_party,
    )


@app.cell(hide_code=True)
def _(chosen_num_topic, letter_name, mo, muni, top_sums, topic_party):
    mo.md(rf"""
    /// admonition | Most prioritized topics
    If two or more topics in the top three are equally popular, then they are listed in a random order.
    ## Top three topics for all candidates in {muni.value}
    ### 🥇 1. {top_sums[-1][1]}
    ### 🥈 2. {top_sums[-2][1]}
    ### 🥉 3. {top_sums[-3][1]} 

    ## Top three topics for candidates from the party: {letter_name[topic_party.value]} in {muni.value}
    ### 🥇 1. {chosen_num_topic[-1][1]}
    ### 🥈 2. {chosen_num_topic[-2][1]}
    ### 🥉 3. {chosen_num_topic[-3][1]}
    ///
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    /// details | Bias in graphs
    The graphs below may have a significant amount of bias.

    Not every candidate has chosen the same amount of prioritized topics. If a candidate choose more topics, they have a greater representation in the graphs. The same goes for parties overall.

    The more candidates are running and have answered the quiz from a party, the greater the party is represented.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(
    get_color,
    letters,
    np,
    plt,
    stats_for_chosen_party,
    topic_count_by_party,
    topic_party,
    topics,
):
    barfig, barax = plt.subplots(1,2)
    allbar = barax[0]
    pbar = barax[1]
    width = 0.5
    # create bar chart for all topics and parties
    for ite in range(len(topic_count_by_party)):
        bottom = np.sum(topic_count_by_party[:ite], axis = 0)
        newbar = allbar.bar(topics[1:], topic_count_by_party[ite], width, label=letters[ite], bottom=bottom, color = get_color(letters[ite]))

    allbar.set_title("Priotized topics seen by party")
    labels = allbar.get_xticklabels()
    plt.setp(labels, rotation = 45, horizontalalignment = 'right')

    # create bar chart for chosen party
    newbar = pbar.bar(topics[1:], stats_for_chosen_party, width, color = get_color(topic_party.value))
    pbar.set_title(f"Priotized topics for party: {topic_party.value}")
    labels = pbar.get_xticklabels()
    plt.setp(labels, rotation = 45, horizontalalignment = 'right')

    # show the two bar graphs
    plt.tight_layout()
    allbar.legend(bbox_to_anchor=(-0.5, 1.07))
    pbar
    return


if __name__ == "__main__":
    app.run()
