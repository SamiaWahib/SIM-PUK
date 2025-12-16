import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("# data visualization with marimo")
    return (mo,)


@app.cell
def _(mo):
    import plotly.express as px
    import pandas as pd

    mo.md("#pick a municipality:")
    municipalities = mo.ui.dropdown(["koebenhavn", "frederiksberg", 
                      "hvidovre", "taarnby", 
                      "dragoer", "roedovre", 
                      "lyngby-taarbaek", 
                      "broendby", "gentofte", 
                      "gladsaxe", "herlev",
                      "ballerup", "glostrup",
                      "albertslund", "ishoej", "vallensbaek", 
                      "furesoe", "rudersdal", "greve"], value="koebenhavn")

    municipalities
    return municipalities, pd, px


@app.cell
def _(mo, municipalities, pd, px):
    data = pd.read_csv(f"./data/{municipalities.value}.csv")

    _plot = px.scatter(
        data, x="age", y="number of kids", color="party"
    )

    plot = mo.ui.plotly(_plot)
    return data, plot


@app.cell
def _(mo):
    mo.md("""
    # visualizing number of kids for each candidate and their age for chosen municipality
    """)
    return


@app.cell
def _(mo, plot):
    mo.hstack([plot, plot.value])
    return


app._unparsable_cell(
    r"""
    for mun32
    """,
    name="_"
)


@app.cell
def _(data, mo):
    mo.vstack(
        [
            mo.hstack(
                [
                    data
                ]
            )
        ]
    )
    # mo.ui.altair_chart(data)
    return


@app.cell
def _(mo):
    mo.md("""
    # elected candidates and their parties
    """)
    return


@app.cell
def _(mo, pd):
    import altair as alt

    mun = ["koebenhavn", "frederiksberg", 
                      "hvidovre", "taarnby", 
                      "dragoer", "roedovre", 
                      "lyngby-taarbaek", 
                      "broendby", "gentofte", 
                      "gladsaxe", "herlev",
                      "ballerup", "glostrup",
                      "albertslund", "ishoej", "vallensbaek", 
                      "furesoe", "rudersdal", "greve"]



    for m in mun:
        ele = pd.read_csv(f"./data/{m}.csv")
        for j in range(1,len(ele)):
            elected = ele[(ele['elected'] == True)]


    chart2 = alt.Chart(elected).mark_point().encode(
        x=alt.X("party"),
        y=alt.Y("age"),
        color="party"
    ).add_params()

    chart2 = mo.ui.altair_chart(chart2)
    chart2
    return


@app.cell
def _(mo):
    mo.md("""
    # interactive elements
    """)
    return


@app.cell
def _(mo):
    import polars as pl

    df = pl.read_csv("./data/frederiksberg.csv")
    mo.ui.data_explorer(df)
    return


@app.cell
def _(mo):
    mo.md("""
    # visualizing elected candidates from all municipalities
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
