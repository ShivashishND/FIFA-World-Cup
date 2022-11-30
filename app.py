from matplotlib.pyplot import text, xlabel
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

df = pd.read_csv('international_matches.csv')

df['away_team_result'] = np.where(df['home_team_result'] == 'Lose', 'Win', (np.where(df['home_team_result'] == 'Draw', 'Draw', 'Lose')))

all_teams = sorted(df['home_team'].unique())

with st.sidebar:
    box = st.radio(
        "What would you like to see?",
        ('Offence/Defence scores', 'Head to Head', 'Prediction'))

if box == 'Head to Head':
    with st.sidebar:
        team1 = st.selectbox('Choose team 1', all_teams)
        team2 = st.selectbox('Choose team 2', all_teams)
    team1_home = df['home_team'] == team1
    team1_away = df['away_team'] == team1
    team2_home = df['home_team'] == team2
    team2_away = df['away_team'] == team2
    team1_team2 = df[(team1_home & team2_away) | (team1_away & team2_home)]
    team1_team2_scores = team1_team2[['home_team', 'away_team', 'home_team_fifa_rank', 'away_team_fifa_rank','home_team_score', 'away_team_score', 'home_team_result', 'away_team_result']]

    team1_home_df = df[team1_home]
    team1_away_df = df[team1_away]
    team2_home_df = df[team2_home]
    team2_away_df = df[team2_away]
    peak_rank_team1 = min(team1_home_df['home_team_fifa_rank'].min(), team1_away_df['away_team_fifa_rank'].min())
    peak_rank_team2 = min(team2_home_df['home_team_fifa_rank'].min(), team2_away_df['away_team_fifa_rank'].min())

    peaks = {team1:peak_rank_team1, team2:peak_rank_team2}
    scores = {team1:0, team2:0}
    results = {team1:0,'draw':0,team2:0}
    for row, cols in team1_team2.iterrows():
        if cols['home_team_result'] == 'Draw':
            results['draw'] += 1
            scores[team1] += cols['home_team_score']
            scores[team2] += cols['home_team_score']
            
        elif cols['home_team_result'] == 'Win':
            if cols['home_team'] == team1:
                results[team1] += 1
                scores[team1] += cols['home_team_score']
                scores[team2] += cols['away_team_score']
            else:
                results[team2] += 1
                scores[team2] += cols['home_team_score']
                scores[team1] += cols['away_team_score']
        
        else:
            if cols['home_team'] == team1:
                results[team2] += 1
                scores[team1] += cols['home_team_score']
                scores[team2] += cols['away_team_score']
            else:
                results[team1] += 1
                scores[team2] += cols['home_team_score']
                scores[team1] += cols['away_team_score']

    results = pd.DataFrame.from_dict(results, orient = 'index')
    scores = pd.DataFrame.from_dict(scores, orient = 'index')
    peaks = pd.DataFrame.from_dict(peaks, orient = 'index')

    results.reset_index(inplace=True)
    scores.reset_index(inplace=True)
    peaks.reset_index(inplace=True)

    results.columns = ['Teams','Results']
    scores.columns = ['Teams', 'Goals']
    peaks.columns = ['Teams', 'Rank']

    peak_chart = (
        alt.Chart(peaks)
        .mark_bar()
        .encode(
            alt.X("Teams:O"),
            alt.Y("Rank:Q"),
            alt.Color("Teams:O",scale=alt.Scale(scheme='dark2')),
            alt.Tooltip(["Teams", "Rank"]),
        )
        .properties(
        height=600,
        width=800
    )
    )

    result_chart = (
        alt.Chart(results)
        .mark_bar()
        .encode(
            alt.X("Teams:O"),
            alt.Y("Results:Q"),
            alt.Color("Teams:O",scale=alt.Scale(scheme='dark2')),
            alt.Tooltip(["Teams", "Results"]),
        )
        .properties(
        height=600,
        width=800
    )
    )

    score_chart = (
        alt.Chart(scores)
        .mark_bar()
        .encode(
            alt.X("Teams:O"),
            alt.Y("Goals:Q"),
            alt.Color("Teams:O",scale=alt.Scale(scheme='dark2')),
            alt.Tooltip(["Teams", "Goals"])
        )
        .properties(
        height=600,
        width=800
    )
    )

    peak_text = peak_chart.mark_text(align="center", baseline="bottom").encode(
        text=alt.Text("Rank:Q",format=",.0f")
    )

    result_text = result_chart.mark_text(align="center", baseline="bottom").encode(
        text=alt.Text("Results:Q",format=",.0f")
    )

    score_text = score_chart.mark_text(align="center", baseline="bottom").encode(
        text=alt.Text("Goals:Q",format=",.0f")
    )

    st.title("Peak Ranks")
    st.write(peak_chart+peak_text)
    st.title("Wins against each other")
    st.write(result_chart+result_text)
    st.title("Goals against each other")
    st.write(score_chart+score_text)

# # st.write(scores)