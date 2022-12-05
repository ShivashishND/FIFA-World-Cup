import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
from collections import Counter

df = pd.read_csv('international_matches.csv')
world_cup_df = pd.read_csv('WorldCups.csv')

fifa2022_teams = ['Qatar', 'Ecuador', 'Senegal', 'Netherlands',
                'England', 'IR Iran', 'USA', 'Wales',
                'Argentina', 'Saudi Arabia', 'Mexico', 'Poland',
                'France', 'Australia', 'Denmark', 'Tunisia',
                'Spain', 'Costa Rica', 'Germany', 'Japan',
                'Belgium', 'Canada', 'Morocco', 'Croatia',
                'Brazil', 'Serbia', 'Switzerland', 'Cameroon',
                'Portugal', 'Ghana', 'Uruguay', 'Korea Republic']

df['away_team_result'] = np.where(df['home_team_result'] == 'Lose', 'Win', (np.where(df['home_team_result'] == 'Draw', 'Draw', 'Lose')))
df['rank_difference'] = df['home_team_fifa_rank'] - df['away_team_fifa_rank']
df['average_rank'] = (df['home_team_fifa_rank'] + df['away_team_fifa_rank'])/2
df['point_difference'] = df['home_team_total_fifa_points'] - df['away_team_total_fifa_points']
df['score_difference'] = df['home_team_score'] - df['away_team_score']
df['is_won'] = df['score_difference'] > 0 # take draw as lost
df['is_stake'] = df['tournament'] != 'Friendly'

world_cup_df['GoalsPerMatches'] = world_cup_df['GoalsScored'] / world_cup_df['MatchesPlayed']

all_teams = sorted(df['home_team'].unique())

with st.sidebar:
    box = st.radio(
        "What would you like to see?",
        ('World Cup Summary', 'Offence/Defence scores', 'Head to Head', 'Prediction'))

if box == 'World Cup Summary':

    with st.sidebar:
        summary_box = st.radio("Choose", ("Overall Summary", "Stats by year"))

    if summary_box == 'Overall Summary':

        group_by_winner = world_cup_df.groupby(['Winner']).count()
        group_by_winner = group_by_winner[['Year']]
        group_by_winner.reset_index(inplace=True)
        group_by_winner.columns = ['Winner','Count']
        group_by_winner.sort_values(by='Count',ascending=False, inplace=True)

        # Most Wins
        most_wins_chart = alt.Chart(group_by_winner).mark_bar().encode(
        x=alt.X("Winner:O",sort='-y', title="Country"),
        y=alt.Y('Count', title="World Cup Wins"),
        color = alt.Color("Winner:O",scale=alt.Scale(scheme='dark2'))
        ).properties(width=600)
        st.title("World cup wins by country")
        st.write(most_wins_chart)

        # Goals per matches by years
        st.title("Goals per matches over the years")
        goals_per_matches_chart = alt.Chart(world_cup_df).mark_bar().encode(
        x='Year:O',
        y=alt.Y('GoalsPerMatches:Q', title="Goals per matches"),
        color = alt.Color("Year:O",scale=alt.Scale(scheme='dark2'))
        ).properties(width=600)

        mean_rule = alt.Chart(world_cup_df).mark_rule(color='red').encode(
        y='mean(GoalsPerMatches):Q'
        )

        st.write(goals_per_matches_chart+mean_rule)
    
    if summary_box == "Stats by year":
        years = world_cup_df['Year'].sort_values(ascending=False)
        with st.sidebar:
            input_year = st.selectbox("Choose Year", years)
        row = world_cup_df[world_cup_df['Year'] == input_year]
        st.title(input_year)
        col1, col2 = st.columns(2)

        
        winner = row['Winner'].values[0]
        runner_up = row['Runners-Up'].values[0]
        total_teams = row['QualifiedTeams'].values[0]
        total_match = row['MatchesPlayed'].values[0]
        total_goals = row['GoalsScored'].values[0]
        with col1:
            st.metric(label="Winner", value=winner)
            st.metric(label="Runner up", value=runner_up)
        with col2:
            st.metric(label="Total Teams", value=total_teams)
            st.metric(label="Total Matches", value=total_match)
            st.metric(label="Total Goals", value=total_goals)

if box == "Offence/Defence scores":

    with st.sidebar:
        suboption = st.radio('Select how to view the offence/defence scores',
        ('Get Offence/Defence score of a country','Get all countries of a particular score'))

    columns_with_null = [col for col in df.columns if df[col].isnull().any()]
    for x in fifa2022_teams:
        for y in columns_with_null:
            df[y].fillna(df[df[y[0:9]]==x][y].mean(), inplace=True)

    Teams_goalkeeper_score = []
    for x in fifa2022_teams:
        gk_score = np.round((df[df["home_team"]==x]['home_team_goalkeeper_score'].mean() + df[df["away_team"]==x]['away_team_goalkeeper_score'].mean())/2, 2)
        Teams_goalkeeper_score.append(gk_score)

    Teams_goalkeeper_scores = pd.DataFrame({'Team':fifa2022_teams, 'Gk score': Teams_goalkeeper_score}).sort_values('Gk score', ascending=False).reset_index(drop=True)
    Teams_goalkeeper_scores.index += 1

    bar_chart_gk_score = alt.Chart(Teams_goalkeeper_scores).mark_bar().encode(
    y=alt.Y('Gk score:Q',title='Goal Keeper Score'),
    x=alt.X('Team:N',sort='-y',title='FIFA 2022 Teams'),
    color = alt.Color("Gk score:O",scale=alt.Scale(scheme='dark2'))
    ).properties(
        height=700,
        width = 600
    )

    bar_chart_gk_score_text = bar_chart_gk_score.mark_text(align="center", baseline="bottom").encode(
        text=alt.Text("Gk score:Q",format=",.0f")
    )

    Teams_df_score = []
    for x in fifa2022_teams:
        df_score = np.round((df[df["home_team"]==x]['home_team_mean_defense_score'].mean() + df[df["away_team"]==x]['away_team_mean_defense_score'].mean())/2, 2)
        Teams_df_score.append(df_score)

    Teams_defense_scores = pd.DataFrame({'Team':fifa2022_teams, 'Df score': Teams_df_score}).sort_values('Df score', ascending=False).reset_index(drop=True)
    Teams_defense_scores.index += 1

    bar_chart_df_score = alt.Chart(Teams_defense_scores).mark_bar().encode(
    y=alt.Y('Df score:Q',title='Defense Score'),
    x=alt.X('Team:N',sort='-y',title='FIFA 2022 Teams'),
    color = alt.Color("Df score:O",scale=alt.Scale(scheme='dark2'))
    ).properties(
        height=700,
        width = 600
    )

    bar_chart_df_score_text = bar_chart_df_score.mark_text(align="center", baseline="bottom").encode(
    text=alt.Text("Df score:Q",format=",.0f")
    )

    Teams_of_score = []
    for x in fifa2022_teams:
        of_score = np.round((df[df["home_team"]==x]['home_team_mean_offense_score'].mean() + df[df["away_team"]==x]['away_team_mean_offense_score'].mean())/2, 2)
        Teams_of_score.append(of_score)

    Teams_offence_scores = pd.DataFrame({'Team':fifa2022_teams, 'Of score': Teams_of_score}).sort_values('Of score', ascending=False).reset_index(drop=True)
    Teams_offence_scores.index += 1

    bar_chart_of_score = alt.Chart(Teams_offence_scores).mark_bar().encode(
    y=alt.Y('Of score:Q',title='Offence Score'),
    x=alt.X('Team:N',sort='-y',title='FIFA 2022 Teams'),
    color = alt.Color("Of score:O",scale=alt.Scale(scheme='dark2'))
    ).properties(
        height=700,
        width = 600
    )

    bar_chart_of_score_text = bar_chart_of_score.mark_text(align="center", baseline="bottom").encode(
    text=alt.Text("Of score:Q",format=",.0f")
    )

    Team_midfield_score = []
    for x in fifa2022_teams:
        md_score = np.round((df[df["home_team"]==x]['home_team_mean_midfield_score'].mean() + df[df["away_team"]==x]['away_team_mean_midfield_score'].mean())/2, 2)
        Team_midfield_score.append(md_score)

    Team_midfield_scores = pd.DataFrame({'Team':fifa2022_teams, 'Md score': Team_midfield_score}).sort_values('Md score', ascending=False).reset_index(drop=True)
    Team_midfield_scores.index += 1

    bar_chart_mid_score = alt.Chart(Team_midfield_scores).mark_bar().encode(
    y=alt.Y('Md score:Q',title='Mid Field Score'),
    x=alt.X('Team:N',sort='-y',title='FIFA 2022 Teams'),
    color = alt.Color("Md score:O",scale=alt.Scale(scheme='dark2'))
    ).properties(
        height=700,
        width = 600
    )

    bar_chart_mid_score_text = bar_chart_mid_score.mark_text(align="center", baseline="bottom").encode(
    text=alt.Text("Md score:Q",format=",.0f")
    )

    df_1 = pd.merge(Teams_goalkeeper_scores, Teams_defense_scores, on="Team")
    df_2 = pd.merge(Teams_offence_scores, Team_midfield_scores, on="Team")
    df_combine = pd.merge(df_1, df_2, on="Team")

    if suboption == 'Get all countries of a particular score':

        with st.sidebar:
            score_type = st.radio(
                "Select a score to visualize",
                ('Goalkeeper score', 'Defense score', 'Offense score', 'Mid field score'))

        if score_type == 'Goalkeeper score':
            st.write(bar_chart_gk_score + bar_chart_gk_score_text)
        elif score_type == 'Defense score':
            st.write(bar_chart_df_score + bar_chart_df_score_text)
        elif score_type == 'Offense score':
            st.write(bar_chart_of_score + bar_chart_of_score_text)
        elif score_type == 'Mid field score':
            st.write(bar_chart_mid_score + bar_chart_mid_score_text)
    
    if suboption == 'Get Offence/Defence score of a country':
        with st.sidebar:
            team = st.selectbox(
            'select a Team',
            fifa2022_teams)
                    
        df_team = df_combine.loc[df_combine['Team'] == team]
        dft = df_team.set_index('Team').T
        dft.reset_index(inplace=True)
        dft.columns = ['scores',team]
        team_score_chart = (
            alt.Chart(dft)
            .mark_bar()
            .encode(
                alt.Y(team+":Q"),
                alt.X("scores:O"),
                alt.Color("scores:O",scale=alt.Scale(scheme='dark2')),
                alt.Tooltip([team, "scores"]),
            )
            .properties(
            height=300,
            width=600
        )
        )

        team_score_chart_text = team_score_chart.mark_text(align="center", baseline="bottom").encode(
        text=alt.Text(team+":O",format=",.0f")
        )        

        st.write(team_score_chart + team_score_chart_text)   

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
        height=300,
        width=600
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
        height=300,
        width=600
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
        height=300,
        width=600
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

if box == 'Prediction':
    X, y = df.loc[:,['average_rank', 'rank_difference', 'point_difference']], df['is_won']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    logreg = linear_model.LogisticRegression(C=1e-5)
    features = PolynomialFeatures(degree=2)
    model = Pipeline([
        ('polynomial_features', features),
        ('logistic_regression', logreg)
    ])
    model = model.fit(X_train, y_train)
    world_cup_rankings_home = df[['home_team','home_team_fifa_rank','home_team_total_fifa_points']].loc[df['home_team'].isin(fifa2022_teams)& (df['date']>'2019-01-01')] #Get last 2 years of information (try to get all time too, get interesting results!)
    world_cup_rankings_away = df[['away_team','away_team_fifa_rank','away_team_total_fifa_points']].loc[df['away_team'].isin(fifa2022_teams)& (df['date']>'2019-01-01')]
    world_cup_rankings_home = world_cup_rankings_home.set_index(['home_team'])
    world_cup_rankings_home = world_cup_rankings_home.groupby('home_team').mean()
    world_cup_rankings_away = world_cup_rankings_away.groupby('away_team').mean()
    
    simulation_winners = list()
    simulation_results_winners = list()
    simulation_results_round16 = list()
    simulation_df_round16 = list()
    simulation_results_quarterfinal = list()
    simulation_df_quarterfinal = list()
    simulation_results_semifinal = list()
    simulation_df_semifinal = list()

    n_simulations = 1000

    for j in tqdm(range(n_simulations)):
        candidates = [ 'Netherlands', 'Senegal', 'England','USA', 'Argentina', 'Poland', 'France', 'Australia','Japan', 'Spain', 'Morocco', 'Croatia', 'Brazil', 'Switzerland', 'Portugal','Korea Republic']
        finals = ['round_of_16', 'quarterfinal', 'semifinal', 'final']
        for f in finals:
            
            iterations = int(len(candidates) / 2)
            winners = []
            prob = []
            for i in range(iterations):
                home = candidates[i*2]
                away = candidates[i*2+1]
                
                row = pd.DataFrame(np.array([[np.nan, np.nan, True]]), columns=X_test.columns)
                home_rank = world_cup_rankings_home.loc[home, 'home_team_fifa_rank']
                home_points = world_cup_rankings_home.loc[home, 'home_team_total_fifa_points']
                opp_rank = world_cup_rankings_away.loc[away, 'away_team_fifa_rank']
                opp_points = world_cup_rankings_away.loc[away, 'away_team_total_fifa_points']
                row['average_rank'] = (home_rank + opp_rank) / 2
                row['rank_difference'] = home_rank - opp_rank
                row['point_difference'] = home_points - opp_points
                home_win_prob = model.predict_proba(row)[:,1][0]
                
                simulated_outcome = np.random.binomial(1, home_win_prob)

                if simulated_outcome <= 0.5:
                    winners.append(away)
                else:
                    winners.append(home)

                if simulated_outcome <= 0.5:
                    prob.append(1 - simulated_outcome)
                else:
                    prob.append(simulated_outcome)
            
            if f == 'round_of_16':
                step_16 = ['round_16'] * 8
                candidates_round_16 = zip(step_16, winners, prob)
                df_candidates_round_16 = pd.DataFrame(candidates_round_16, columns = ['Step','Team','Prob'])
                simulation_df_round16.append(df_candidates_round_16)
                simulation_results_round16.append(winners)

            if f == 'quarterfinal':
                step_quarterfinal = ['quarterfinal'] * 4
                candidates_quarterfinal = zip(step_quarterfinal,winners, prob)
                df_candidates_quarterfinal = pd.DataFrame(candidates_quarterfinal, columns = ['Step','Team','Prob'])
                simulation_df_quarterfinal.append(df_candidates_quarterfinal)
                simulation_results_quarterfinal.append(winners)

            if f == 'semifinal':    
                step_semifinal = ['semifinal'] * 2
                candidates_semifinal = zip(step_semifinal,winners, prob)
                df_candidates_semifinal = pd.DataFrame(candidates_semifinal, columns = ['Step','Team','Prob'])
                simulation_df_semifinal.append(df_candidates_semifinal)
                simulation_results_semifinal.append(winners)

            if f == 'final':    
                step_final = ['final'] * 1
                candidates_final = zip(step_final,winners, prob)
                df_candidates_final = pd.DataFrame(candidates_final, columns = ['Step','Team','Prob'])
                simulation_winners.append(df_candidates_final)
                simulation_results_winners.append(winners)
            
            candidates = winners

    df_candidates_round_16 = pd.concat(simulation_df_round16)
    df_candidates_quarterfinal = pd.concat(simulation_df_quarterfinal)
    df_candidates_semifinal = pd.concat(simulation_df_semifinal)
    df_candidates_final = pd.concat(simulation_winners)

    df_results = pd.concat([df_candidates_round_16,df_candidates_quarterfinal,df_candidates_semifinal,df_candidates_final]) #final DataFrame

    simulation_results_round16 = sum(simulation_results_round16, [])
    simulation_results_quarterfinal = sum(simulation_results_quarterfinal, [])
    simulation_results_semifinal = sum(simulation_results_semifinal, [])
    simulations_winners = sum(simulation_results_winners, [])

    lst_results = [simulation_results_round16,simulation_results_quarterfinal,simulation_results_semifinal,simulations_winners]

    results = Counter(simulations_winners).most_common()
    x,y = zip(*results)

    winning_prediction = pd.DataFrame(
        {
            'teams' : x,
            'score' : y
        }
    )
    
    top_four = winning_prediction.nlargest(4,'score')
    st.title("Most Probable winners in order")
    st.subheader(top_four.iloc[0].values[0])
    st.subheader(top_four.iloc[1].values[0])
    st.subheader(top_four.iloc[2].values[0])
    st.subheader(top_four.iloc[3].values[0])
    st.title('Chances of winning')

    predict_chart = (
    alt.Chart(winning_prediction)
    .mark_bar()
    .encode(
        x = alt.X("teams:O", sort = '-y'),
        y = alt.Y("score"),
        color = alt.Color("teams:O",scale=alt.Scale(scheme='dark2')),
        tooltip = alt.Tooltip(["teams", "score"])
    )
    .properties(height=600,
    width=800)
    )

    st.write(predict_chart)
