import streamlit as st
import joblib
import pandas as pd

# Load data
df = pd.read_csv("data/archive/results.csv")
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] > '2002-01-01'].copy()
df = df[df['tournament'] != 'Friendly'].copy()

def determine_result(row):
    if row['home_score'] > row['away_score']: return 0
    if row['home_score'] < row['away_score']: return 1
    return 2

df['result'] = df.apply(determine_result, axis=1)

def get_win_rate(team, df_before):
    games = df_before[df_before['home_team'] == team]
    if len(games) == 0: return 0.5
    return (games['result'] == 0).sum() / len(games)

def get_avg_goals(team, df_before):
    games = df_before[df_before['home_team'] == team]
    if len(games) == 0: return 1.0
    return games['home_score'].mean()

def get_away_avg_goals(team, df_before):
    games = df_before[df_before['away_team'] == team]
    if len(games) == 0: return 1.0
    return games['away_score'].mean()

def get_avg_conceded(team, df_before):
    games = df_before[df_before['home_team'] == team]
    if len(games) == 0: return 1.0
    return games['away_score'].mean()

# Load model
model = joblib.load('models/model.pkl')
le = joblib.load('models/label_encoder.pkl')

# App UI
st.title("⚽ World Cup Match Predictor")
st.write("Predict the outcome of any international football match!")

major_teams = [
    "Brazil", "Germany", "France", "Argentina", "Spain", "England",
    "Portugal", "Netherlands", "Italy", "Belgium", "Croatia", "Uruguay",
    "Mexico", "Colombia", "Chile", "Peru", "Ecuador", "United States",
    "Senegal", "Morocco", "Nigeria", "Ghana", "Cameroon", "Egypt",
    "Japan", "South Korea", "Australia", "Iran", "Saudi Arabia", "Qatar",
    "Denmark", "Switzerland", "Poland", "Serbia", "Czech Republic",
    "Hungary", "Sweden", "Norway", "Wales", "Scotland", "Turkey",
    "Russia", "Ukraine", "Costa Rica", "Panama", "Honduras", "Paraguay",
    "Bolivia", "Venezuela", "Algeria", "Tunisia", "Ivory Coast", "Mali"
]


teams = sorted([t for t in major_teams if t in le.classes_])

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox(" Home Team", teams)
with col2:
    away_team = st.selectbox(" Away Team", teams)

is_neutral = st.checkbox("Neutral Venue?")

if st.button("Predict Result"):
    home_code = le.transform([home_team])[0]
    opp_code = le.transform([away_team])[0]
    home_wr = get_win_rate(home_team, df)
    away_wr = get_win_rate(away_team, df)
    home_goals = get_avg_goals(home_team, df)
    away_goals = get_away_avg_goals(away_team, df)
    home_conceded = get_avg_conceded(home_team, df)

    features = pd.DataFrame([[home_code, opp_code, is_neutral, home_wr, away_wr, 
             home_goals, away_goals, home_conceded,1]],
             columns=['home_code', 'opp_code', 'is_neutral', 'home_win_rate', 
                      'away_win_rate', 'home_avg_goals', 'away_avg_goals', 
                      'home_avg_conceded', 'tournament_weight'])

    proba = model.predict_proba(features)[0]

    st.subheader(f"{home_team} vs {away_team}")
    col1, col2, col3 = st.columns(3)
    col1.metric(" Home Win", f"{round(proba[0]*100, 1)}%")
    col2.metric(" Away Win", f"{round(proba[1]*100, 1)}%")
    col3.metric("Draw", f"{round(proba[2]*100, 1)}%")