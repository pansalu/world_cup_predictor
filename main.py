import pandas as pd

df = pd.read_csv("data/archive/results.csv")
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] > '2002-01-01'].copy()
df = df[df['tournament'] != 'Friendly'].copy() 
def determine_result(row):
    if row['home_score'] > row['away_score']: return 0
    if row['home_score'] < row['away_score']: return 1
    return 2

df['result'] = df.apply(determine_result, axis=1)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
le.fit(all_teams)


df['home_code'] = le.transform(df['home_team'])
df['opp_code'] = le.transform(df['away_team'])

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df['is_neutral'] = df['neutral'].astype(int)



y = df['result']
def get_win_rate(team, date, df_before):
    games = df_before[(df_before['home_team'] == team) & (df_before['date'] < date)]
    if len(games) == 0:
        return 0.5
    return (games['result'] == 0).sum() / len(games)

df['home_win_rate'] = df.apply(lambda row: get_win_rate(row['home_team'], row['date'], df), axis=1)
df['away_win_rate'] = df.apply(lambda row: get_win_rate(row['away_team'], row['date'], df), axis=1)
def get_avg_goals(team, date, df_before):
    games = df_before[(df_before['home_team'] == team) & (df_before['date'] < date)]
    if len(games) == 0:
        return 1.0
    return games['home_score'].mean()


def get_avg_conceded(team, date, df_before):
    games = df_before[(df_before['home_team'] == team) & (df_before['date'] < date)]
    if len(games) == 0:
        return 1.0
    return games['away_score'].mean()


def tournament_weight(t):
    if 'FIFA World Cup' in str(t): return 2
    return 1

def get_away_avg_goals(team, date, df_before):
    games = df_before[(df_before['away_team'] == team) & (df_before['date'] < date)]
    if len(games) == 0:
        return 1.0
    return games['away_score'].mean()


df['home_avg_goals'] = df.apply(lambda row: get_avg_goals(row['home_team'], row['date'], df), axis=1)
df['away_avg_goals'] = df.apply(lambda row: get_away_avg_goals(row['away_team'], row['date'], df), axis=1)
df['home_avg_conceded'] = df.apply(lambda row: get_avg_conceded(row['home_team'], row['date'], df), axis=1)
df['tournament_weight'] = df['tournament'].apply(tournament_weight)
X = df[['home_code', 'opp_code', 'is_neutral', 'home_win_rate', 'away_win_rate','home_avg_goals','away_avg_goals','home_avg_conceded','tournament_weight']]

train_df = df[df['date'] < '2022-01-01']
test_df = df[df['date'] >= '2022-01-01']

X_train = train_df[['home_code', 'opp_code', 'is_neutral', 'home_win_rate', 
                     'away_win_rate', 'home_avg_goals', 'away_avg_goals', 
                     'home_avg_conceded', 'tournament_weight']]
y_train = train_df['result']

X_test = test_df[['home_code', 'opp_code', 'is_neutral', 'home_win_rate', 
                   'away_win_rate', 'home_avg_goals', 'away_avg_goals', 
                   'home_avg_conceded', 'tournament_weight']]
y_test = test_df['result']


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


def predict_match(home_team, away_team):
    if home_team not in le.classes_ or away_team not in le.classes_:
        print("Team not found!")
        return
    
    today = pd.Timestamp.today()
    
    home_code = le.transform([home_team])[0]
    opp_code = le.transform([away_team])[0]
    is_neutral = 0
    home_wr = get_win_rate(home_team, today, df)
    away_wr = get_win_rate(away_team, today, df)
    home_goals = get_avg_goals(home_team, today, df)
    away_goals = get_away_avg_goals(away_team, today, df)
    home_conceded = get_avg_conceded(home_team, today, df)
    tournament_w = 1

    features = features = pd.DataFrame([[home_code, opp_code, is_neutral, home_wr, away_wr, 
             home_goals, away_goals, home_conceded, tournament_w]],
             columns=['home_code', 'opp_code', 'is_neutral', 'home_win_rate', 
                      'away_win_rate', 'home_avg_goals', 'away_avg_goals', 
                      'home_avg_conceded', 'tournament_weight'])
    
    proba = model.predict_proba(features)[0]
    
    print(f"\n{home_team} vs {away_team}")
    print(f"Home Win:  {round(proba[0]*100, 1)}%")
    print(f"Away Win:  {round(proba[1]*100, 1)}%")
    print(f"Draw:      {round(proba[2]*100, 1)}%")

predict_match("Brazil", "Germany")
predict_match("Portugal", "Argentina")
predict_match("Germany", "Argentina")
predict_match("Argentina", "Germany")


import joblib
import os

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
print("Model saved!")


