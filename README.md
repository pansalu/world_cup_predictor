# World Cup Match Predictor

This project predicts the outcome of international football matches using historical FIFA match data and machine learning.

Built with Streamlit and scikit-learn.

## Features

- Predicts win/draw/loss probabilities between two teams
- Displays historical team statistics
- Uses past international match data for training
- Interactive Streamlit interface

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit

## Installation

```bash
git clone https://github.com/pansalu/world_cup_predictor.git
cd world_cup_predictor
pip install -r requirements.txt
streamlit run app/app.py
```

## Dataset

The model was trained using historical international football match data.

Dataset source:
- Kaggle International Football Results Dataset

## Model

The prediction system uses a machine learning classification model trained on:
- Team win rates
- Average goals scored
- Average goals conceded
- Historical match performance

## Example Prediction

Input:
- Brazil (Home) vs Argentina (Away)

Output:
- Brazil Win: 58%
- Draw: 27%
- Argentina Win: 15%

## Project Structure

world_cup_predictor/
WORLD_CUP_PREDICTOR/
├── app/
│   └── app.py          # Main Streamlit UI and prediction logic
├── data/               # Raw and processed datasets (Ignored by Git)
├── models/             # Trained ML models and encoders (Ignored by Git)
├── .gitignore          # Prevents large files from uploading
├── main.py             # Model training and data preprocessing script
└── README.md           # Project documentation

## Future Improvements

- Add FIFA rankings
- Improve prediction accuracy
- Add tournament simulation
- Deploy the project online

## Author

Panshul Sudhakaran