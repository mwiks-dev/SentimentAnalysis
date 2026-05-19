import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from flask import Flask, request, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')
    print("Download complete.")

sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    """
    Analyzes the sentiment of a given text using VADER.
    Returns a dictionary of sentiment scores.
    """
    scores = sia.polarity_scores(text)
    return scores

server = Flask(__name__)

@server.route("/")
def index():
    return "Welcome to the Sentiment Analysis API and Dashboard!"

@server.route('/api/sentiment', methods=['POST'])
def analyze_sentiment_api():
    """
    Flask API endpoint to get sentiment scores for a given text.
    Expects a JSON payload with a "text" key.
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid request. 'text' key is required."}), 400
    
    text_to_analyze = data['text']
    sentiment_scores = get_sentiment(text_to_analyze)
    return jsonify(sentiment_scores)

app = dash.Dash(__name__, server=server, routes_pathname_prefix='/dashboard/')

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '800px', 'margin': 'auto', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}, children=[
    html.H1(
        'Sentiment Analysis Dashboard',
        style={'textAlign': 'center', 'color': '#333'}
    ),
    html.P(
        'Enter text below to analyze its sentiment.',
        style={'textAlign': 'center', 'color': '#666'}
    ),
    
    dcc.Textarea(
        id='text-input',
        placeholder='Enter your text here...',
        style={
            'width': '100%', 
            'height': 150, 
            'padding': '10px',
            'borderRadius': '5px',
            'border': '1px solid #ccc',
            'fontSize': '16px'
        }
    ),
    
    html.Button(
        'Analyze Sentiment', 
        id='analyze-button', 
        n_clicks=0,
        style={
            'display': 'block',
            'margin': '20px auto',
            'padding': '10px 20px',
            'fontSize': '16px',
            'cursor': 'pointer',
            'backgroundColor': '#007BFF',
            'color': 'white',
            'border': 'none',
            'borderRadius': '5px'
        }
    ),
    
    dcc.Loading(
        id="loading-spinner",
        type="circle",
        children=dcc.Graph(id='sentiment-chart')
    )
])

@app.callback(
    Output('sentiment-chart', 'figure'),
    [Input('analyze-button', 'n_clicks')],
    [State('text-input', 'value')]
)
def update_chart(n_clicks, text_value):
    if n_clicks == 0 or not text_value:
        return go.Figure().update_layout(
            title_text='Please enter text and click "Analyze"',
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': 'Waiting for input...',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 16, 'color': '#888'}
            }]
        )
    
    scores = get_sentiment(text_value)
    
    labels = ['Negative', 'Neutral', 'Positive', 'Compound']
    values = [scores['neg'], scores['neu'], scores['pos'], scores['compound']]
    colors = ["#BA3A23", '#D3D3D3', '#32CD32', '#4682B4']
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f'{v:.3f}' for v in values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title_text='Sentiment Analysis Scores',
        xaxis_title='Sentiment',
        yaxis_title='Score',
        yaxis=dict(range=[-1, 1]) if 'Compound' in labels else dict(range=[0, 1]),
        plot_bgcolor='white',
        paper_bgcolor='#f9f9f9',
        font={'color': '#333'}
    )
    
    return fig

if __name__ == '__main__':
    app.run(
        host = '0.0.0.0',
        port=int(os.environ.get('PORT',10000)),
        debug=False
    )