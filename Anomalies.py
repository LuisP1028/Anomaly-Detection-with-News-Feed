import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, RepeatVector, TimeDistributed, Dense, Input as KerasInput
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import dash
from dash import Dash, dcc, html, Input, Output, State, callback_context

# -----------------------------
# Global Settings & Constants
# -----------------------------
ALPHA_VANTAGE_KEY = ""
market_open = '09:30'
market_close = '16:00'
eastern = 'US/Eastern'
period = "60d"
interval = "1h"
seq_size = 4  # sequence length for LSTM

# -----------------------------
# Helper Functions
# -----------------------------
def fetch_and_process(symbol):
    """
    Download data from yfinance, convert index to Eastern time and filter for market hours.
    """
    data = yf.download(symbol, period=period, interval=interval)[["Open", "Close"]]
    data.index = pd.to_datetime(data.index)
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert(eastern)
    data = data.between_time(market_open, market_close)
    data.sort_index(inplace=True)
    data.dropna(inplace=True)
    return data

def to_sequences(x, seq_size=1):
    """Convert a DataFrame/Series into overlapping sequences."""
    sequences = []
    for i in range(len(x) - seq_size):
        sequences.append(x.iloc[i:(i + seq_size)].values)
    return np.array(sequences)

def fetch_basic_news(date):
    """
    Fetch basic market news from Alpha Vantage API with smart date filtering.
    """
    try:
        # Convert input date to datetime
        click_date = pd.to_datetime(date)
        current_date = datetime.now()
        
        # Set start date (always 3 days before)
        start_date = click_date - timedelta(days=3)
        
        # Set end date (either 3 days after or current date, whichever is earlier)
        temp_end_date = click_date + timedelta(days=3)
        end_date = min(temp_end_date, current_date)
        
        print(f"Anomaly date: {click_date.date()}")
        print(f"Current date: {current_date.date()}")
        print(f"Searching for news between {start_date.date()} and {end_date.date()}")
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": ALPHA_VANTAGE_KEY,
            "limit": 50,
            "sort": "RELEVANCE",
            "time_from": start_date.strftime("%Y%m%dT%H%M"),
            "time_to": end_date.strftime("%Y%m%dT%H%M")
        }
        
        print("Making API request...")
        print(f"API Parameters: {params}")
        
        response = requests.get(url, params=params)
        data = response.json()
        
        print(f"Response status: {response.status_code}")
        
        if "feed" not in data:
            print("No feed in response")
            print(f"Response data: {data}")
            return []
        
        print(f"Found {len(data['feed'])} total news items")
        
        news_items = []
        for item in data["feed"]:
            try:
                news_time = datetime.strptime(item["time_published"], "%Y%m%dT%H%M%S")
                
                # Only include news within our date range
                if start_date <= news_time <= end_date:
                    news_item = {
                        'title': item["title"],
                        'source': item["source"],
                        'url': item["url"],
                        'date': news_time,
                        'summary': item.get("summary", "No summary available"),
                        'sentiment': item.get("overall_sentiment_label", "N/A"),
                        'sentiment_score': item.get("overall_sentiment_score", "N/A"),
                        'relevance_score': item.get("relevance_score", "N/A"),
                        'topics': item.get("topics", [])
                    }
                    news_items.append(news_item)
                    print(f"Added news item: {news_item['title']} (Date: {news_time})")
            except Exception as e:
                print(f"Error parsing news item: {e}")
                print(f"Problematic item: {item}")
                continue
        
        print(f"Found {len(news_items)} relevant news items")
        return sorted(news_items, key=lambda x: x['date'], reverse=True)
        
    except Exception as e:
        print(f"Error in fetch_basic_news: {e}")
        return []

# -----------------------------
# LSTM Autoencoder Model
# -----------------------------
def build_model(input_shape):
    model = Sequential([
        KerasInput(shape=input_shape),
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        RepeatVector(input_shape[0]),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        TimeDistributed(Dense(input_shape[1]))
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

# -----------------------------
# Dash App Setup
# -----------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Stock Anomaly Detection Dashboard", className="dashboard-title"),
    html.Div([
        html.Div([
            html.Label("Symbol 1 (for LSTM training):"),
            dcc.Input(id="symbol1-input", type="text", value="AAPL", className="input-field"),
        ], style={'display': 'inline-block', 'margin-right': '20px'}),
        html.Div([
            html.Label("Symbol 2 (for comparison):"),
            dcc.Input(id="symbol2-input", type="text", value="MSFT", className="input-field"),
        ], style={'display': 'inline-block'}),
        html.Button("Submit", id="submit-button", n_clicks=0, className="submit-button")
    ], style={'margin': '10px'}, className="input-container"),
    
    html.Div([
        html.Label("Threshold (MAE):"),
        dcc.Slider(
            id="threshold-slider",
            min=0,
            max=1,
            step=0.01,
            value=0.3,
            marks={i/10: str(i/10) for i in range(11)},
            className="threshold-slider"
        ),
    ], style={'margin': '10px'}, className="slider-container"),
    
    dcc.Store(id="stored-data"),
    
    html.Div([
        dcc.Graph(id="mae-histogram", className="graph"),
        dcc.Graph(id="anomaly-plot-symbol1", className="graph"),
        dcc.Graph(id="anomaly-plot-symbol2", className="graph"),
    ], className="graphs-container"),
    
    html.Div(
        id="news-modal",
        className="modal",
        style={"display": "none"},
        children=[
            html.Div(
                className="modal-content",
                children=[
                    html.H3(id="news-modal-header", className="modal-header"),
                    html.Div(id="news-content", className="modal-body"),
                    html.Button("Close", id="close-modal", n_clicks=0, className="close-button")
                ]
            )
        ]
    )
])

# -----------------------------
# Callbacks
# -----------------------------
@app.callback(
    Output("stored-data", "data"),
    Input("submit-button", "n_clicks"),
    State("symbol1-input", "value"),
    State("symbol2-input", "value")
)
def update_data(n_clicks, symbol1, symbol2):
    if n_clicks is None or n_clicks == 0:
        symbol1 = "AAPL"
        symbol2 = "MSFT"
    
    data1 = fetch_and_process(symbol1)
    data2 = fetch_and_process(symbol2)
    
    if data1.empty or data2.empty:
        return {}
    
    start_date = max(data1.index.min(), data2.index.min())
    end_date = min(data1.index.max(), data2.index.max())
    data1 = data1.loc[start_date:end_date]
    data2 = data2.loc[start_date:end_date]
    
    data_combined = pd.concat([
        data1.add_prefix(f"{symbol1}_"),
        data2.add_prefix(f"{symbol2}_")
    ], axis=1)
    data_combined.interpolate(method='linear', inplace=True)
    
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    
    data_combined[[f"{symbol1}_Open", f"{symbol1}_Close"]] = scaler1.fit_transform(
        data_combined[[f"{symbol1}_Open", f"{symbol1}_Close"]]
    )
    data_combined[[f"{symbol2}_Open", f"{symbol2}_Close"]] = scaler2.fit_transform(
        data_combined[[f"{symbol2}_Open", f"{symbol2}_Close"]]
    )
    
    close_data = data_combined[[f"{symbol1}_Close"]]
    train, test = train_test_split(close_data, test_size=0.3, shuffle=False)
    
    trainX = to_sequences(train, seq_size)
    testX = to_sequences(test, seq_size)
    
    if len(trainX) == 0 or len(testX) == 0:
        return {}
    
    input_shape = (trainX.shape[1], trainX.shape[2])
    model = build_model(input_shape)
    model.fit(
        trainX, trainX,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )
    
    testPredict = model.predict(testX)
    testMAE = np.mean(np.abs(testPredict - testX), axis=(1,2))
    
    anomaly_df = test.copy().iloc[seq_size:].copy()
    anomaly_df["testMAE"] = testMAE
    anomaly_df["max_trainMAE"] = 0.3
    anomaly_df["anomaly"] = anomaly_df["testMAE"] > anomaly_df["max_trainMAE"]
    
    inv_data1 = scaler1.inverse_transform(data_combined[[f"{symbol1}_Open", f"{symbol1}_Close"]])
    inv_data2 = scaler2.inverse_transform(data_combined[[f"{symbol2}_Open", f"{symbol2}_Close"]])
    
    data_combined[f"{symbol1}_Close_Inverse"] = inv_data1[:, 1]
    data_combined[f"{symbol2}_Close_Inverse"] = inv_data2[:, 1]
    
    anomaly_df = anomaly_df.join(
        data_combined[[f"{symbol1}_Close_Inverse", f"{symbol2}_Close_Inverse"]],
        how='left'
    )
    
    anomalies = anomaly_df[anomaly_df["anomaly"]].copy()
    
    return {
        "symbol1": symbol1,
        "symbol2": symbol2,
        "dates": [str(d) for d in anomaly_df.index],
        "price1": anomaly_df[f"{symbol1}_Close_Inverse"].tolist(),
        "price2": anomaly_df[f"{symbol2}_Close_Inverse"].tolist(),
        "mae": anomaly_df["testMAE"].tolist(),
        "anomaly_flags": anomaly_df["anomaly"].tolist(),
        "anomaly_dates": [str(d) for d in anomalies.index],
        "anom_price1": anomalies[f"{symbol1}_Close_Inverse"].tolist(),
        "anom_price2": anomalies[f"{symbol2}_Close_Inverse"].tolist()
    }

@app.callback(
    [Output("mae-histogram", "figure"),
     Output("anomaly-plot-symbol1", "figure"),
     Output("anomaly-plot-symbol2", "figure")],
    [Input("stored-data", "data"),
     Input("threshold-slider", "value")]
)
def update_graphs(stored_data, threshold):
    if not stored_data:
        return go.Figure(), go.Figure(), go.Figure()
    
    symbol1 = stored_data.get("symbol1", "Symbol1")
    symbol2 = stored_data.get("symbol2", "Symbol2")
    dates = pd.to_datetime(stored_data.get("dates", []))
    price1 = np.array(stored_data.get("price1", []))
    price2 = np.array(stored_data.get("price2", []))
    mae = np.array(stored_data.get("mae", []))
    
    anomaly_flags = mae > threshold
    anomaly_dates = dates[anomaly_flags]
    anom_price1 = price1[anomaly_flags]
    anom_price2 = price2[anomaly_flags]
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=mae, nbinsx=30, name='Test MAE'))
    fig_hist.add_vline(
        x=threshold,
        line_width=3,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold",
        annotation_position="top right"
    )
    fig_hist.update_layout(
        title="Test MAE Distribution",
        xaxis_title="MAE",
        yaxis_title="Count",
        width=600,
        height=300
    )
    
    fig_anom1 = go.Figure()
    fig_anom1.add_trace(go.Scatter(
        x=dates,
        y=price1,
        mode='lines',
        name=f'{symbol1} Close Price'
    ))
    fig_anom1.add_trace(go.Scatter(
        x=anomaly_dates,
        y=anom_price1,
        mode='markers',
        marker=dict(color='red', size=8),
        name='Anomalies'
    ))
    fig_anom1.update_layout(
        title=f'{symbol1} Close Price with Anomalies Highlighted',
        xaxis_title="Date",
        yaxis_title="Price",
        width=800,
        height=400
    )
    
    fig_anom2 = go.Figure()
    fig_anom2.add_trace(go.Scatter(
        x=dates,
        y=price2,
        mode='lines',
        name=f'{symbol2} Close Price'
    ))
    fig_anom2.add_trace(go.Scatter(
        x=anomaly_dates,
        y=anom_price2,
        mode='markers',
        marker=dict(color='red', size=8),
        name='Anomalies'
    ))
    fig_anom2.update_layout(
        title=f'{symbol2} Close Price with Anomalies Highlighted',
        xaxis_title="Date",
        yaxis_title="Price",
        width=800,
        height=400
    )
    
    return fig_hist, fig_anom1, fig_anom2

@app.callback(
    [Output("news-modal", "style"),
     Output("news-modal-header", "children"),
     Output("news-content", "children")],
    [Input("anomaly-plot-symbol1", "clickData"),
     Input("close-modal", "n_clicks")],
    [State("stored-data", "data"),
     State("news-modal", "style")]
)
def display_news(clickData, close_clicks, stored_data, current_style):
    ctx = callback_context
    if not ctx.triggered:
        return {"display": "none"}, "", []
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "close-modal":
        return {"display": "none"}, "", []
    
    if not clickData or not stored_data:
        return {"display": "none"}, "", []
    
    point_data = clickData["points"][0]
    click_date = point_data["x"]
    symbol = stored_data["symbol1"]
    
    print(f"Clicked date: {click_date}")
    
    # Fetch news using new function with smart date handling
    news_items = fetch_basic_news(click_date)
    
    # Create news content with improved formatting
    news_content = []
    for item in news_items:
        # Determine sentiment color
        sentiment_color = {
            "Bullish": "green",
            "Somewhat-Bullish": "lightgreen",
            "Neutral": "gray",
            "Somewhat-Bearish": "orange",
            "Bearish": "red"
        }.get(item['sentiment'], "gray")
        
        news_content.append(html.Div([
            html.H4(item['title']),
            html.P([
                f"Source: {item['source']} | ",
                f"Date: {item['date'].strftime('%Y-%m-%d %H:%M:%S')} | ",
                html.Span(
                    f"Sentiment: {item['sentiment']}",
                    style={'color': sentiment_color, 'fontWeight': 'bold'}
                )
            ]),
            html.P(item['summary']),
            html.A("Read More", href=item['url'], target="_blank"),
            html.Hr()
        ], className="news-item"))
    
    if not news_content:
        news_content = [html.P("No news found for this date range.")]
    
    header = f"Market News for {symbol} ({click_date})"
    
    modal_style = {
        "display": "block",
        "position": "fixed",
        "z-index": "1000",
        "left": "0",
        "top": "0",
        "width": "100%",
        "height": "100%",
        "overflow": "auto",
        "backgroundColor": "rgba(0,0,0,0.4)",
    }
    
    return modal_style, header, news_content

# -----------------------------
# Run the App
# -----------------------------
if __name__ == '__main__':
    app.run_server(debug=True)