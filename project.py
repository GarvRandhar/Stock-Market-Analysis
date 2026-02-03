import datetime
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from threading import Timer
import webbrowser
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob


def predict_future_prices_linear(hist, days=5):
    hist = hist[-30:]  
    X = np.arange(len(hist)).reshape(-1, 1)
    y = hist['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(hist), len(hist) + days).reshape(-1, 1)
    future_preds = model.predict(future_X)

    return future_preds


def add_technical_indicators(hist):
    hist = hist.copy()
    
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
    hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
    hist['EMA_12'] = hist['Close'].ewm(span=12, adjust=False).mean()
    
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))
    
    hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
    hist['BB_Std'] = hist['Close'].rolling(window=20).std()
    hist['BB_Upper'] = hist['BB_Middle'] + (hist['BB_Std'] * 2)
    hist['BB_Lower'] = hist['BB_Middle'] - (hist['BB_Std'] * 2)
    
    return hist


def fetch_news_sentiment(ticker_symbol):
    try:
        company_name = ticker_symbol.replace('.NS', '')
        
        stock = yf.Ticker(ticker_symbol)
        news = stock.news[:5]
        
        sentiments = []
        for article in news:
            title = article.get('title', '')
            sentiment = TextBlob(title).sentiment.polarity
            sentiments.append({
                'title': title,
                'link': article.get('link', '#'),
                'sentiment': sentiment,
                'source': article.get('publisher', 'Unknown')
            })
        
        return sentiments
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


def get_sentiment_color(sentiment_score):
    if sentiment_score > 0.1:
        return colors['positive']
    elif sentiment_score < -0.1:
        return colors['negative']
    else:
        return colors['neutral']


def get_sentiment_label(sentiment_score):
    if sentiment_score > 0.1:
        return "Positive"
    elif sentiment_score < -0.1:
        return "Negative"
    else:
        return "Neutral"


external_stylesheets = [
    dbc.themes.DARKLY,  
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "StockVision - Indian Market Analysis"

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
            
            * {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }
            
            body {
                background: linear-gradient(135deg, #0F0F1E 0%, #1A1A2E 100%);
                scroll-behavior: smooth;
            }
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 10px;
                height: 10px;
            }
            
            ::-webkit-scrollbar-track {
                background: rgba(30, 30, 46, 0.3);
                border-radius: 10px;
            }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #6366F1, #8B5CF6);
                border-radius: 10px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #8B5CF6, #6366F1);
            }
            
            /* Card hover effects */
            .card {
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            }
            
            .card:hover {
                transform: translateY(-4px);
                box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.3) !important;
            }
            
            /* Button enhancements */
            .btn {
                transition: all 0.3s ease !important;
                font-weight: 600 !important;
                letter-spacing: 0.5px !important;
                text-transform: uppercase !important;
                font-size: 0.85rem !important;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
                border: none !important;
                box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
            }
            
            .btn-primary:hover {
                background: linear-gradient(135deg, #8B5CF6, #6366F1) !important;
                box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
                transform: translateY(-2px);
            }
            
            .btn-secondary {
                background: linear-gradient(135deg, #10B981, #059669) !important;
                border: none !important;
                box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4) !important;
            }
            
            .btn-secondary:hover {
                background: linear-gradient(135deg, #059669, #10B981) !important;
                box-shadow: 0 6px 20px rgba(16, 185, 129, 0.6) !important;
                transform: translateY(-2px);
            }
            
            /* Input styling */
            .form-control, .form-select {
                background: rgba(30, 30, 46, 0.6) !important;
                border: 1px solid rgba(99, 102, 241, 0.3) !important;
                color: #F8F9FA !important;
                transition: all 0.3s ease !important;
            }
            
            .form-control:focus, .form-select:focus {
                background: rgba(30, 30, 46, 0.8) !important;
                border-color: #6366F1 !important;
                box-shadow: 0 0 0 0.2rem rgba(99, 102, 241, 0.25) !important;
            }
            
            /* Badge styling */
            .badge {
                padding: 0.5em 1em !important;
                font-weight: 600 !important;
                letter-spacing: 0.5px !important;
            }
            
            /* Table styling */
            .table {
                font-size: 0.9rem !important;
            }
            
            .table-hover tbody tr:hover {
                background-color: rgba(99, 102, 241, 0.1) !important;
            }
            
            /* Tab styling */
            .tab {
                background: rgba(30, 30, 46, 0.6) !important;
                border: 1px solid rgba(99, 102, 241, 0.2) !important;
                color: #F8F9FA !important;
                transition: all 0.3s ease !important;
            }
            
            .tab--selected {
                background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
                border-color: #6366F1 !important;
            }
            
            /* Loading animation */
            ._dash-loading {
                opacity: 0.3;
            }
            
            /* Card header styling */
            .card-header {
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2)) !important;
                border-bottom: 1px solid rgba(99, 102, 241, 0.3) !important;
                font-weight: 600 !important;
            }
            
            /* Smooth fade-in animation */
            @keyframes fadeIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .card {
                animation: fadeIn 0.6s ease-out;
            }
            
            /* Radio button styling */
            .form-check-input:checked {
                background-color: #6366F1 !important;
                border-color: #6366F1 !important;
            }
            
            /* Toast notification styling */
            .toast {
                background: rgba(30, 30, 46, 0.95) !important;
                border: 1px solid rgba(99, 102, 241, 0.3) !important;
                backdrop-filter: blur(10px);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

card_style = {
    'boxShadow': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
    'borderRadius': '16px',
    'marginBottom': '24px',
    'backgroundColor': 'rgba(30, 30, 46, 0.8)',
    'backdropFilter': 'blur(10px)',
    'border': '1px solid rgba(99, 102, 241, 0.18)',
    'transition': 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
}

graph_config = {
    'displayModeBar': True,
    'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'resetScale2d'],
    'displaylogo': False,
}

colors = {
    'background': '#0F0F1E',
    'text': '#F8F9FA',
    'primary': '#6366F1',
    'secondary': '#10B981',
    'accent': '#F59E0B',
    'positive': '#10B981',
    'negative': '#EF4444',
    'neutral': '#F59E0B',
    'dark_card': 'rgba(30, 30, 46, 0.7)',
    'card_header': '#1E1E2E',
    'chart_bg': '#1A1A2E',
    'grid_color': 'rgba(99, 102, 241, 0.1)',
    'border_color': 'rgba(99, 102, 241, 0.2)',
    'gradient_start': '#6366F1',
    'gradient_end': '#8B5CF6',
    'success_light': '#34D399',
    'danger_light': '#F87171',
    'info': '#3B82F6',
    'warning': '#FBBF24'
}

popular_stocks = [
    {'label': 'Reliance Industries', 'value': 'RELIANCE.NS'},
    {'label': 'Tata Consultancy Services', 'value': 'TCS.NS'},
    {'label': 'HDFC Bank', 'value': 'HDFCBANK.NS'},
    {'label': 'Infosys', 'value': 'INFY.NS'},
    {'label': 'ICICI Bank', 'value': 'ICICIBANK.NS'},
]

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1([
                    html.I(className="fas fa-chart-line me-3"),
                    "StockVision - Indian Market Analysis"
                ], className="display-4 fw-bold text-center text-light mb-0"),
                html.P("Real-time analysis & prediction of NSE stocks",
                       className="lead text-center text-light opacity-75")
            ], className="pt-4 pb-3")
        ], width=12)
    ], style={
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'borderRadius': '0 0 20px 20px',
        'marginBottom': '30px',
        'boxShadow': '0 10px 40px rgba(102, 126, 234, 0.3)'
    }),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Stock Selection", className="text-center text-light")),
                dbc.CardBody([
                    html.Label("Enter NSE stock ticker:", className="form-label fw-bold"),
                    dbc.Input(
                        id='input-stock',
                        value='RELIANCE.NS',
                        type='text',
                        placeholder='e.g., INFY.NS, TCS.NS',
                        className='mb-3 bg-dark text-light border-secondary'
                    ),
                    html.Label("Popular Stocks:", className="form-label fw-bold"),
                    dbc.RadioItems(
                        id='popular-stocks',
                        options=popular_stocks,
                        value='RELIANCE.NS',
                        className="mb-3",
                        inline=False
                    ),
                    html.Label("Compare with Stock:", className="form-label fw-bold"),
                    dbc.Input(
                        id='compare-stock',
                        value='',
                        type='text',
                        placeholder='e.g., TCS.NS (optional)',
                        className='mb-3 bg-dark text-light border-secondary'
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-search me-2"), "Analyze"],
                        id="analyze-button",
                        color="primary",
                        className="w-100 mb-2"
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-sync-alt me-2"), "Refresh Data"],
                        id="refresh-button",
                        color="secondary",
                        className="w-100",
                        n_clicks=0
                    ),
                ])
            ], style=card_style),

            dbc.Card([
                dbc.CardHeader(html.H5("Current Stock Price", className="text-center")),
                dbc.CardBody([
                    html.Div(id='stock-details', className='text-center')
                ])
            ], style=card_style),

            dbc.Card([
                dbc.CardHeader(html.H5([
                    html.I(className="fas fa-robot me-2"),
                    "AI Price Predictions"
                ], className="text-center")),
                dbc.CardBody([
                    html.Div(id='financials-data')
                ])
            ], style=card_style)
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-chart-line me-2"),
                            "Price History with Technical Indicators"
                        ], className="mb-0 d-inline"),
                        dbc.ButtonGroup([
                            dbc.Button("1W", id="1w-button", color="primary", outline=True, size="sm",
                                       className="mx-1"),
                            dbc.Button("1M", id="1m-button", color="primary", size="sm", className="mx-1"),
                            dbc.Button("3M", id="3m-button", color="primary", outline=True, size="sm",
                                       className="mx-1"),
                            dbc.Button("1Y", id="1y-button", color="primary", outline=True, size="sm",
                                       className="mx-1"),
                        ], className="float-end")
                    ], className="d-flex justify-content-between align-items-center")
                ]),
                dbc.CardBody([
                    dcc.Tabs(id="chart-tabs", value="price", children=[
                        dcc.Tab(label="Price & MA", value="price", children=[
                            dcc.Loading(
                                id="loading-1",
                                type="circle",
                                children=html.Div(id='output-graphs')
                            )
                        ]),
                        dcc.Tab(label="RSI", value="rsi", children=[
                            dcc.Loading(
                                type="circle",
                                children=html.Div(id='rsi-graph')
                            )
                        ]),
                        dcc.Tab(label="Bollinger Bands", value="bb", children=[
                            dcc.Loading(
                                type="circle",
                                children=html.Div(id='bb-graph')
                            )
                        ]),
                    ])
                ])
            ], style=card_style),

            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-clock me-2"),
                            "Intraday Performance"
                        ], className="mb-0 d-inline"),
                        html.Span("5-minute intervals", className="text-muted float-end")
                    ], className="d-flex justify-content-between align-items-center")
                ]),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-2",
                        type="circle",
                        children=html.Div(id='intraday-graph')
                    )
                ])
            ], style=card_style),

            dbc.Card([
                dbc.CardHeader(html.H5([
                    html.I(className="fas fa-exchange-alt me-2"),
                    "Stock Comparison"
                ], className="text-center")),
                dbc.CardBody([
                    dcc.Loading(
                        type="circle",
                        children=html.Div(id='comparison-graph')
                    )
                ])
            ], style=card_style)
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5([
                    html.I(className="fas fa-calendar-day me-2"),
                    "Today's Snapshot"
                ], className="text-center")),
                dbc.CardBody([
                    html.Div(id='daily-data-box')
                ])
            ], style=card_style),

            dbc.Card([
                dbc.CardHeader(html.H5([
                    html.I(className="fas fa-building me-2"),
                    "Company Fundamentals"
                ], className="text-center")),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-3",
                        type="circle",
                        children=html.Div(id='fundamentals-data')
                    )
                ])
            ], style=card_style),

            dbc.Card([
                dbc.CardHeader(html.H5([
                    html.I(className="fas fa-file-invoice-dollar me-2"),
                    "Quarterly Financials"
                ], className="text-center")),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-4",
                        type="circle",
                        children=html.Div(id='quarterly-financials-data')
                    )
                ])
            ], style=card_style),

            dbc.Card([
                dbc.CardHeader(html.H5([
                    html.I(className="fas fa-chart-line me-2"),
                    "Comprehensive Financial Details"
                ], className="text-center")),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-5",
                        type="circle",
                        children=html.Div(id='comprehensive-financials-data')
                    )
                ])
            ], style=card_style)
        ], width=3)
    ]),

    dbc.Row([
        dbc.Col([
            html.Footer([
                html.P([
                    "StockVision © 2025 | Data provided by Yahoo Finance | ",
                    html.Small("Last updated: ", id="last-updated", className="text-muted")
                ], className="text-center text-muted")
            ], className="py-3 mt-4")
        ], width=12)
    ]),

    dcc.Interval(
        id='interval-component',
        interval=5 * 60 * 1000,
        n_intervals=0
    ),
    dcc.Store(id='time-period-store', data='1m'), 
    dcc.Store(id='last-update-time', data=''),

    dbc.Toast(
        id="update-toast",
        header="Data Updated",
        is_open=False,
        dismissable=True,
        icon="success",
        duration=3000,
        style={"position": "fixed", "top": 10, "right": 10, "width": 300, "zIndex": 1999}
    )
], fluid=True, style={'backgroundColor': colors['background'], 'minHeight': '100vh'})


@app.callback(
    Output('input-stock', 'value'),
    Input('popular-stocks', 'value')
)
def update_stock_input(selected_stock):
    return selected_stock


@app.callback(
    [Output('time-period-store', 'data'),
     Output('1w-button', 'outline'),
     Output('1m-button', 'outline'),
     Output('3m-button', 'outline'),
     Output('1y-button', 'outline')],
    [Input('1w-button', 'n_clicks'),
     Input('1m-button', 'n_clicks'),
     Input('3m-button', 'n_clicks'),
     Input('1y-button', 'n_clicks')],
    [State('time-period-store', 'data')]
)
def update_time_period(n1, n2, n3, n4, current_period):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "1mo", True, False, True, True

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == "1w-button":
        return "1wk", False, True, True, True
    elif button_id == "1m-button":
        return "1mo", True, False, True, True
    elif button_id == "3m-button":
        return "3mo", True, True, False, True
    elif button_id == "1y-button":
        return "1y", True, True, True, False

    return current_period, True, False, True, True


@app.callback(
    [Output('stock-details', 'children'),
     Output('fundamentals-data', 'children'),
     Output('quarterly-financials-data', 'children'),
     Output('daily-data-box', 'children'),
     Output('output-graphs', 'children'),
     Output('rsi-graph', 'children'),
     Output('bb-graph', 'children'),
     Output('intraday-graph', 'children'),
     Output('comparison-graph', 'children'),
     Output('comprehensive-financials-data', 'children'),
     Output('financials-data', 'children'),
     Output('last-updated', 'children'),
     Output('last-update-time', 'data'),
     Output('update-toast', 'is_open'),
     Output('update-toast', 'children')],
    [Input('analyze-button', 'n_clicks'),
     Input('refresh-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('input-stock', 'value'),
     State('compare-stock', 'value'),
     State('time-period-store', 'data'),
     State('last-update-time', 'data')]
)
def update_stock_info(analyze_clicks, refresh_clicks, n_intervals, stock_ticker, compare_ticker, time_period, last_update):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    refresh_message = "Data refreshed automatically"
    if button_id == 'refresh-button':
        refresh_message = "Data refreshed manually"
    elif button_id == 'analyze-button':
        refresh_message = f"Loaded data for {stock_ticker}"

    now = datetime.datetime.now()
    timestamp = now.strftime("%H:%M:%S, %d %b %Y")

    show_toast = button_id in ['refresh-button', 'analyze-button']

    try:
        if not stock_ticker:
            stock_ticker = "RELIANCE.NS"

        stock_ticker = stock_ticker.upper().strip()

        stock = yf.Ticker(stock_ticker)
        if time_period == "1wk":
            hist_period = "5d"
        elif time_period == "1mo":
            hist_period = "1mo"
        elif time_period == "3mo":
            hist_period = "3mo"
        elif time_period == "1y":
            hist_period = "1y"
        else:
            hist_period = "1mo"  

        hist = stock.history(period=hist_period, interval="1d")
        intraday = stock.history(period="1d", interval="5m")
        
        hist_with_indicators = add_technical_indicators(hist)

        if hist.empty or intraday.empty:
            raise ValueError(f"No data found for {stock_ticker}. Please check if the symbol is correct.")

        info = stock.info
        company_name = info.get('shortName', stock_ticker)

        fundamentals_data = html.Div([
            html.Table([
                html.Tr([
                    html.Td(html.I(className="fas fa-chart-pie text-primary"), style={'width': '40px'}),
                    html.Td("P/E Ratio"),
                    html.Td(f"{info.get('trailingPE', 'N/A'):.2f}" if isinstance(info.get('trailingPE'),
                                                                                 (int, float)) else "N/A",
                            className="fw-bold text-end")
                ]),
                html.Tr([
                    html.Td(html.I(className="fas fa-dollar-sign text-success")),
                    html.Td("Market Cap"),
                    html.Td(f"₹{info.get('marketCap') / 1e9:.2f}B" if info.get('marketCap') else "N/A",
                            className="fw-bold text-end")
                ]),
                html.Tr([
                    html.Td(html.I(className="fas fa-coins text-warning")),
                    html.Td("EPS"),
                    html.Td(f"₹{info.get('trailingEps', 'N/A')}" if info.get('trailingEps') else "N/A",
                            className="fw-bold text-end")
                ]),
                html.Tr([
                    html.Td(html.I(className="fas fa-percentage text-info")),
                    html.Td("Dividend Yield"),
                    html.Td(f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else "N/A",
                            className="fw-bold text-end")
                ]),
                html.Tr([
                    html.Td(html.I(className="fas fa-industry text-secondary")),
                    html.Td("Sector"),
                    html.Td(info.get('sector', 'N/A'), className="fw-bold text-end")
                ]),
                html.Tr([
                    html.Td(html.I(className="fas fa-building text-light")),
                    html.Td("Industry"),
                    html.Td(info.get('industry', 'N/A'), className="fw-bold text-end")
                ]),
            ], className='table table-hover text-light')
        ])

        quarterly = stock.quarterly_financials
        if quarterly.empty:
            quarterly_financials_data = html.Div([
                html.P("No financial data available.", className='text-danger')
            ])
        else:
            latest_quarter = quarterly.iloc[:, 0]
            quarterly_financials_data = html.Div([
                html.Table([
                    html.Tr([
                        html.Td(html.I(className="fas fa-money-bill-wave text-success"), style={'width': '40px'}),
                        html.Td("Total Revenue"),
                        html.Td(f"₹{latest_quarter.get('Total Revenue', 'N/A') / 1e7:.2f} Cr" if pd.notna(
                            latest_quarter.get('Total Revenue')) else "N/A",
                                className="fw-bold text-end")
                    ]),
                    html.Tr([
                        html.Td(html.I(className="fas fa-hand-holding-usd text-info")),
                        html.Td("Gross Profit"),
                        html.Td(f"₹{latest_quarter.get('Gross Profit', 'N/A') / 1e7:.2f} Cr" if pd.notna(
                            latest_quarter.get('Gross Profit')) else "N/A",
                                className="fw-bold text-end")
                    ]),
                    html.Tr([
                        html.Td(html.I(className="fas fa-wallet text-primary")),
                        html.Td("Net Income"),
                        html.Td(f"₹{latest_quarter.get('Net Income', 'N/A') / 1e7:.2f} Cr" if pd.notna(
                            latest_quarter.get('Net Income')) else "N/A",
                                className="fw-bold text-end")
                    ]),
                    html.Tr([
                        html.Td(html.I(className="fas fa-chart-bar text-warning")),
                        html.Td("EBITDA"),
                        html.Td(f"₹{latest_quarter.get('EBITDA', 'N/A') / 1e7:.2f} Cr" if pd.notna(
                            latest_quarter.get('EBITDA')) else "N/A",
                                className="fw-bold text-end")
                    ]),
                    html.Tr([
                        html.Td(html.I(className="fas fa-cogs text-secondary")),
                        html.Td("Operating Income"),
                        html.Td(f"₹{latest_quarter.get('Operating Income', 'N/A') / 1e7:.2f} Cr" if pd.notna(
                            latest_quarter.get('Operating Income')) else "N/A",
                                className="fw-bold text-end")
                    ]),
                ], className='table table-hover text-light')
            ])

        latest = hist.iloc[-1]
        open_price = latest['Open']
        close_price = latest['Close']
        high_price = latest['High']
        low_price = latest['Low']
        volume = latest['Volume']

        percent_change = ((close_price - open_price) / open_price) * 100
        price_direction = "up" if percent_change > 0 else "down"
        color = colors['positive'] if percent_change > 0 else colors['negative']
        icon = "fa-arrow-up" if percent_change > 0 else "fa-arrow-down"

        stock_details = html.Div([
            html.Div([
                html.H4(f"{company_name}", className='text-center mb-1 fw-bold', 
                       style={'color': colors['text'], 'letterSpacing': '0.5px'}),
                html.H6(f"({stock_ticker})", className='text-center mb-3', 
                       style={'color': colors['neutral'], 'fontWeight': '500'}),
            ]),
            html.Div([
                html.H1([
                    f"₹{close_price:.2f}"
                ], className='text-center mb-2 fw-bold', 
                   style={'fontSize': '2.5rem', 'color': colors['primary'], 'letterSpacing': '-1px'}),
                html.Div([
                    html.I(className=f"fas {icon} me-2", style={'fontSize': '1.2rem'}),
                    html.Span(f"{abs(percent_change):.2f}%", className='fw-bold', 
                             style={'fontSize': '1.3rem'})
                ], className='text-center mb-3', 
                   style={
                       'color': color,
                       'padding': '8px 20px',
                       'borderRadius': '12px',
                       'background': f'linear-gradient(135deg, {color}22, {color}11)',
                       'display': 'inline-block',
                       'border': f'1px solid {color}44'
                   })
            ]),
            html.Hr(style={'borderColor': colors['border_color'], 'margin': '20px 0'}),
            html.Div([
                dbc.Badge([
                    html.I(className="fas fa-chart-bar me-2"),
                    f"Vol: {volume:,.0f}"
                ], color="info", className="me-2 px-3 py-2", 
                   style={'fontSize': '0.85rem', 'fontWeight': '600'}),
                dbc.Badge([
                    html.I(className="fas fa-building me-2"),
                    "NSE"
                ], color="primary", className="px-3 py-2",
                   style={'fontSize': '0.85rem', 'fontWeight': '600'})
            ], className="text-center")
        ], style={'padding': '10px'})

        daily_data_box = html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-door-open", 
                              style={'color': colors['info'], 'fontSize': '1.2rem', 'marginBottom': '8px'}),
                        html.Span("Open", className="d-block text-muted", 
                                 style={'fontSize': '0.75rem', 'fontWeight': '600', 'letterSpacing': '0.5px'}),
                        html.H4(f"₹{open_price:.2f}", className="mb-0 fw-bold", 
                               style={'color': colors['text']})
                    ], className="text-center p-3", 
                       style={
                           'background': 'rgba(59, 130, 246, 0.1)',
                           'borderRadius': '12px',
                           'border': '1px solid rgba(59, 130, 246, 0.2)'
                       })
                ], className="col-6 mb-3"),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-door-closed", 
                              style={'color': colors['primary'], 'fontSize': '1.2rem', 'marginBottom': '8px'}),
                        html.Span("Close", className="d-block text-muted", 
                                 style={'fontSize': '0.75rem', 'fontWeight': '600', 'letterSpacing': '0.5px'}),
                        html.H4(f"₹{close_price:.2f}", className="mb-0 fw-bold", 
                               style={'color': colors['text']})
                    ], className="text-center p-3", 
                       style={
                           'background': 'rgba(99, 102, 241, 0.1)',
                           'borderRadius': '12px',
                           'border': '1px solid rgba(99, 102, 241, 0.2)'
                       })
                ], className="col-6 mb-3"),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-arrow-up", 
                              style={'color': colors['positive'], 'fontSize': '1.2rem', 'marginBottom': '8px'}),
                        html.Span("High", className="d-block text-muted", 
                                 style={'fontSize': '0.75rem', 'fontWeight': '600', 'letterSpacing': '0.5px'}),
                        html.H4(f"₹{high_price:.2f}", className="mb-0 fw-bold", 
                               style={'color': colors['positive']})
                    ], className="text-center p-3", 
                       style={
                           'background': 'rgba(16, 185, 129, 0.1)',
                           'borderRadius': '12px',
                           'border': '1px solid rgba(16, 185, 129, 0.2)'
                       })
                ], className="col-6 mb-3"),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-arrow-down", 
                              style={'color': colors['negative'], 'fontSize': '1.2rem', 'marginBottom': '8px'}),
                        html.Span("Low", className="d-block text-muted", 
                                 style={'fontSize': '0.75rem', 'fontWeight': '600', 'letterSpacing': '0.5px'}),
                        html.H4(f"₹{low_price:.2f}", className="mb-0 fw-bold", 
                               style={'color': colors['negative']})
                    ], className="text-center p-3", 
                       style={
                           'background': 'rgba(239, 68, 68, 0.1)',
                           'borderRadius': '12px',
                           'border': '1px solid rgba(239, 68, 68, 0.2)'
                       })
                ], className="col-6 mb-3"),
            ], className="row"),
            html.Div([
                html.Div([
                    html.Span("Today's Change", className="d-block text-center mb-2", 
                             style={'fontSize': '0.8rem', 'fontWeight': '600', 'color': colors['text'], 'opacity': '0.7'}),
                    html.Div([
                        html.I(className=f"fas {icon} me-2"),
                        f"{price_direction.upper()} by {abs(percent_change):.2f}%"
                    ], className="text-center fw-bold py-2", 
                       style={
                           'color': color,
                           'fontSize': '1.1rem',
                           'background': f'linear-gradient(135deg, {color}22, {color}11)',
                           'borderRadius': '10px',
                           'border': f'1px solid {color}44'
                       })
                ], style={'marginTop': '10px'})
            ])
        ])

        future_predictions = predict_future_prices_linear(hist)
        today = datetime.datetime.now().date()
        future_dates = [(today + datetime.timedelta(days=i + 1)).strftime("%d %b") for i in
                        range(len(future_predictions))]

        trends = []
        for i in range(len(future_predictions)):
            if i == 0:
                prev_price = close_price
            else:
                prev_price = future_predictions[i - 1]

            current_price = future_predictions[i]
            trend_icon = "fa-arrow-up text-success" if current_price > prev_price else "fa-arrow-down text-danger"
            trends.append(trend_icon)

        financials_data = html.Div([
            html.P("Next 5 Trading Days", className="text-center text-muted mb-3"),
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Date"),
                        html.Th("Price", className="text-end"),
                        html.Th("Trend", className="text-center", style={"width": "40px"})
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(date),
                        html.Td(f"₹{price:.2f}", className="text-end fw-bold"),
                        html.Td(html.I(className=f"fas {icon} text-center"), className="text-center")
                    ]) for date, price, icon in zip(future_dates, future_predictions, trends)
                ])
            ], className='table table-sm table-hover text-light')
        ])

        common_layout = {
            'margin': {'l': 50, 'r': 50, 't': 60, 'b': 50},
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'center',
                'x': 0.5,
                'bgcolor': 'rgba(0,0,0,0.3)',
                'bordercolor': colors['border_color'],
                'borderwidth': 1,
                'font': {'size': 11, 'color': colors['text']}
            },
            'font': {'family': 'Inter, system-ui, sans-serif', 'color': colors['text'], 'size': 12},
            'plot_bgcolor': colors['chart_bg'],
            'paper_bgcolor': colors['chart_bg'],
            'hovermode': 'x unified',
            'hoverlabel': {
                'bgcolor': 'rgba(30, 30, 46, 0.95)',
                'font_size': 13,
                'font_family': 'Inter, system-ui, sans-serif',
                'bordercolor': colors['primary']
            }
        }

        from plotly.subplots import make_subplots
        
        price_fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=('', '')
        )
        
        price_fig.add_trace(
            go.Scatter(
                x=hist_with_indicators.index,
                y=hist_with_indicators['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color=colors['primary'], width=3),
                fill='tozeroy',
                fillcolor=f'rgba(99, 102, 241, 0.1)',
                hovertemplate='<b>Date</b>: %{x|%d %b %Y}<br><b>Price</b>: ₹%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        price_fig.add_trace(
            go.Scatter(
                x=hist_with_indicators.index,
                y=hist_with_indicators['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color=colors['success_light'], width=2, dash='dash'),
                hovertemplate='<b>SMA 20</b>: ₹%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        price_fig.add_trace(
            go.Scatter(
                x=hist_with_indicators.index,
                y=hist_with_indicators['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color=colors['accent'], width=2, dash='dot'),
                hovertemplate='<b>SMA 50</b>: ₹%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        volume_colors = [colors['positive'] if hist_with_indicators['Close'].iloc[i] >= hist_with_indicators['Open'].iloc[i] 
                        else colors['negative'] for i in range(len(hist_with_indicators))]
        
        price_fig.add_trace(
            go.Bar(
                x=hist_with_indicators.index,
                y=hist_with_indicators['Volume'],
                name='Volume',
                marker=dict(color=volume_colors, opacity=0.5),
                hovertemplate='<b>Volume</b>: %{y:,.0f}<extra></extra>',
                showlegend=True
            ),
            row=2, col=1
        )
        
        price_fig.update_layout(
            title={
                'text': f"{time_period.upper()} Price History with Moving Averages",
                'font': {'size': 18, 'color': colors['text'], 'family': 'Inter, sans-serif'},
                'x': 0.5,
                'xanchor': 'center'
            },
            height=500,
            showlegend=True,
            **common_layout
        )
        
        price_fig.update_xaxes(
            title_text='Date',
            showgrid=True,
            gridcolor=colors['grid_color'],
            gridwidth=1,
            row=2, col=1
        )
        
        price_fig.update_xaxes(
            showgrid=True,
            gridcolor=colors['grid_color'],
            gridwidth=1,
            row=1, col=1
        )
        
        price_fig.update_yaxes(
            title_text='Price (₹)',
            showgrid=True,
            gridcolor=colors['grid_color'],
            gridwidth=1,
            row=1, col=1
        )
        
        price_fig.update_yaxes(
            title_text='Volume',
            showgrid=False,
            row=2, col=1
        )
        
        closing_price_graph = dcc.Graph(figure=price_fig, config=graph_config)

        rsi_fig = go.Figure()
        
        rsi_fig.add_hrect(
            y0=70, y1=100,
            fillcolor=colors['negative'],
            opacity=0.15,
            layer="below",
            line_width=0,
            annotation_text="Overbought",
            annotation_position="top right"
        )
        
        rsi_fig.add_hrect(
            y0=0, y1=30,
            fillcolor=colors['positive'],
            opacity=0.15,
            layer="below",
            line_width=0,
            annotation_text="Oversold",
            annotation_position="bottom right"
        )
        
        rsi_fig.add_trace(
            go.Scatter(
                x=hist_with_indicators.index,
                y=hist_with_indicators['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color=colors['info'], width=3),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.1)',
                hovertemplate='<b>Date</b>: %{x|%d %b}<br><b>RSI</b>: %{y:.2f}<extra></extra>'
            )
        )
        
        rsi_fig.add_hline(
            y=70,
            line_dash="dash",
            line_color=colors['danger_light'],
            line_width=2,
            opacity=0.7
        )
        rsi_fig.add_hline(
            y=50,
            line_dash="dot",
            line_color=colors['neutral'],
            line_width=1,
            opacity=0.5
        )
        rsi_fig.add_hline(
            y=30,
            line_dash="dash",
            line_color=colors['success_light'],
            line_width=2,
            opacity=0.7
        )
        
        rsi_fig.update_layout(
            title={
                'text': 'Relative Strength Index (RSI)',
                'font': {'size': 18, 'color': colors['text'], 'family': 'Inter, sans-serif'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis={
                'title': 'Date',
                'showgrid': True,
                'gridcolor': colors['grid_color'],
                'gridwidth': 1
            },
            yaxis={
                'title': 'RSI',
                'showgrid': True,
                'gridcolor': colors['grid_color'],
                'gridwidth': 1,
                'range': [0, 100]
            },
            height=400,
            **common_layout
        )
        rsi_graph = dcc.Graph(figure=rsi_fig, config=graph_config)

        bb_fig = go.Figure()
        
        bb_fig.add_trace(
            go.Scatter(
                x=hist_with_indicators.index,
                y=hist_with_indicators['BB_Upper'],
                mode='lines',
                name='Upper Band',
                line=dict(color=colors['danger_light'], width=2, dash='dash'),
                hovertemplate='<b>Upper Band</b>: ₹%{y:.2f}<extra></extra>',
                showlegend=True
            )
        )
        
        bb_fig.add_trace(
            go.Scatter(
                x=hist_with_indicators.index,
                y=hist_with_indicators['BB_Lower'],
                mode='lines',
                name='Lower Band',
                line=dict(color=colors['success_light'], width=2, dash='dash'),
                hovertemplate='<b>Lower Band</b>: ₹%{y:.2f}<extra></extra>',
                fill='tonexty',
                fillcolor='rgba(99, 102, 241, 0.08)',
                showlegend=True
            )
        )
        
        bb_fig.add_trace(
            go.Scatter(
                x=hist_with_indicators.index,
                y=hist_with_indicators['BB_Middle'],
                mode='lines',
                name='Middle Band (SMA 20)',
                line=dict(color=colors['warning'], width=2, dash='dot'),
                hovertemplate='<b>Middle Band</b>: ₹%{y:.2f}<extra></extra>'
            )
        )
        
        bb_fig.add_trace(
            go.Scatter(
                x=hist_with_indicators.index,
                y=hist_with_indicators['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color=colors['primary'], width=3),
                hovertemplate='<b>Date</b>: %{x|%d %b}<br><b>Price</b>: ₹%{y:.2f}<extra></extra>'
            )
        )
        
        bb_fig.update_layout(
            title={
                'text': 'Bollinger Bands (20-period, 2σ)',
                'font': {'size': 18, 'color': colors['text'], 'family': 'Inter, sans-serif'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis={
                'title': 'Date',
                'showgrid': True,
                'gridcolor': colors['grid_color'],
                'gridwidth': 1
            },
            yaxis={
                'title': 'Price (₹)',
                'showgrid': True,
                'gridcolor': colors['grid_color'],
                'gridwidth': 1
            },
            height=400,
            **common_layout
        )
        bb_graph = dcc.Graph(figure=bb_fig, config=graph_config)

        intraday_fig = go.Figure()

        if len(intraday) > 10:
            intraday_fig.add_trace(
                go.Candlestick(
                    x=intraday.index,
                    open=intraday['Open'],
                    high=intraday['High'],
                    low=intraday['Low'],
                    close=intraday['Close'],
                    increasing_line_color=colors['positive'],
                    decreasing_line_color=colors['negative'],
                    increasing_fillcolor=colors['positive'],
                    decreasing_fillcolor=colors['negative'],
                    name='Price',
                    hovertemplate='<b>Time</b>: %{x|%H:%M}<br>' +
                                 '<b>Open</b>: ₹%{open:.2f}<br>' +
                                 '<b>High</b>: ₹%{high:.2f}<br>' +
                                 '<b>Low</b>: ₹%{low:.2f}<br>' +
                                 '<b>Close</b>: ₹%{close:.2f}<extra></extra>'
                )
            )
        else:
            intraday_fig.add_trace(
                go.Scatter(
                    x=intraday.index,
                    y=intraday['Close'],
                    mode='lines+markers',
                    name='Intraday Price',
                    line=dict(color=colors['primary'], width=3),
                    marker=dict(size=8, color=colors['primary'], line=dict(color=colors['text'], width=1)),
                    fill='tozeroy',
                    fillcolor='rgba(99, 102, 241, 0.1)',
                    hovertemplate='<b>Time</b>: %{x|%H:%M}<br><b>Price</b>: ₹%{y:.2f}<extra></extra>'
                )
            )

        intraday_fig.update_layout(
            title={
                'text': "Today's Intraday Performance (5-min intervals)",
                'font': {'size': 18, 'color': colors['text'], 'family': 'Inter, sans-serif'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis={
                'title': 'Time',
                'showgrid': True,
                'gridcolor': colors['grid_color'],
                'gridwidth': 1
            },
            yaxis={
                'title': 'Price (₹)',
                'showgrid': True,
                'gridcolor': colors['grid_color'],
                'gridwidth': 1
            },
            height=400,
            **common_layout
        )
        intraday_graph = dcc.Graph(figure=intraday_fig, config=graph_config)

        comparison_graph = html.Div()
        if compare_ticker:
            try:
                compare_ticker = compare_ticker.upper().strip()
                compare_stock = yf.Ticker(compare_ticker)
                compare_hist = compare_stock.history(period=hist_period, interval="1d")
                
                if not compare_hist.empty:
                    hist_norm = ((hist['Close'] / hist['Close'].iloc[0]) - 1) * 100
                    compare_norm = ((compare_hist['Close'] / compare_hist['Close'].iloc[0]) - 1) * 100
                    
                    comp_fig = go.Figure()
                    comp_fig.add_trace(
                        go.Scatter(
                            x=hist_norm.index,
                            y=hist_norm.values,
                            mode='lines',
                            name=stock_ticker,
                            line=dict(color=colors['primary'], width=2),
                            hovertemplate='<b>%{fullData.name}</b><br>Change: %{y:.2f}%<extra></extra>'
                        )
                    )
                    comp_fig.add_trace(
                        go.Scatter(
                            x=compare_norm.index,
                            y=compare_norm.values,
                            mode='lines',
                            name=compare_ticker,
                            line=dict(color=colors['secondary'], width=2),
                            hovertemplate='<b>%{fullData.name}</b><br>Change: %{y:.2f}%<extra></extra>'
                        )
                    )
                    
                    comp_fig.update_layout(
                        title={
                            'text': f"Stock Comparison: {stock_ticker} vs {compare_ticker}",
                            'font': {'size': 18, 'color': colors['text'], 'family': 'Inter, sans-serif'},
                            'x': 0.5,
                            'xanchor': 'center'
                        },
                        xaxis={
                            'title': 'Date',
                            'showgrid': True,
                            'gridcolor': colors['grid_color'],
                            'gridwidth': 1
                        },
                        yaxis={
                            'title': 'Change (%)',
                            'showgrid': True,
                            'gridcolor': colors['grid_color'],
                            'gridwidth': 1,
                            'zeroline': True,
                            'zerolinecolor': colors['border_color'],
                            'zerolinewidth': 2
                        },
                        height=350,
                        **common_layout
                    )
                    
                    comparison_graph = dcc.Graph(figure=comp_fig, config=graph_config)
            except Exception as e:
                comparison_graph = html.Div(f"Could not load comparison stock: {str(e)}", className="text-warning")
        else:
            comparison_graph = html.Div("Enter a stock ticker above to compare", className="text-muted text-center p-4")

        try:
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            income_stmt = stock.income_stmt
            
            comprehensive_financials = []
            
            if not balance_sheet.empty:
                latest_bs = balance_sheet.iloc[:, 0]
                comprehensive_financials.append(html.Div([
                    html.H6([
                        html.I(className="fas fa-balance-scale me-2", style={'color': colors['primary']}),
                        "Balance Sheet"
                    ], className="fw-bold mb-3", style={'color': colors['text']}),
                    html.Table([
                        html.Tr([
                            html.Td(html.I(className="fas fa-coins text-warning"), style={'width': '40px'}),
                            html.Td("Total Assets"),
                            html.Td(f"₹{latest_bs.get('Total Assets', 0) / 1e9:.2f}B" if pd.notna(latest_bs.get('Total Assets')) else "N/A",
                                   className="fw-bold text-end")
                        ]),
                        html.Tr([
                            html.Td(html.I(className="fas fa-file-invoice text-danger")),
                            html.Td("Total Liabilities"),
                            html.Td(f"₹{latest_bs.get('Total Liabilities Net Minority Interest', 0) / 1e9:.2f}B" if pd.notna(latest_bs.get('Total Liabilities Net Minority Interest')) else "N/A",
                                   className="fw-bold text-end")
                        ]),
                        html.Tr([
                            html.Td(html.I(className="fas fa-chart-pie text-success")),
                            html.Td("Stockholders Equity"),
                            html.Td(f"₹{latest_bs.get('Stockholders Equity', 0) / 1e9:.2f}B" if pd.notna(latest_bs.get('Stockholders Equity')) else "N/A",
                                   className="fw-bold text-end")
                        ]),
                        html.Tr([
                            html.Td(html.I(className="fas fa-money-bill-wave text-info")),
                            html.Td("Cash & Equivalents"),
                            html.Td(f"₹{latest_bs.get('Cash And Cash Equivalents', 0) / 1e9:.2f}B" if pd.notna(latest_bs.get('Cash And Cash Equivalents')) else "N/A",
                                   className="fw-bold text-end")
                        ]),
                    ], className='table table-sm table-hover text-light')
                ], className="mb-4"))
            
            if not cash_flow.empty:
                latest_cf = cash_flow.iloc[:, 0]
                comprehensive_financials.append(html.Div([
                    html.H6([
                        html.I(className="fas fa-exchange-alt me-2", style={'color': colors['secondary']}),
                        "Cash Flow"
                    ], className="fw-bold mb-3", style={'color': colors['text']}),
                    html.Table([
                        html.Tr([
                            html.Td(html.I(className="fas fa-arrow-circle-down text-success"), style={'width': '40px'}),
                            html.Td("Operating Cash Flow"),
                            html.Td(f"₹{latest_cf.get('Operating Cash Flow', 0) / 1e9:.2f}B" if pd.notna(latest_cf.get('Operating Cash Flow')) else "N/A",
                                   className="fw-bold text-end")
                        ]),
                        html.Tr([
                            html.Td(html.I(className="fas fa-tools text-primary")),
                            html.Td("Investing Cash Flow"),
                            html.Td(f"₹{latest_cf.get('Investing Cash Flow', 0) / 1e9:.2f}B" if pd.notna(latest_cf.get('Investing Cash Flow')) else "N/A",
                                   className="fw-bold text-end")
                        ]),
                        html.Tr([
                            html.Td(html.I(className="fas fa-hand-holding-usd text-warning")),
                            html.Td("Financing Cash Flow"),
                            html.Td(f"₹{latest_cf.get('Financing Cash Flow', 0) / 1e9:.2f}B" if pd.notna(latest_cf.get('Financing Cash Flow')) else "N/A",
                                   className="fw-bold text-end")
                        ]),
                        html.Tr([
                            html.Td(html.I(className="fas fa-chart-line text-info")),
                            html.Td("Free Cash Flow"),
                            html.Td(f"₹{latest_cf.get('Free Cash Flow', 0) / 1e9:.2f}B" if pd.notna(latest_cf.get('Free Cash Flow')) else "N/A",
                                   className="fw-bold text-end")
                        ]),
                    ], className='table table-sm table-hover text-light')
                ], className="mb-4"))
            
            if not income_stmt.empty:
                latest_is = income_stmt.iloc[:, 0]
                comprehensive_financials.append(html.Div([
                    html.H6([
                        html.I(className="fas fa-file-invoice-dollar me-2", style={'color': colors['accent']}),
                        "Income Statement"
                    ], className="fw-bold mb-3", style={'color': colors['text']}),
                    html.Table([
                        html.Tr([
                            html.Td(html.I(className="fas fa-dollar-sign text-success"), style={'width': '40px'}),
                            html.Td("Total Revenue"),
                            html.Td(f"₹{latest_is.get('Total Revenue', 0) / 1e9:.2f}B" if pd.notna(latest_is.get('Total Revenue')) else "N/A",
                                   className="fw-bold text-end")
                        ]),
                        html.Tr([
                            html.Td(html.I(className="fas fa-receipt text-danger")),
                            html.Td("Cost of Revenue"),
                            html.Td(f"₹{latest_is.get('Cost Of Revenue', 0) / 1e9:.2f}B" if pd.notna(latest_is.get('Cost Of Revenue')) else "N/A",
                                   className="fw-bold text-end")
                        ]),
                        html.Tr([
                            html.Td(html.I(className="fas fa-chart-bar text-info")),
                            html.Td("Gross Profit"),
                            html.Td(f"₹{latest_is.get('Gross Profit', 0) / 1e9:.2f}B" if pd.notna(latest_is.get('Gross Profit')) else "N/A",
                                   className="fw-bold text-end")
                        ]),
                        html.Tr([
                            html.Td(html.I(className="fas fa-wallet text-primary")),
                            html.Td("Operating Income"),
                            html.Td(f"₹{latest_is.get('Operating Income', 0) / 1e9:.2f}B" if pd.notna(latest_is.get('Operating Income')) else "N/A",
                                   className="fw-bold text-end")
                        ]),
                        html.Tr([
                            html.Td(html.I(className="fas fa-money-check-alt text-success")),
                            html.Td("Net Income"),
                            html.Td(f"₹{latest_is.get('Net Income', 0) / 1e9:.2f}B" if pd.notna(latest_is.get('Net Income')) else "N/A",
                                   className="fw-bold text-end")
                        ]),
                    ], className='table table-sm table-hover text-light')
                ]))
            
            comprehensive_financials.append(html.Div([
                html.H6([
                    html.I(className="fas fa-calculator me-2", style={'color': colors['info']}),
                    "Key Financial Ratios"
                ], className="fw-bold mb-3 mt-4", style={'color': colors['text']}),
                html.Table([
                    html.Tr([
                        html.Td(html.I(className="fas fa-percentage text-primary"), style={'width': '40px'}),
                        html.Td("Profit Margin"),
                        html.Td(f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get('profitMargins') else "N/A",
                               className="fw-bold text-end")
                    ]),
                    html.Tr([
                        html.Td(html.I(className="fas fa-chart-area text-success")),
                        html.Td("Operating Margin"),
                        html.Td(f"{info.get('operatingMargins', 0) * 100:.2f}%" if info.get('operatingMargins') else "N/A",
                               className="fw-bold text-end")
                    ]),
                    html.Tr([
                        html.Td(html.I(className="fas fa-coins text-warning")),
                        html.Td("Return on Assets"),
                        html.Td(f"{info.get('returnOnAssets', 0) * 100:.2f}%" if info.get('returnOnAssets') else "N/A",
                               className="fw-bold text-end")
                    ]),
                    html.Tr([
                        html.Td(html.I(className="fas fa-hand-holding-usd text-info")),
                        html.Td("Return on Equity"),
                        html.Td(f"{info.get('returnOnEquity', 0) * 100:.2f}%" if info.get('returnOnEquity') else "N/A",
                               className="fw-bold text-end")
                    ]),
                    html.Tr([
                        html.Td(html.I(className="fas fa-balance-scale text-danger")),
                        html.Td("Debt to Equity"),
                        html.Td(f"{info.get('debtToEquity', 0):.2f}" if info.get('debtToEquity') else "N/A",
                               className="fw-bold text-end")
                    ]),
                    html.Tr([
                        html.Td(html.I(className="fas fa-tint text-primary")),
                        html.Td("Current Ratio"),
                        html.Td(f"{info.get('currentRatio', 0):.2f}" if info.get('currentRatio') else "N/A",
                               className="fw-bold text-end")
                    ]),
                ], className='table table-sm table-hover text-light')
            ]))
            
            comprehensive_financials_data = html.Div(comprehensive_financials)
        except Exception as e:
            comprehensive_financials_data = html.Div(f"Financial data unavailable: {str(e)}", className="text-muted text-center")

        output_graphs = closing_price_graph

        return (stock_details, fundamentals_data, quarterly_financials_data, daily_data_box,
                output_graphs, rsi_graph, bb_graph, intraday_graph, comparison_graph, 
                comprehensive_financials_data, financials_data, timestamp, timestamp, show_toast, refresh_message)

    except Exception as e:
        error_msg = str(e)
        error_details = html.Div([
            html.I(className="fas fa-exclamation-triangle text-danger me-2"),
            html.Span(f"Error: {error_msg}")
        ])

        return (
            error_details, "", "", "",
            html.Div("Chart unavailable", className="text-center text-muted p-5"),
            html.Div("Chart unavailable", className="text-center text-muted p-5"),
            html.Div("Chart unavailable", className="text-center text-muted p-5"),
            html.Div("Chart unavailable", className="text-center text-muted p-5"),
            html.Div("Chart unavailable", className="text-center text-muted p-5"),
            html.Div("Error loading news", className="text-center text-muted p-5"),
            "", timestamp, timestamp, show_toast, "Error loading data"
        )


@app.callback(
    Output("refresh-button", "children"),
    [Input("refresh-button", "n_clicks")]
)
def update_refresh_text(n_clicks):
    if n_clicks and n_clicks > 0:
        return [html.I(className="fas fa-sync-alt me-2"), "Refresh Data"]
    return [html.I(className="fas fa-sync-alt me-2"), "Refresh Data"]


if __name__ == '__main__':
    port = 8051
    use_reloader = True
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        Timer(1, lambda: webbrowser.open(f'http://localhost:{port}')).start()
    app.run(debug=True, port=port, use_reloader=use_reloader)
