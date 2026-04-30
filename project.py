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
app.title = "StockVision — Indian Market Analysis"

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

            * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

            body { background: #111113; }

            ::-webkit-scrollbar { width: 6px; height: 6px; }
            ::-webkit-scrollbar-track { background: #18181b; }
            ::-webkit-scrollbar-thumb { background: #3f3f46; border-radius: 3px; }
            ::-webkit-scrollbar-thumb:hover { background: #52525b; }

            .card { border-radius: 6px !important; }

            .btn {
                font-weight: 500 !important;
                font-size: 0.8rem !important;
                border-radius: 4px !important;
                transition: background 0.15s ease !important;
            }
            .btn-primary {
                background: #2563eb !important;
                border: none !important;
                box-shadow: none !important;
            }
            .btn-primary:hover { background: #1d4ed8 !important; }
            .btn-secondary {
                background: #27272a !important;
                border: 1px solid #3f3f46 !important;
                color: #d4d4d8 !important;
                box-shadow: none !important;
            }
            .btn-secondary:hover { background: #3f3f46 !important; }

            .form-control, .form-select {
                background: #18181b !important;
                border: 1px solid #27272a !important;
                color: #e4e4e7 !important;
                border-radius: 4px !important;
                font-size: 0.85rem !important;
            }
            .form-control:focus, .form-select:focus {
                border-color: #3b82f6 !important;
                box-shadow: 0 0 0 1px rgba(59,130,246,0.3) !important;
            }

            .badge { font-weight: 500 !important; }

            .table { font-size: 0.82rem !important; color: #a1a1aa !important; }
            .table-hover tbody tr:hover { background-color: rgba(63,63,70,0.3) !important; }

            .tab {
                background: #18181b !important;
                border: 1px solid #27272a !important;
                color: #a1a1aa !important;
                font-size: 0.8rem !important;
            }
            .tab--selected {
                background: #27272a !important;
                border-bottom: 2px solid #3b82f6 !important;
                color: #e4e4e7 !important;
            }

            ._dash-loading { opacity: 0.3; }

            .card-header {
                background: #18181b !important;
                border-bottom: 1px solid #27272a !important;
                font-weight: 500 !important;
                font-size: 0.85rem !important;
                padding: 10px 16px !important;
            }

            .form-check-input:checked {
                background-color: #3b82f6 !important;
                border-color: #3b82f6 !important;
            }

            .toast {
                background: #18181b !important;
                border: 1px solid #27272a !important;
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
    'borderRadius': '6px',
    'marginBottom': '16px',
    'backgroundColor': '#18181b',
    'border': '1px solid #27272a',
}

graph_config = {
    'displayModeBar': True,
    'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'resetScale2d'],
    'displaylogo': False,
}

colors = {
    'background': '#111113',
    'text': '#e4e4e7',
    'primary': '#3b82f6',
    'secondary': '#22c55e',
    'accent': '#eab308',
    'positive': '#22c55e',
    'negative': '#ef4444',
    'neutral': '#a1a1aa',
    'dark_card': '#18181b',
    'card_header': '#18181b',
    'chart_bg': '#111113',
    'grid_color': 'rgba(63,63,70,0.4)',
    'border_color': '#27272a',
    'gradient_start': '#3b82f6',
    'gradient_end': '#3b82f6',
    'success_light': '#22c55e',
    'danger_light': '#ef4444',
    'info': '#3b82f6',
    'warning': '#eab308'
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
                html.H5("StockVision", className="fw-600 mb-0",
                        style={'color': '#e4e4e7', 'fontSize': '1rem'}),
                html.Span("Indian Market Analysis",
                          style={'color': '#71717a', 'fontSize': '0.75rem', 'marginLeft': '8px'})
            ], className="d-flex align-items-center", style={'padding': '12px 0'})
        ], width=12)
    ], style={
        'borderBottom': '1px solid #27272a',
        'marginBottom': '16px',
    }),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Stock Selection"),
                dbc.CardBody([
                    html.Label("NSE Ticker", className="form-label",
                              style={'fontSize': '0.75rem', 'color': '#71717a'}),
                    dbc.Input(
                        id='input-stock',
                        value='RELIANCE.NS',
                        type='text',
                        placeholder='e.g., INFY.NS, TCS.NS',
                        className='mb-3'
                    ),
                    html.Label("Quick Select", className="form-label",
                              style={'fontSize': '0.75rem', 'color': '#71717a'}),
                    dbc.RadioItems(
                        id='popular-stocks',
                        options=popular_stocks,
                        value='RELIANCE.NS',
                        className="mb-3",
                        inline=False
                    ),
                    html.Label("Compare With", className="form-label",
                              style={'fontSize': '0.75rem', 'color': '#71717a'}),
                    dbc.Input(
                        id='compare-stock',
                        value='',
                        type='text',
                        placeholder='e.g., TCS.NS',
                        className='mb-3'
                    ),
                    dbc.Button(
                        "Analyze",
                        id="analyze-button",
                        color="primary",
                        className="w-100 mb-2"
                    ),
                    dbc.Button(
                        "Refresh",
                        id="refresh-button",
                        color="secondary",
                        className="w-100",
                        n_clicks=0
                    ),
                ])
            ], style=card_style),

            dbc.Card([
                dbc.CardHeader("Price"),
                dbc.CardBody([
                    html.Div(id='stock-details', className='text-center')
                ])
            ], style=card_style),

            dbc.Card([
                dbc.CardHeader("Predictions (5-day)"),
                dbc.CardBody([
                    html.Div(id='financials-data')
                ])
            ], style=card_style)
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Span("Price History", style={'color': '#e4e4e7'}),
                        dbc.ButtonGroup([
                            dbc.Button("1W", id="1w-button", color="primary", outline=True, size="sm"),
                            dbc.Button("1M", id="1m-button", color="primary", size="sm"),
                            dbc.Button("3M", id="3m-button", color="primary", outline=True, size="sm"),
                            dbc.Button("1Y", id="1y-button", color="primary", outline=True, size="sm"),
                        ])
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
                        dcc.Tab(label="Bollinger", value="bb", children=[
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
                        html.Span("Intraday", style={'color': '#e4e4e7'}),
                        html.Span("5m", style={'color': '#52525b', 'fontSize': '0.75rem'})
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
                dbc.CardHeader("Comparison"),
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
                dbc.CardHeader("Today"),
                dbc.CardBody([
                    html.Div(id='daily-data-box')
                ])
            ], style=card_style),

            dbc.Card([
                dbc.CardHeader("Fundamentals"),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-3",
                        type="circle",
                        children=html.Div(id='fundamentals-data')
                    )
                ])
            ], style=card_style),

            dbc.Card([
                dbc.CardHeader("Quarterly Results"),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-4",
                        type="circle",
                        children=html.Div(id='quarterly-financials-data')
                    )
                ])
            ], style=card_style),

            dbc.Card([
                dbc.CardHeader("Financials"),
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
                    "StockVision · Yahoo Finance · ",
                    html.Span(id="last-updated", style={'color': '#52525b'})
                ], style={'color': '#3f3f46', 'fontSize': '0.75rem'}, className="text-center mb-0")
            ], style={'borderTop': '1px solid #27272a', 'padding': '12px 0', 'marginTop': '16px'})
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
                    html.Td("P/E Ratio", style={'color': '#71717a'}),
                    html.Td(f"{info.get('trailingPE', 'N/A'):.2f}" if isinstance(info.get('trailingPE'),
                                                                                 (int, float)) else "N/A",
                            className="text-end", style={'color': '#e4e4e7'})
                ]),
                html.Tr([
                    html.Td("Market Cap", style={'color': '#71717a'}),
                    html.Td(f"₹{info.get('marketCap') / 1e9:.2f}B" if info.get('marketCap') else "N/A",
                            className="text-end", style={'color': '#e4e4e7'})
                ]),
                html.Tr([
                    html.Td("EPS", style={'color': '#71717a'}),
                    html.Td(f"₹{info.get('trailingEps', 'N/A')}" if info.get('trailingEps') else "N/A",
                            className="text-end", style={'color': '#e4e4e7'})
                ]),
                html.Tr([
                    html.Td("Div Yield", style={'color': '#71717a'}),
                    html.Td(f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else "N/A",
                            className="text-end", style={'color': '#e4e4e7'})
                ]),
                html.Tr([
                    html.Td("Sector", style={'color': '#71717a'}),
                    html.Td(info.get('sector', 'N/A'), className="text-end", style={'color': '#e4e4e7'})
                ]),
                html.Tr([
                    html.Td("Industry", style={'color': '#71717a'}),
                    html.Td(info.get('industry', 'N/A'), className="text-end", style={'color': '#e4e4e7'})
                ]),
            ], className='table table-sm table-hover')
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
                        html.Td("Revenue", style={'color': '#71717a'}),
                        html.Td(f"₹{latest_quarter.get('Total Revenue', 'N/A') / 1e7:.2f} Cr" if pd.notna(
                            latest_quarter.get('Total Revenue')) else "N/A",
                                className="text-end", style={'color': '#e4e4e7'})
                    ]),
                    html.Tr([
                        html.Td("Gross Profit", style={'color': '#71717a'}),
                        html.Td(f"₹{latest_quarter.get('Gross Profit', 'N/A') / 1e7:.2f} Cr" if pd.notna(
                            latest_quarter.get('Gross Profit')) else "N/A",
                                className="text-end", style={'color': '#e4e4e7'})
                    ]),
                    html.Tr([
                        html.Td("Net Income", style={'color': '#71717a'}),
                        html.Td(f"₹{latest_quarter.get('Net Income', 'N/A') / 1e7:.2f} Cr" if pd.notna(
                            latest_quarter.get('Net Income')) else "N/A",
                                className="text-end", style={'color': '#e4e4e7'})
                    ]),
                    html.Tr([
                        html.Td("EBITDA", style={'color': '#71717a'}),
                        html.Td(f"₹{latest_quarter.get('EBITDA', 'N/A') / 1e7:.2f} Cr" if pd.notna(
                            latest_quarter.get('EBITDA')) else "N/A",
                                className="text-end", style={'color': '#e4e4e7'})
                    ]),
                    html.Tr([
                        html.Td("Oper. Income", style={'color': '#71717a'}),
                        html.Td(f"₹{latest_quarter.get('Operating Income', 'N/A') / 1e7:.2f} Cr" if pd.notna(
                            latest_quarter.get('Operating Income')) else "N/A",
                                className="text-end", style={'color': '#e4e4e7'})
                    ]),
                ], className='table table-sm table-hover')
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
                html.Span(f"{company_name}", style={'color': '#e4e4e7', 'fontSize': '0.85rem', 'fontWeight': '500'}),
                html.Span(f" {stock_ticker}", style={'color': '#52525b', 'fontSize': '0.75rem'}),
            ], className='text-center mb-2'),
            html.Div([
                html.Span(f"₹{close_price:.2f}", style={'fontSize': '1.6rem', 'fontWeight': '600', 'color': '#e4e4e7'}),
            ], className='text-center mb-1'),
            html.Div([
                html.Span(f"{'▲' if percent_change > 0 else '▼'} {abs(percent_change):.2f}%",
                         style={'color': color, 'fontSize': '0.85rem', 'fontWeight': '500'})
            ], className='text-center mb-2'),
            html.Hr(style={'borderColor': '#27272a', 'margin': '12px 0'}),
            html.Div([
                html.Span(f"Vol {volume:,.0f}", style={'color': '#71717a', 'fontSize': '0.75rem'}),
            ], className="text-center")
        ], style={'padding': '8px'})

        daily_data_box = html.Div([
            html.Table([
                html.Tr([
                    html.Td("Open", style={'color': '#71717a', 'fontSize': '0.8rem', 'padding': '6px 0'}),
                    html.Td(f"₹{open_price:.2f}", className="text-end",
                            style={'color': '#e4e4e7', 'fontSize': '0.8rem', 'fontWeight': '500', 'padding': '6px 0'})
                ]),
                html.Tr([
                    html.Td("Close", style={'color': '#71717a', 'fontSize': '0.8rem', 'padding': '6px 0'}),
                    html.Td(f"₹{close_price:.2f}", className="text-end",
                            style={'color': '#e4e4e7', 'fontSize': '0.8rem', 'fontWeight': '500', 'padding': '6px 0'})
                ]),
                html.Tr([
                    html.Td("High", style={'color': '#71717a', 'fontSize': '0.8rem', 'padding': '6px 0'}),
                    html.Td(f"₹{high_price:.2f}", className="text-end",
                            style={'color': colors['positive'], 'fontSize': '0.8rem', 'fontWeight': '500', 'padding': '6px 0'})
                ]),
                html.Tr([
                    html.Td("Low", style={'color': '#71717a', 'fontSize': '0.8rem', 'padding': '6px 0'}),
                    html.Td(f"₹{low_price:.2f}", className="text-end",
                            style={'color': colors['negative'], 'fontSize': '0.8rem', 'fontWeight': '500', 'padding': '6px 0'})
                ]),
                html.Tr([
                    html.Td("Change", style={'color': '#71717a', 'fontSize': '0.8rem', 'padding': '6px 0'}),
                    html.Td(f"{'▲' if percent_change > 0 else '▼'} {abs(percent_change):.2f}%", className="text-end",
                            style={'color': color, 'fontSize': '0.8rem', 'fontWeight': '500', 'padding': '6px 0'})
                ]),
            ], style={'width': '100%'})
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
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Date", style={'color': '#71717a', 'fontWeight': '500'}),
                        html.Th("Price", className="text-end", style={'color': '#71717a', 'fontWeight': '500'}),
                        html.Th("", style={"width": "30px"})
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(date, style={'color': '#a1a1aa'}),
                        html.Td(f"₹{price:.2f}", className="text-end", style={'color': '#e4e4e7'}),
                        html.Td(html.I(className=f"fas {icon}"), className="text-center")
                    ]) for date, price, icon in zip(future_dates, future_predictions, trends)
                ])
            ], className='table table-sm table-hover')
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
                'bgcolor': '#18181b',
                'font_size': 12,
                'font_family': 'Inter, system-ui, sans-serif',
                'bordercolor': '#3f3f46'
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
                fillcolor='rgba(59, 130, 246, 0.08)',
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
                fillcolor='rgba(59, 130, 246, 0.06)',
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
                    name='Price'
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
                    fillcolor='rgba(59, 130, 246, 0.08)',
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
                    html.Span("Balance Sheet", style={'color': '#a1a1aa', 'fontSize': '0.75rem', 'fontWeight': '500'}),
                    html.Table([
                        html.Tr([
                            html.Td("Total Assets", style={'color': '#71717a'}),
                            html.Td(f"₹{latest_bs.get('Total Assets', 0) / 1e9:.2f}B" if pd.notna(latest_bs.get('Total Assets')) else "N/A",
                                   className="text-end", style={'color': '#e4e4e7'})
                        ]),
                        html.Tr([
                            html.Td("Total Liabilities", style={'color': '#71717a'}),
                            html.Td(f"₹{latest_bs.get('Total Liabilities Net Minority Interest', 0) / 1e9:.2f}B" if pd.notna(latest_bs.get('Total Liabilities Net Minority Interest')) else "N/A",
                                   className="text-end", style={'color': '#e4e4e7'})
                        ]),
                        html.Tr([
                            html.Td("Equity", style={'color': '#71717a'}),
                            html.Td(f"₹{latest_bs.get('Stockholders Equity', 0) / 1e9:.2f}B" if pd.notna(latest_bs.get('Stockholders Equity')) else "N/A",
                                   className="text-end", style={'color': '#e4e4e7'})
                        ]),
                        html.Tr([
                            html.Td("Cash", style={'color': '#71717a'}),
                            html.Td(f"₹{latest_bs.get('Cash And Cash Equivalents', 0) / 1e9:.2f}B" if pd.notna(latest_bs.get('Cash And Cash Equivalents')) else "N/A",
                                   className="text-end", style={'color': '#e4e4e7'})
                        ]),
                    ], className='table table-sm table-hover')
                ], className="mb-3"))
            
            if not cash_flow.empty:
                latest_cf = cash_flow.iloc[:, 0]
                comprehensive_financials.append(html.Div([
                    html.Span("Cash Flow", style={'color': '#a1a1aa', 'fontSize': '0.75rem', 'fontWeight': '500'}),
                    html.Table([
                        html.Tr([
                            html.Td("Operating", style={'color': '#71717a'}),
                            html.Td(f"₹{latest_cf.get('Operating Cash Flow', 0) / 1e9:.2f}B" if pd.notna(latest_cf.get('Operating Cash Flow')) else "N/A",
                                   className="text-end", style={'color': '#e4e4e7'})
                        ]),
                        html.Tr([
                            html.Td("Investing", style={'color': '#71717a'}),
                            html.Td(f"₹{latest_cf.get('Investing Cash Flow', 0) / 1e9:.2f}B" if pd.notna(latest_cf.get('Investing Cash Flow')) else "N/A",
                                   className="text-end", style={'color': '#e4e4e7'})
                        ]),
                        html.Tr([
                            html.Td("Financing", style={'color': '#71717a'}),
                            html.Td(f"₹{latest_cf.get('Financing Cash Flow', 0) / 1e9:.2f}B" if pd.notna(latest_cf.get('Financing Cash Flow')) else "N/A",
                                   className="text-end", style={'color': '#e4e4e7'})
                        ]),
                        html.Tr([
                            html.Td("Free CF", style={'color': '#71717a'}),
                            html.Td(f"₹{latest_cf.get('Free Cash Flow', 0) / 1e9:.2f}B" if pd.notna(latest_cf.get('Free Cash Flow')) else "N/A",
                                   className="text-end", style={'color': '#e4e4e7'})
                        ]),
                    ], className='table table-sm table-hover')
                ], className="mb-3"))
            
            if not income_stmt.empty:
                latest_is = income_stmt.iloc[:, 0]
                comprehensive_financials.append(html.Div([
                    html.Span("Income Statement", style={'color': '#a1a1aa', 'fontSize': '0.75rem', 'fontWeight': '500'}),
                    html.Table([
                        html.Tr([
                            html.Td("Revenue", style={'color': '#71717a'}),
                            html.Td(f"₹{latest_is.get('Total Revenue', 0) / 1e9:.2f}B" if pd.notna(latest_is.get('Total Revenue')) else "N/A",
                                   className="text-end", style={'color': '#e4e4e7'})
                        ]),
                        html.Tr([
                            html.Td("Cost of Rev.", style={'color': '#71717a'}),
                            html.Td(f"₹{latest_is.get('Cost Of Revenue', 0) / 1e9:.2f}B" if pd.notna(latest_is.get('Cost Of Revenue')) else "N/A",
                                   className="text-end", style={'color': '#e4e4e7'})
                        ]),
                        html.Tr([
                            html.Td("Gross Profit", style={'color': '#71717a'}),
                            html.Td(f"₹{latest_is.get('Gross Profit', 0) / 1e9:.2f}B" if pd.notna(latest_is.get('Gross Profit')) else "N/A",
                                   className="text-end", style={'color': '#e4e4e7'})
                        ]),
                        html.Tr([
                            html.Td("Oper. Income", style={'color': '#71717a'}),
                            html.Td(f"₹{latest_is.get('Operating Income', 0) / 1e9:.2f}B" if pd.notna(latest_is.get('Operating Income')) else "N/A",
                                   className="text-end", style={'color': '#e4e4e7'})
                        ]),
                        html.Tr([
                            html.Td("Net Income", style={'color': '#71717a'}),
                            html.Td(f"₹{latest_is.get('Net Income', 0) / 1e9:.2f}B" if pd.notna(latest_is.get('Net Income')) else "N/A",
                                   className="text-end", style={'color': '#e4e4e7'})
                        ]),
                    ], className='table table-sm table-hover')
                ]))
            
            comprehensive_financials.append(html.Div([
                html.Span("Key Ratios", style={'color': '#a1a1aa', 'fontSize': '0.75rem', 'fontWeight': '500'}),
                html.Table([
                    html.Tr([
                        html.Td("Profit Margin", style={'color': '#71717a'}),
                        html.Td(f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get('profitMargins') else "N/A",
                               className="text-end", style={'color': '#e4e4e7'})
                    ]),
                    html.Tr([
                        html.Td("Oper. Margin", style={'color': '#71717a'}),
                        html.Td(f"{info.get('operatingMargins', 0) * 100:.2f}%" if info.get('operatingMargins') else "N/A",
                               className="text-end", style={'color': '#e4e4e7'})
                    ]),
                    html.Tr([
                        html.Td("ROA", style={'color': '#71717a'}),
                        html.Td(f"{info.get('returnOnAssets', 0) * 100:.2f}%" if info.get('returnOnAssets') else "N/A",
                               className="text-end", style={'color': '#e4e4e7'})
                    ]),
                    html.Tr([
                        html.Td("ROE", style={'color': '#71717a'}),
                        html.Td(f"{info.get('returnOnEquity', 0) * 100:.2f}%" if info.get('returnOnEquity') else "N/A",
                               className="text-end", style={'color': '#e4e4e7'})
                    ]),
                    html.Tr([
                        html.Td("D/E Ratio", style={'color': '#71717a'}),
                        html.Td(f"{info.get('debtToEquity', 0):.2f}" if info.get('debtToEquity') else "N/A",
                               className="text-end", style={'color': '#e4e4e7'})
                    ]),
                    html.Tr([
                        html.Td("Current Ratio", style={'color': '#71717a'}),
                        html.Td(f"{info.get('currentRatio', 0):.2f}" if info.get('currentRatio') else "N/A",
                               className="text-end", style={'color': '#e4e4e7'})
                    ]),
                ], className='table table-sm table-hover')
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
        error_details = html.Div(
            f"Error: {error_msg}", style={'color': '#ef4444', 'fontSize': '0.8rem'}
        )

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
    return "Refresh"


if __name__ == '__main__':
    port = 8051
    use_reloader = True
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        Timer(1, lambda: webbrowser.open(f'http://localhost:{port}')).start()
    app.run(debug=True, port=port, use_reloader=use_reloader)
