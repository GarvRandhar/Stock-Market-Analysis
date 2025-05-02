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


def predict_future_prices_linear(hist, days=5):
    hist = hist[-30:]  
    X = np.arange(len(hist)).reshape(-1, 1)
    y = hist['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(hist), len(hist) + days).reshape(-1, 1)
    future_preds = model.predict(future_X)

    return future_preds


external_stylesheets = [
    dbc.themes.DARKLY,  
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "StockVision - Indian Market Analysis"

card_style = {
    'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
    'borderRadius': '10px',
    'marginBottom': '20px',
    'backgroundColor': '#2C3E50',
    'transition': 'transform 0.3s',
}

graph_config = {
    'displayModeBar': True,
    'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'resetScale2d'],
    'displaylogo': False,
}

colors = {
    'background': '#1E1E2E',
    'text': '#EEEEEE',
    'primary': '#3498DB',
    'secondary': '#2ECC71',
    'accent': '#E74C3C',
    'positive': '#2ECC71',
    'negative': '#E74C3C',
    'neutral': '#F39C12',
    'dark_card': '#2C3E50',
    'card_header': '#34495E'
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
    ], style={'background': 'linear-gradient(120deg, #2C3E50, #1A237E)', 'borderRadius': '0 0 15px 15px',
              'marginBottom': '25px'}),

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
                            "Price History"
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
                    dcc.Loading(
                        id="loading-1",
                        type="circle",
                        children=html.Div(id='output-graphs')
                    )
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
        interval=60 * 1000,  
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
     Output('intraday-graph', 'children'),
     Output('financials-data', 'children'),
     Output('last-updated', 'children'),
     Output('last-update-time', 'data'),
     Output('update-toast', 'is_open'),
     Output('update-toast', 'children')],
    [Input('analyze-button', 'n_clicks'),
     Input('refresh-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('input-stock', 'value'),
     State('time-period-store', 'data'),
     State('last-update-time', 'data')]
)
def update_stock_info(analyze_clicks, refresh_clicks, n_intervals, stock_ticker, time_period, last_update):
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
            html.H3(f"{company_name}", className='text-center mb-2'),
            html.H6(f"({stock_ticker})", className='text-center text-muted mb-3'),
            html.H2([
                f"₹{close_price:.2f} ",
                html.Span([
                    html.I(className=f"fas {icon} me-1"),
                    f"{abs(percent_change):.2f}%"
                ], style={'color': color, 'fontSize': '1rem'})
            ], className='text-center mb-3'),
            html.Div([
                dbc.Badge(f"Vol: {volume:,.0f}", color="secondary", className="me-2 px-2 py-1"),
                dbc.Badge("NSE", color="primary", className="px-2 py-1")
            ], className="text-center")
        ])

        daily_data_box = html.Div([
            html.Div([
                html.Div([
                    html.Span("Open", className="text-muted d-block"),
                    html.H5(f"₹{open_price:.2f}", className="mb-0")
                ], className="col-6 mb-3"),
                html.Div([
                    html.Span("Close", className="text-muted d-block"),
                    html.H5(f"₹{close_price:.2f}", className="mb-0")
                ], className="col-6 mb-3"),
                html.Div([
                    html.Span("High", className="text-muted d-block"),
                    html.H5(f"₹{high_price:.2f}", className="mb-0 text-success")
                ], className="col-6 mb-3"),
                html.Div([
                    html.Span("Low", className="text-muted d-block"),
                    html.H5(f"₹{low_price:.2f}", className="mb-0 text-danger")
                ], className="col-6 mb-3"),
            ], className="row text-center"),
            html.Div([
                html.Span("Change", className="text-muted d-block text-center"),
                html.Div([
                    html.I(className=f"fas {icon} me-2"),
                    f"{price_direction.upper()} by {abs(percent_change):.2f}%"
                ], className="text-center fw-bold", style={'color': color})
            ], className="mt-2")
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
            'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40},
            'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02},
            'font': {'color': colors['text']},
            'plot_bgcolor': colors['dark_card'],
            'paper_bgcolor': colors['dark_card'],
            'hovermode': 'x unified'
        }

        closing_price_graph = dcc.Graph(
            figure={
                'data': [
                    go.Scatter(
                        x=hist.index,
                        y=hist['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color=colors['primary'], width=2),
                        fill='tozeroy',
                        fillcolor=f'rgba(52, 152, 219, 0.2)',
                        hovertemplate='<b>Date</b>: %{x|%d %b}<br><b>Price</b>: ₹%{y:.2f}<extra></extra>'
                    )
                ],
                'layout': {
                    'title': f"{time_period} Price History",
                    'xaxis': {
                        'title': 'Date',
                        'showgrid': True,
                        'gridcolor': 'rgba(255, 255, 255, 0.1)'
                    },
                    'yaxis': {
                        'title': 'Price (₹)',
                        'showgrid': True,
                        'gridcolor': 'rgba(255, 255, 255, 0.1)'
                    },
                    'height': 350,
                    **common_layout
                }
            },
            config=graph_config
        )

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
                    name='Price'
                )
            )
        else:
            intraday_fig.add_trace(
                go.Scatter(
                    x=intraday.index,
                    y=intraday['Close'],
                    mode='lines+markers',
                    name='Intraday',
                    line=dict(color=colors['secondary'], width=2),
                    marker=dict(size=6),
                    hovertemplate='<b>Time</b>: %{x|%H:%M}<br><b>Price</b>: ₹%{y:.2f}<extra></extra>'
                )
            )

        intraday_fig.update_layout(
            title="Today's Intraday Performance",
            xaxis={
                'title': 'Time',
                'showgrid': True,
                'gridcolor': 'rgba(255, 255, 255, 0.1)'
            },
            yaxis={
                'title': 'Price (₹)',
                'showgrid': True,
                'gridcolor': 'rgba(255, 255, 255, 0.1)'
            },
            height=350,
            margin={'l': 40, 'r': 40, 't': 40, 'b': 40},
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02},
            font={'color': colors['text']},
            plot_bgcolor=colors['dark_card'],
            paper_bgcolor=colors['dark_card'],
            hovermode='x unified'
        )

        intraday_graph = dcc.Graph(figure=intraday_fig, config=graph_config)
        output_graphs = closing_price_graph

        return (stock_details, fundamentals_data, quarterly_financials_data, daily_data_box,
                output_graphs, intraday_graph, financials_data, timestamp, timestamp, show_toast, refresh_message)

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
    port = 8050
    use_reloader = True
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        Timer(1, lambda: webbrowser.open(f'http://localhost:{port}')).start()
    app.run(debug=True, port=port, use_reloader=use_reloader)
