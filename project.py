import datetime
import os

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from threading import Timer
import webbrowser
import yfinance as yf

def predict_future_prices_linear(hist, days=5):
    hist = hist[-30:]  # Last 30 days
    X = np.arange(len(hist)).reshape(-1, 1)
    y = hist['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(hist), len(hist) + days).reshape(-1, 1)
    future_preds = model.predict(future_X)

    return future_preds

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Indian Stock Market Analysis"

app.layout = dbc.Container([
    dcc.Loading(
        id="loading-spinner",
        type="circle",
        color="#00BFFF",
        fullscreen=True,
        children=html.Div([
            dbc.Row([
                dbc.Col(html.H1("Indian Stock Market Analysis", className='text-center text-light mb-4'), width=12),
            ]),
            dbc.Row([
                dbc.Col(html.H4("Enter NSE stock ticker (e.g., RELIANCE.NS, TCS.NS):", className='text-light'), width={"size": 6, "offset": 3}),
                dbc.Col(dcc.Input(
                    id='input-stock',
                    value='RELIANCE.NS',
                    type='text',
                    placeholder='Enter NSE Symbol like INFY.NS',
                    debounce=True,
                    className='form-control mb-3'
                ), width={"size": 6, "offset": 3}),
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='stock-details', className='text-center text-light'), width=12),
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='fundamentals-data', className='my-4 text-light'), width=12),
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='quarterly-financials-data', className='my-4 text-light'), width=12),
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='daily-data-box', className='my-4 text-light'), width=12),
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='output-graphs'), width=12),
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='intraday-graph'), width=12),
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='financials-data', className='my-4 text-light'), width=12),
            ]),
            dcc.Interval(
                id='interval-component',
                interval=60*1000,
                n_intervals=0
            )
        ])
    )
], fluid=True, style={'backgroundColor': '#121212'})

@app.callback(
    [Output('stock-details', 'children'),
     Output('fundamentals-data', 'children'),
     Output('quarterly-financials-data', 'children'),
     Output('daily-data-box', 'children'),
     Output('output-graphs', 'children'),
     Output('intraday-graph', 'children'),
     Output('financials-data', 'children')],
    [Input('input-stock', 'value'),
     Input('interval-component', 'n_intervals')]
)
def Stock_info(stock_ticker, n_intervals):
    try:
        if not stock_ticker:
            stock_ticker = "RELIANCE.NS"

        stock_ticker = stock_ticker.upper().strip()

        stock = yf.Ticker(stock_ticker)
        hist = stock.history(period="1mo", interval="1d")
        intraday = stock.history(period="1d", interval="5m")

        if hist.empty or intraday.empty:
            raise ValueError(f"No data found for {stock_ticker}. Please check if the symbol is correct.")

        info = stock.info
        company_name = info.get('shortName', stock_ticker)

        # Fundamental Data Panel
        fundamentals_data = html.Div([
            html.H4("Company Fundamentals", className='text-primary mb-3'),
            html.Table([
                html.Tr([html.Th("P/E Ratio"), html.Td(info.get('trailingPE', 'N/A'))]),
                html.Tr([html.Th("Market Cap"), html.Td(f"₹{info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else "N/A")]),
                html.Tr([html.Th("EPS"), html.Td(info.get('trailingEps', 'N/A'))]),
                html.Tr([html.Th("Dividend Yield"), html.Td(f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")]),
                html.Tr([html.Th("Sector"), html.Td(info.get('sector', 'N/A'))]),
                html.Tr([html.Th("Industry"), html.Td(info.get('industry', 'N/A'))]),
            ], className='table table-striped table-hover text-light')
        ], className='border p-4 bg-dark rounded text-light')

        # Quarterly Financials
        quarterly = stock.quarterly_financials
        if quarterly.empty:
            quarterly_financials_data = html.Div([
                html.H4("Quarterly Financials", className='text-primary mb-3'),
                html.P("No financial data available.", className='text-danger')
            ])
        else:
            latest_quarter = quarterly.iloc[:, 0]  
            quarterly_financials_data = html.Div([
                html.H4("Quarterly Financial Snapshot", className='text-primary mb-3'),
                html.Table([
                    html.Tr([html.Th("Total Revenue"), html.Td(f"₹{latest_quarter.get('Total Revenue', 'N/A')/1e7:.2f} Cr" if pd.notna(latest_quarter.get('Total Revenue')) else "N/A")]),
                    html.Tr([html.Th("Gross Profit"), html.Td(f"₹{latest_quarter.get('Gross Profit', 'N/A')/1e7:.2f} Cr" if pd.notna(latest_quarter.get('Gross Profit')) else "N/A")]),
                    html.Tr([html.Th("Net Income"), html.Td(f"₹{latest_quarter.get('Net Income', 'N/A')/1e7:.2f} Cr" if pd.notna(latest_quarter.get('Net Income')) else "N/A")]),
                    html.Tr([html.Th("EBITDA"), html.Td(f"₹{latest_quarter.get('EBITDA', 'N/A')/1e7:.2f} Cr" if pd.notna(latest_quarter.get('EBITDA')) else "N/A")]),
                    html.Tr([html.Th("Operating Income"), html.Td(f"₹{latest_quarter.get('Operating Income', 'N/A')/1e7:.2f} Cr" if pd.notna(latest_quarter.get('Operating Income')) else "N/A")]),
                ], className='table table-striped table-hover text-light')
            ], className='border p-4 bg-dark rounded text-light')

        # Stock details
        latest = hist.iloc[-1]
        open_price = latest['Open']
        close_price = latest['Close']
        high_price = latest['High']
        low_price = latest['Low']
        volume = latest['Volume']

        percent_change = ((close_price - open_price) / open_price) * 100
        price_direction = "up" if percent_change > 0 else "down"
        color = "lightgreen" if percent_change > 0 else "salmon"

        stock_details = html.Div([
            html.H4(f"{company_name} ({stock_ticker})", className='text-primary'),
            html.P(f"Open Price: ₹{open_price:.2f}"),
            html.P(f"Close Price: ₹{close_price:.2f}"),
            html.P(f"High Price: ₹{high_price:.2f}"),
            html.P(f"Low Price: ₹{low_price:.2f}"),
            html.P(f"Volume: {volume:.0f}"),
            html.P(f"Price is {price_direction} by {abs(percent_change):.2f}%", style={'color': color})
        ], className='border p-3 bg-dark rounded text-light')

        # Daily data
        daily_data_box = html.Div([
            html.H4("Today's Data", className='text-primary'),
            html.Table([
                html.Tr([html.Th("Open"), html.Td(f"₹{open_price:.2f}")]),
                html.Tr([html.Th("High"), html.Td(f"₹{high_price:.2f}")]),
                html.Tr([html.Th("Low"), html.Td(f"₹{low_price:.2f}")]),
                html.Tr([html.Th("Close"), html.Td(f"₹{close_price:.2f}")]),
                html.Tr([html.Th("Volume"), html.Td(f"{volume:.0f}")]),
                html.Tr([html.Th("Change"), html.Td(f"{price_direction} by {abs(percent_change):.2f}%", style={'color': color})])
            ], className='table table-striped table-hover text-light')
        ], className='border p-4 bg-dark rounded text-light')

        # Future Prediction
        future_predictions = predict_future_prices_linear(hist)

        financials_data = html.Div([
            html.H4("Future Price Predictions (Next 5 Days)", className='text-primary'),
            html.Table([
                html.Tr([html.Th("Day"), html.Th("Predicted Close Price")]),
                *[html.Tr([html.Td(f"Day {i + 1}"), html.Td(f"₹{price:.2f}")]) for i, price in enumerate(future_predictions)]
            ], className='table table-striped table-hover text-light')
        ], className='border p-4 bg-dark rounded text-light')

        # Graphs
        closing_price_graph = dcc.Graph(
            figure={
                'data': [go.Scatter(x=hist.index, y=hist['Close'], mode='lines+markers', name='Close Price', line=dict(color='orange'))],
                'layout': go.Layout(
                    title=f"Closing Prices of {company_name}",
                    xaxis={'title': 'Date', 'gridcolor': '#E2E2E2'},
                    yaxis={'title': 'Price (₹)', 'gridcolor': '#E2E2E2'},
                    transition={'duration': 500},
                    height=400,
                    plot_bgcolor='#1E1E1E',
                    paper_bgcolor='#121212'
                )
            }
        )

        intraday_graph = dcc.Graph(
            figure={
                'data': [go.Scatter(x=intraday.index, y=intraday['Close'], mode='lines+markers', name='Intraday 5-min', line=dict(color='cyan'))],
                'layout': go.Layout(
                    title=f"Intraday 5-Minute Close Price ({company_name})",
                    xaxis={'title': 'Time', 'gridcolor': '#E2E2E2'},
                    yaxis={'title': 'Price (₹)', 'gridcolor': '#E2E2E2'},
                    transition={'duration': 500},
                    height=400,
                    plot_bgcolor='#1E1E1E',
                    paper_bgcolor='#121212'
                )
            }
        )

        output_graphs = html.Div([closing_price_graph])

        return stock_details, fundamentals_data, quarterly_financials_data, daily_data_box, output_graphs, intraday_graph, financials_data

    except Exception as e:
        alert = dbc.Alert(
            f"Error: {str(e)}",
            color="danger",
            dismissable=True,
            is_open=True
        )
        return alert, "", "", "", "", "", ""

if __name__ == '__main__':
    def open_browser():
        webbrowser.open_new('http://127.0.0.1:8050/')
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        Timer(1, open_browser).start()
    app.run(debug=True)
