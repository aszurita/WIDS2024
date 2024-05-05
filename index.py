from app import server,app
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import dash
import dash_bootstrap_components as dbc
# Pages
from pages import analisis, modelo,home


external_stylesheets = [dbc.themes.YETI]  
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True

dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Inicio", href="/home",    className='text-xlarge'),
        dbc.DropdownMenuItem("Analisis", href="/analisis",    className='text-xlarge'),
        dbc.DropdownMenuItem("Modelo", href="/modelo",    className='text-xlarge'),
    ],
    nav = True,
    in_navbar = True,
    label = "Secciones",
    className='text-xlarge'
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="assets/images/wids-logo.png", height="70px",width='200px')),                   
                        dbc.Col(dbc.NavbarBrand("WIDS DATATHON 2024",className='title-navbar')),
                    ],
                    align='center',
                ),
                href="/home",
            ),
            dbc.NavbarToggler(id="navbar-toggler2"),
            dbc.Collapse(
                dbc.Nav(
                    # right align dropdown menu with ml-auto className
                    [dropdown], navbar=True
                ),
                id="navbar-collapse2",
                navbar=True,
                style={'justifyContent':'flex-end'},
            ),
        ]
    ),
    color='Info',
    className="navbar",
)

def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

for i in [2]:
    app.callback(
        Output(f"navbar-collapse{i}", "is_open"),
        [Input(f"navbar-toggler{i}", "n_clicks")],
        [State(f"navbar-collapse{i}", "is_open")],
    )(toggle_navbar_collapse)

# embedding the navigation bar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
],className='background')


@app.callback(Output('page-content', 'children'),
            [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/analisis':
        return analisis.layout
    elif pathname == '/modelo':
        return modelo.layout
    else:
        return home.layout

if __name__ == '__main__':
    app.run_server(host='127.0.0.1', debug=True)