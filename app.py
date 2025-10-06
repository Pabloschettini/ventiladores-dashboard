# app.py
"""
Dashboard Interativo: Previsão de Falhas em Ventiladores Centrífugos
Autor: Pablo Schettini Loureiro
Descrição: Dashboard interativo para explorar dados simulados de manutenção de ventiladores.
"""

import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Carregar dados simulados
# ---------------------------
data = pd.read_csv('data/ventiladores_simulados.csv', parse_dates=['Timestamp'])

# ---------------------------
# 2. Preparar modelo preditivo
# ---------------------------
X = data[['Vibration', 'Temperature', 'Hours_Operated']]
y = data['Failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
data['Predicted_Failure'] = model.predict(X)

# ---------------------------
# 3. Inicializar app Dash
# ---------------------------
app = dash.Dash(__name__)
app.title = "Dashboard Manutenção Preditiva"

# ---------------------------
# 4. Layout do Dashboard
# ---------------------------
app.layout = html.Div([
    html.H1("Dashboard Interativo: Manutenção Preditiva de Ventiladores Centrífugos",
            style={'textAlign': 'center'}),
    
    html.Div([
        html.Label("Selecione o ventilador:"),
        dcc.Dropdown(
            id='fan-dropdown',
            options=[{'label': f, 'value': f} for f in sorted(data['Fan_ID'].unique())],
            value=sorted(data['Fan_ID'].unique())[0]
        )
    ], style={'width': '50%', 'margin': 'auto', 'padding': '20px'}),

    dcc.Graph(id='vibration-temp-graph'),
    dcc.Graph(id='vibration-time-graph')
])

# ---------------------------
# 5. Callback para atualização dos gráficos
# ---------------------------
@app.callback(
    [dash.dependencies.Output('vibration-temp-graph', 'figure'),
     dash.dependencies.Output('vibration-time-graph', 'figure')],
    [dash.dependencies.Input('fan-dropdown', 'value')]
)
def update_graph(fan_id):
    df = data[data['Fan_ID'] == fan_id]
    
    fig1 = px.scatter(df, x='Vibration', y='Temperature', color='Predicted_Failure',
                      title=f'Falhas Previstas - Ventilador {fan_id}',
                      labels={'Predicted_Failure': 'Falha Prevista'},
                      color_discrete_map={0: 'blue', 1: 'red'})
    
    fig2 = px.line(df, x='Timestamp', y='Vibration', color='Predicted_Failure',
                   title=f'Evolução da Vibração - Ventilador {fan_id}',
                   labels={'Predicted_Failure': 'Falha Prevista'},
                   color_discrete_map={0: 'blue', 1: 'red'})
    
    return fig1, fig2

# ---------------------------
# 6. Rodar o servidor
# ---------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8060, debug=True)

