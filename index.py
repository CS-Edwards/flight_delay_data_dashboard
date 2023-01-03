import dash
from dash import html, dcc, dash_table as dt
import dash_daq as daq
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree
import plotly.express as px
import numpy as np
import sklearn.metrics as sm


data = pd.read_csv('C:\\Users\\Candace Edwards\\flight_data_dashboard\\data\\bquxjob_77dbbe1f_18574d93a6e.csv')
print(data.info())

feature_cols = ['departure_lat',	'departure_lon',	'arrival_lat',	'arrival_lon',	'departure_schedule',	'departure_actual',	'departure_delay','arrival_schedule','arrival_actual','arrival_delay'	]
data_2 = data.filter(feature_cols, axis=1)

#correlation matrix
cormat=data_2.corr()
fig = px.imshow(cormat, color_continuous_scale="Viridis", title= 'Pearson Correlation Matrix')



models = {'Linear Regression': linear_model.LinearRegression,
          'Decision Tree': tree.DecisionTreeRegressor}





#DASH

app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])

app.layout = html.Div([
    
    html.Div([
        html.Div([
            html.H3('Airline Delay Data Dashboard', style = {'margin-bottom':'0px', 'color':'#192444'}),
            html.H4('By: CS - Edwards', style = {'margin-bottom':'0px', 'color':'#192444'}),
            html.A(
                html.Button('Full Data in Colab'),
                href='https://github.com/CS-Edwards/flight_delay_data_dashboard/blob/main/flight_delay_reg_dashboard.ipynb',
                 target='_blank',
                ),
            html.Link('mailto: cedward2@hawaii.edu')


            
        ], id='title1')
        
        
    ], id= 'header',style={'margin-bottom':'25px'}),
    

    #charts container
    html.Div([

        #info box [drop down menu and correlation matrix]
        html.Div([
            #drop down
            html.Div([
                 #html.H4("Predicting Flight Delays"),
                 html.P("Select model:"),
                dcc.Dropdown(
                    id='dropdown',
                    options=["Linear Regression", "Decision Tree"],
                    value='Linear Regression',
                    clearable=False
                )
            ], id='drop_down_menu'),

            #corr matrix
            html.Div([
                #html.H4('Correlation Matrix'),
                dcc.Graph(figure=fig)
            ], style={'padding-top': '20px'})


        ], className='chart_div', id = 'left-info-box', style={'display':'flex','flex-direction':'column'}),

        #regression charts
        html.Div([
            html.H4("Flight Delay Predictions"),
            dcc.Graph(id="graph")
        ], className='chart_div', id = 'middle-chart-box'),

        #regression accuracy scores
        #info box [drop down menu and correlation matrix]
        html.Div([
            html.Div([
                daq.Gauge(
                    id = 'r2_score',
                    showCurrentValue=True,
                    color={"gradient":True,"ranges":{"green":[.66,1],"yellow":[.33,.66],"red":[0,.33]}},
                    value=.56, #adjust via callback
                    label='R2 Score',
                    max=1,
                    min=0,
                )
            ]),
            html.Div([
                daq.Gauge(
                    id='exp_var_score',
                    showCurrentValue=True,
                    color={"gradient":True,"ranges":{"green":[.66,1],"yellow":[.33,.66],"red":[0,.33]}},
                    value=.8, #adjust via callback
                    label='Explain Variance Score',
                    max=1,
                    min=0,
                )



            ])



        ], className='chart_div', id = 'right-info-box', style={'display':'flex','flex-direction':'column'})


    ], className = 'chart_container', style = {'display': 'flex', 'flex-direction':'row' })





], id = 'mainContainer', style = {'display':'flex', 'flex-direction':'column'})



#call back functions


@app.callback(
    [Output("graph", "figure"), Output('r2_score', "value"), Output('exp_var_score', "value")],
    [Input('dropdown', "value")])

def train_and_display(name):

    #Train/Test Split data
    X = data_2['departure_delay'] #training feature
    y = data_2['arrival_delay'] #label
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4, random_state=23)
    
    X_train = np.array(X_train).reshape(-1,1)
    X_test = np.array(X_test).reshape(-1,1)


    model = models[name]()
    model.fit(X_train, y_train)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    y_pred = model.predict(X_test)

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, 
                   name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, 
                   name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, 
                   name='prediction'),
                  
    ])

    fig.update_layout(xaxis_title ='Feature: Departure Delay (mins)', yaxis_title ='Label: Arrival Delay (mins)')


    #r2 score
    value = sm.r2_score(y_test, y_pred)

    
   #explain variance score
    value_exp = sm.explained_variance_score(y_test, y_pred)

    return  fig, value, value_exp


#run server
if __name__ == '__main__':
    app.run_server(debug=True)