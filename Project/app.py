# %%
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, ALL, dash, html, dcc
from pages import tab1, tab2, tab3, tab4, tab5, tab6, tab7

from data import df
import plotly.graph_objects as go
import scipy
from plotly.subplots import make_subplots

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# %%
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.YETI], suppress_callback_exceptions=True)

app.layout = html.Div([
    html.Img(src='assets/vt.png',
             style={'width': '5%', 'display': 'inline-block', 'verticalAlign': 'middle'}),
    html.H1("Credit card Approval",
            style={'color': 'blue', 'fontsize': 25, 'fontFamily': 'serif',
                   'display': 'inline-block', 'verticalAlign': 'middle', 'marginLeft': '10px'}),
    # dbc.Button("reset", id="getDataset", style={'float': 'right'}),

    html.Br(),
    dcc.Tabs(id="tabs-example", value='tab-1', children=[
        dcc.Tab(label='Data Cleaning', value='tab-1',
                style={'backgroundColor': 'lightblue', 'fontSize': '15px', 'margin': '2px'},
                selected_style={'backgroundColor': 'blue', 'color': 'white'}),
        dcc.Tab(label='Outlier detection and removal', value='tab-2',
                style={'backgroundColor': 'lightblue', 'fontSize': '15px', 'margin': '2px'},
                selected_style={'backgroundColor': 'blue', 'color': 'white'}),
        dcc.Tab(label='Dimensionality Seduction(PTA)', value='tab-3',
                style={'backgroundColor': 'lightblue', 'fontSize': '15px', 'margin': '2px'},
                selected_style={'backgroundColor': 'blue', 'color': 'white'}),
        dcc.Tab(label='Normality tests', value='tab-4',
                style={'backgroundColor': 'lightblue', 'fontSize': '15px', 'margin': '2px'},
                selected_style={'backgroundColor': 'blue', 'color': 'white'}),
        dcc.Tab(label='Data Transformation', value='tab-5',
                style={'backgroundColor': 'lightblue', 'fontSize': '15px', 'margin': '2px'},
                selected_style={'backgroundColor': 'blue', 'color': 'white'}),
        dcc.Tab(label='Loading Data', value='tab-6',
                style={'backgroundColor': 'lightblue', 'fontSize': '15px', 'margin': '2px'},
                selected_style={'backgroundColor': 'blue', 'color': 'white'}),
        dcc.Tab(label='Plots', value='tab-7',
                style={'backgroundColor': 'lightblue', 'fontSize': '15px', 'margin': '2px'},
                selected_style={'backgroundColor': 'blue', 'color': 'white'}),
    ], style={'fontSize': '15px'}),
    html.Div(id='tabs-content')
], style={'margin': '20px', 'backgroundColor': '#EFF8FB'})


@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs-example', 'value')],
)
def render_content(tab):
    if tab == 'tab-1':
        return tab1.layout
    elif tab == 'tab-2':
        return tab2.layout
    elif tab == 'tab-3':
        return tab3.layout
    elif tab == 'tab-4':
        return tab4.layout
    elif tab == 'tab-5':
        return tab5.layout
    elif tab == 'tab-6':
        return tab6.layout
    elif tab == 'tab-7':
        return tab7.layout


@app.callback(
    [
        Output('modal', 'is_open'),
        Output('modal-body', 'children'),
        Output('close', 'n_clicks'),
        Output('data-table', 'data'),
    ],
    [
        Input('submit-val', 'n_clicks'),
        Input('close', 'n_clicks')],
    [
        State({'type': 'tab1-dropdown', 'index': ALL}, 'value'),
        State({'type': 'tab1-dropdown', 'index': ALL}, 'id'),
        State({'type': 'allow-null', 'index': ALL}, 'value'),
        State({'type': 'remove-duplicate', 'index': ALL}, 'value'),
        State('modal', 'is_open'),
    ],
)
def update_output(n_clicks, close_clicks, dropdown_values, dropdown_indices, allow_null_values,
                  remove_duplicate_values, is_open):
    if close_clicks:
        return False, "", 0, df.head(10).to_dict('records')

    if n_clicks and n_clicks > 0:
        try:
            print('dropdown_values', dropdown_values)
            for i, dropdown_value in enumerate(dropdown_values):
                col = dropdown_indices[i]['index']
                new_dtype = dropdown_values[i]
                print('col : ', col)
                print('data_type : ', new_dtype)
                if new_dtype == 'Integer':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='raise')
                        df[col] = df[col].astype('int64')
                    except ValueError as e:
                        raise ValueError(
                            f"Column {col} contains non-integer values that can't be converted to int64: {str(e)}")

                elif new_dtype == 'Float':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='raise')
                        df[col] = df[col].astype('float64')
                    except ValueError as e:
                        raise ValueError(
                            f"Column {col} contains non-float values that can't be converted to float64: {str(e)}")

                elif new_dtype == 'String':
                    df[col] = df[col].astype('str')

                elif new_dtype == 'Boolean':
                    try:
                        df[col] = df[col].astype('bool')
                    except ValueError as e:
                        raise ValueError(f"Column {col} contains non-boolean values: {str(e)}")

                if 'allow_null' not in (allow_null_values[i] or []):
                    df[col] = df[col].fillna('')

                if remove_duplicate_values[i] is not None and 'remove_duplicate' in remove_duplicate_values[i]:
                    df.drop_duplicates(subset=[col], inplace=True)
                    print('remove')

            return True, "Success: Dataframe updated successfully!", 0, df.head(10).to_dict('records')

        except ValueError as e:
            return True, f"Error: {str(e)}", 0, df.head(10).to_dict('records')

    return is_open, "", close_clicks, df.head(10).to_dict('records')


# tab2.py callbacks
@app.callback(
    [Output('tab2_graph1', 'figure'),
     Output('selected-column', 'children')],
    [Input(col, 'n_clicks_timestamp') for col in numerical_cols]
)
def update_graph(*timestamps):
    import plotly.express as px

    clicked_col = None
    if any(timestamps):
        last_clicked_idx = timestamps.index(max([ts or 0 for ts in timestamps]))
        clicked_col = numerical_cols[last_clicked_idx]

    if clicked_col:
        df_sorted = df.sort_values(by='AGE')
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Box Plot", "Bar Plot", "Violin Plot", "Scatter Plot")
        )
        fig.add_trace(
            px.box(df_sorted, x='AGE_GROUP', y=clicked_col, labels={'y': clicked_col},
                   color_discrete_sequence=px.colors.sequential.Viridis).data[0],
            row=1, col=1,
        )

        fig.update_traces(width=1)
        fig.update_yaxes(title_text=clicked_col, row=1, col=1)
        fig.update_xaxes(title_text="AGE_GROUP", row=1, col=1)

        fig.add_trace(
            px.bar(df_sorted, x='AGE', y=clicked_col, labels={'x': 'AGE', 'y': clicked_col},
                   color_discrete_sequence=px.colors.sequential.Viridis).data[0],
            row=1, col=2
        )
        fig.update_traces(marker_line_width=0, row=1, col=2)
        fig.update_xaxes(title_text="AGE", row=1, col=2)
        fig.update_yaxes(title_text=clicked_col, row=1, col=2)

        fig.add_trace(
            px.violin(df_sorted, x='AGE_GROUP', y=clicked_col, labels={'x': 'AGE_GROUP', 'y': clicked_col},
                      color_discrete_sequence=px.colors.sequential.Viridis).data[0],
            row=2, col=1
        )
        fig.update_xaxes(title_text='AGE_GROUP', row=2, col=1)
        fig.update_yaxes(title_text=clicked_col, row=2, col=1)

        fig.add_trace(
            px.scatter(df_sorted, x='AGE', y=clicked_col, labels={'x': 'AGE', 'y': clicked_col},
                       color_discrete_sequence=px.colors.sequential.Viridis).data[0],
            row=2, col=2
        )

        fig.update_xaxes(title_text="AGE", row=2, col=2)
        fig.update_yaxes(title_text=clicked_col, row=2, col=2)
        return fig, clicked_col

    return {}, "Not selected"


@app.callback(
    Output('tab2_graph2', 'figure'),
    [Input('submit-IQR', 'n_clicks_timestamp'),
     Input('submit-Zscore', 'n_clicks_timestamp')],
    [State('selected-column', 'children')],
)
def update_graph2(iqr_click_time, zscore_click_time, selected_col):
    import plotly.express as px

    if iqr_click_time or zscore_click_time:
        df_sorted = df.sort_values(by='AGE')
        if (iqr_click_time or 0) > (zscore_click_time or 0):
            Q1 = df_sorted[selected_col].quantile(0.25)
            Q3 = df_sorted[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df_clean = df_sorted[(df_sorted[selected_col] > lower_bound) & (df_sorted[selected_col] < upper_bound)]

        else:
            mean = df_sorted[selected_col].mean()
            std = df_sorted[selected_col].std()
            zscore = (df_sorted[selected_col] - mean) / std
            df_clean = df_sorted[(zscore < 3) & (zscore > -3)]
        fig = make_subplots(
            rows=2, cols=2,
        )
        fig.add_trace(
            px.box(df_clean, y=selected_col, labels={'y': selected_col},
                   color_discrete_sequence=px.colors.sequential.Plasma).data[0],
            row=1, col=1,
        )

        fig.update_traces(width=1)
        fig.update_xaxes(title_text='AGE_GROUP', row=1, col=1)

        fig.add_trace(
            px.bar(df_clean, x='AGE', y=selected_col, labels={'x': 'AGE', 'y': selected_col},
                   color_discrete_sequence=px.colors.sequential.Plasma).data[0],
            row=1, col=2
        )
        fig.update_traces(marker_line_width=0, row=1, col=2)
        fig.update_xaxes(title_text="AGE", row=1, col=2)
        fig.update_yaxes(title_text=selected_col, row=1, col=2)

        fig.add_trace(
            px.violin(df_clean, x='AGE_GROUP', y=selected_col, labels={'x': 'AGE_GROUP', 'y': selected_col},
                      color_discrete_sequence=px.colors.sequential.Plasma).data[0],
            row=2, col=1
        )
        fig.update_xaxes(title_text='AGE_GROUP', row=2, col=1)
        fig.update_yaxes(title_text=selected_col, row=2, col=1)

        fig.add_trace(
            px.scatter(df_clean, x='AGE', y=selected_col, labels={'x': 'AGE', 'y': selected_col},
                       color_discrete_sequence=px.colors.sequential.Plasma).data[0],
            row=2, col=2
        )
        fig.update_xaxes(title_text="AGE", row=2, col=2)
        fig.update_yaxes(title_text=selected_col, row=2, col=2)
        return fig
    return {}


# tab3.py callbacks
@app.callback(
    [Output('tab3_graph1', 'figure'),
     Output('tab3_graph2', 'figure'),
     Output('tab3_graph3', 'figure')],
    [Input('submit-dimension', 'n_clicks')],
    [State('column-checklist', 'value'),
     State('dimension', 'value')]
)
def update_graph(n_clicks, selected_cols, dimension):
    import plotly.express as px
    from sklearn.decomposition import PCA
    if n_clicks:
        pca = PCA(n_components=dimension)
        pca_result = pca.fit_transform(df[selected_cols])
        pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i}' for i in range(1, dimension + 1)])

        fig3 = px.pie(pca_df, names=[f'PC{i}' for i in range(1, dimension + 1)],
                      values=pca.explained_variance_ratio_, hole=0.3, title='PCA Explained Variance')

        fig1 = px.scatter(pca_df, x='PC1', y='PC2', title='PCA 2D', size_max=2)

        if dimension >= 3:
            fig2 = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', title='PCA 3D', opacity=0.4
                                 )
            fig2.update_traces(marker=dict(size=3))
        else:
            fig2 = {}
        return fig1, fig2, fig3

    return {}, {}, {}


# tab4.py callbacks
@app.callback(
    Output('tab4_graph1', 'figure'),
    Input('submit-normality', 'n_clicks'),
    State('sampling-size', 'value'),
    State('radio-buttons', 'value'),
    State('radio-columns', 'value'),
)
def update_graph(n_clicks, size, method, selected_col):
    import plotly.express as px
    from scipy.stats import shapiro, ks_1samp, normaltest
    import numpy as np
    from scipy.stats import zscore

    np.random.seed(42)
    sample = np.random.choice(df[selected_col], size)

    if n_clicks:
        if method == '1':
            sample = zscore(sample)
            stat, p = ks_1samp(sample, scipy.stats.norm.cdf)
            method_name = 'Kolmogorov-Smirnov'
        elif method == '2':
            stat, p = shapiro(sample)
            method_name = 'Shapiro-Wilk'
        else:
            stat, p = normaltest(sample)
            method_name = "D'Agostino"

        fig = px.histogram(sample, title=f'{method_name} test, p-value: {p:.2f}')
        return fig

    return {}


# tab5.py callbacks
@app.callback(
    Output('table', 'data'),
    Output('tab5_graph1', 'figure'),
    Input('submit-transformation', 'n_clicks'),
    State('column-dropdown', 'value'),
    State('radio-buttons', 'value'),
)
def update_output(n_clicks, col, method):
    import plotly.express as px
    import numpy as np
    df_copy = df.copy()

    if n_clicks:
        if method == '1':
            df_copy[col] = (df_copy[col] - df_copy[col].min()) / (df_copy[col].max() - df_copy[col].min())
        elif method == '2':
            df_copy[col] = (df_copy[col] - df_copy[col].mean()) / df_copy[col].std()
        elif method == '3':
            df_copy[col] = np.log(df_copy[col][df_copy[col] > 0])
        elif method == '4':
            df_copy[col] = np.sqrt(df_copy[col][df_copy[col] >= 0])

        fig = px.histogram(df_copy, x=col, title=f'{method} transformation')
        return df_copy.round(2).sample(10).to_dict('records'), fig

    return df_copy.round(2).sample(10).to_dict('records'), {}


# tab6.py callbacks
@app.callback(
    [Output('uploaded_table', 'columns'),
     Output('uploaded_table', 'data')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'), ]
)
def update_output(contents, filename, ):
    import base64
    import io
    if contents is None:
        return [], []

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            df_uploaded = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df_uploaded = pd.read_excel(io.BytesIO(decoded))
        else:
            return [], []

        columns = [{"name": i, "id": i} for i in df_uploaded.columns]
        data = df_uploaded.to_dict('records')

        return columns, data

    except Exception as e:
        print(e)
        return [], []


@app.callback(
    Output("download-uploaded-file", "data"),
    Input("btn_download", "n_clicks"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def download_uploaded_file(n_clicks, contents, filename):
    import base64
    import io

    if contents is None:
        return dash.no_update

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            df_uploaded = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df_uploaded = pd.read_excel(io.BytesIO(decoded))
        else:
            return dash.no_update

        return dcc.send_data_frame(df_uploaded.to_csv, filename)
    except Exception as e:
        print(e)
        return dash.no_update


# tab7.py callbacks
@app.callback(
    Output('tab7_graph1', 'figure'),
    Input('submit_tab7_1', 'n_clicks'),
    State('column-dropdown1', 'value'),
    State('column-dropdown2', 'value'),
    State('radio-buttons-plots1', 'value'),
    State('range-slider', 'value')
)
def update_graph(n_clicks, col1, col2, plot_type, range_slider):
    import plotly.express as px
    fig = {}
    if n_clicks:
        start, end = range_slider
        df_range = df.iloc[start:end]
        df_range_sorted = df_range.sort_values(by=col1)

        if plot_type == '1':
            line_plot = df_range_sorted.groupby(col1, observed=False)[col2].mean().reset_index()
            fig = px.line(line_plot, x=col1, y=col2, title=f'Line plot of {col2} vs {col1}')
        elif plot_type == '2':
            fig = px.histogram(df_range_sorted, x=col1, y=col2, histfunc='count',
                               title=f'Count plot of {col2} vs {col1}')
        elif plot_type == '3':
            import plotly.figure_factory as ff
            fig = ff.create_distplot([df_range_sorted[col1].tolist()], group_labels=[col1], show_hist=True,
                                     show_curve=True)
            fig.update_layout(title=f'Dist plot of {col1}')
        elif plot_type == '4':
            correlation_matrix_selected = df_range_sorted[[col1, col2]].corr()
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix_selected.values,
                x=correlation_matrix_selected.columns,
                y=correlation_matrix_selected.index,
                colorscale='Viridis'
            ))
            fig.update_layout(title=f'Heatmap of {col1} and {col2}')
        elif plot_type == '5':
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_range_sorted[col1],
                y=df_range_sorted[col2],
                fill='tozeroy',
                mode='none'
            ))
            fig.update_layout(title=f'Area plot of {col2} vs {col1}')
        elif plot_type == '6':
            fig = px.strip(df_range_sorted, x=col1, y=col2, title=f'Strip plot of {col2} vs {col1}')
        return fig
    return {}


@app.callback(
    Output('memoed', 'children'),
    Input('submit_tab7_3', 'n_clicks'),
    State('textarea', 'value')
)
def update_memo(n_clicks, value):
    if n_clicks:
        return value
    return ""


if __name__ == '__main__':
    app.run_server(debug=True)
