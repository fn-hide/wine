# Importing Libraries

# Data Handling
import pandas as pd
import numpy as np

# Data Visualization
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from IPython.display import display
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

# Statistics & Mathematics
from scipy.stats import shapiro, skew

# Model Selection for Cross Validation
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer, FunctionTransformer
from sklearn.cluster import KMeans

# Machine Learning metrics
from sklearn.metrics import cohen_kappa_score

import random

# Hiding warnings 
import warnings
warnings.filterwarnings("ignore")

# Defining seed and the template for plots
seed = 42
plotly_template = 'simple_white'


def dataframe_description(df):
    """
    This function prints some basic info on the dataset.
    """
    categorical_features = []
    continuous_features = []
    binary_features = []
    
    for col in df.columns:
        if df[col].dtype == object:
            categorical_features.append(col)
        else:
            if df[col].nunique() <= 2:
                binary_features.append(col)
            else:
                continuous_features.append(col)
    
    print("\n{} shape: {}".format(type(df).__name__, df.shape))
    print("\n{:,.0f} samples".format(df.shape[0]))
    print("\n{:,.0f} attributes".format(df.shape[1]))
    print(f'\nMissing Data: \n')
    print(df.isnull().sum())
    print(f'\nDuplicates: {df.duplicated().sum()}')
    print(f'\nData types: \n')
    print(df.dtypes)
    print(f'\nCategorical features: \n')
    if len(categorical_features) == 0:
        print('No Categorical Features')
    else:
        for feature in categorical_features:
            print(feature)
    print(f'\nContinuous features: \n')
    if len(continuous_features) == 0:
        print('No Continuous Features')
    else:
        for feature in continuous_features:
            print(feature)
    print(f'\nBinary features: \n')
    if len(binary_features) == 0:
        print('No Binary Features')
    else:
        for feature in binary_features:
            print(feature)
    print(f'\n{type(df).__name__} Head: \n')
    display(df.head(5))
    print(f'\n{type(df).__name__} Tail: \n')
    display(df.tail(5))
    

def plot_correlation(df):
    '''
    This function is resposible to plot a correlation map among features in the dataset
    '''
    corr = np.round(df.corr(), 2)
    mask = np.triu(np.ones_like(corr, dtype = bool))
    c_mask = np.where(~mask, corr, 100)

    c = []
    for i in c_mask.tolist()[1:]:
        c.append([x for x in i if x != 100])
    
    fig = ff.create_annotated_heatmap(z=c[::-1],
                                      x=corr.index.tolist()[:-1],
                                      y=corr.columns.tolist()[1:][::-1],
                                      colorscale = 'bluyl')

    fig.update_layout(title = {'text': '<b>Feature Correlation <br> <sup>Heatmap</sup></b>'},
                      height = 650, width = 650,
                      margin = dict(t=210, l = 80),
                      template = 'simple_white',
                      yaxis = dict(autorange = 'reversed'))

    fig.add_trace(go.Heatmap(z = c[::-1],
                             colorscale = 'bluyl',
                             showscale = True,
                             visible = False))
    fig.data[1].visible = True

    fig.show()


def describe(df):
    '''
    This function plots a table containing Descriptive Statistics of the Dataframe
    '''
    mean_features = df.mean().round(2).apply(lambda x: "{:,.2f}".format(x)) 
    std_features = df.std().round(2).apply(lambda x: "{:,.2f}".format(x)) 
    q1 = df.quantile(0.25).round(2).apply(lambda x: "{:,.2f}".format(x))
    median = df.quantile(0.5).round(2).apply(lambda x: "{:,.2f}".format(x))
    q3 = df.quantile(0.75).round(2).apply(lambda x: "{:,.2f}".format(x))


    # Generating new Dataframe
    describe_df = pd.DataFrame({'Feature Name': mean_features.index,
                                'Mean': mean_features.values,
                                'Standard Deviation': std_features.values,
                                '25%': q1.values,
                                'Median': median.values,
                                '75%': q3.values})

    # Generating a Table w/ Pyplot
    fig = go.Figure(data = [go.Table(header=dict(values=list(describe_df.columns),
                                                 align = 'center',
                                                 fill_color = 'midnightblue',
                                               font=dict(color = 'white', size = 18)),
                                     cells=dict(values=[describe_df['Feature Name'],
                                                        describe_df['Mean'],
                                                        describe_df['Standard Deviation'],
                                                       describe_df['25%'],
                                                       describe_df['Median'],
                                                       describe_df['75%']],
                                                fill_color = 'gainsboro',
                                                align = 'center'))
                           ])

    fig.update_layout(title = {'text': f'<b>Descriptive Statistics of the Dataframe<br><sup> (Mean, Standard Deviation, 25%, Median, and 75%)</sup></b>'},
                      template = plotly_template,
                      height = 700, width = 950,
                      margin = dict(t = 100))

    fig.show()


def plot_distplot(df, x):  
    '''
    This function creates a distribution plot for continuous variables
    '''
    
    feature = df[x]

    fig = ff.create_distplot([feature], [x], show_hist=False)

    fig.update_layout(
        title={'text': f'<b>Distplot <br> <sup>{x}</sup></b>',
               'xanchor': 'left',
               'x': 0.05},
        height=600,
        width=1000,
        margin=dict(t=100),
        template= plotly_template,
        showlegend=True
    )

    fig.show()


def plot_histogram_matrix(df):
    
    '''
    This function identifies all continuous features within the dataset and plots
    a matrix of histograms for each attribute
    '''
    
    continuous_features = []
    for feat in df.columns:
        if df[feat].nunique() > 2:
            continuous_features.append(feat)
    num_cols = 2
    num_rows = (len(continuous_features) + 1) // num_cols

    fig = make_subplots(rows=num_rows, cols=num_cols)

    for i, feature in enumerate(continuous_features):
        row = i // num_cols + 1
        col = i % num_cols + 1

        fig.add_trace(
            go.Histogram(
                x=df[feature],
                name=feature
            ),
            row=row,
            col=col
        )

        fig.update_xaxes(title_text=feature, row=row, col=col)
        fig.update_yaxes(title_text='Frequency', row=row, col=col)
        fig.update_layout(
            title=f'<b>Histogram Matrix<br> <sup> Continuous Features</sup></b>',
            showlegend=False
        )

    fig.update_layout(
        height=350 * num_rows,
        width=1000,
        margin=dict(t=100, l=80),
        template= plotly_template
    )

    fig.show()


def plot_boxplot_matrix(df):
    
    '''
    This function identifies all continuous features within the dataset and plots
    a matrix of boxplots for each attribute
    '''
    
    continuous_features = []
    for feat in df.columns:
        if df[feat].nunique() > 2:
            continuous_features.append(feat)
    
    num_cols = 2
    num_rows = (len(continuous_features) + 1) // num_cols


    fig = make_subplots(rows=num_rows, cols=num_cols)


    for i, feature in enumerate(continuous_features):
        row = i // num_cols + 1
        col = i % num_cols + 1

        fig.add_trace(
            go.Box(
                x=df[feature],
                name = ' '
            ),
            row=row,
            col=col
        )

        fig.update_yaxes(title_text = ' ', row=row, col=col)
        fig.update_xaxes(title_text= feature, row=row, col=col)
        fig.update_layout(
            title=f'<b>Boxplot Matrix<br> <sup> Continuous Features</sup></b>',
            showlegend=False,
            yaxis=dict(
            tickangle=-90  
        )
        )

    fig.update_layout(
        height=350 * num_rows,
        width=1000,
        margin=dict(t=100, l=80),
        template= plotly_template
    )


    fig.show()


def scatterplot(df, x, y):
    '''
    This function takes a dataframe and X and y axes to plot a scatterplot
    '''

    color_dict = {
        0: 'orange',
        1: 'blue',
        2: 'green',
        3: 'red',
        4: 'black',
        5: 'purple',
        6: 'pink',
        7: 'brown',
        8: 'teal',
        9: 'magenta',
        10: 'cyan',
        11: 'olive',
        12: 'navy',
        13: 'indigo',
        14: 'maroon',
        15: 'turquoise',
        16: 'silver',
        17: 'gold'
    }
    
    color_index = random.choice(list(color_dict.keys()))
    color = color_dict[color_index]

    fig = px.scatter(df, y=y, x=x)
    fig.update_traces(marker=dict(size=10, color=color))
    fig.update_layout(
        title={'text': f'<b>Scatterplot <br> <sup>{x} x {y}</sup></b>'},
        height=750,
        width=850,
        margin=dict(t=80, l=80),
        template= plotly_template
    )
    fig.show()


def clustered_scatterplot(df, y, x, cluster):
    '''
    This function takes a dataframe, x, and y axes to plot a scatterplot colored accordingly to clusters
    It also prints a count of values for each cluster
    '''
    fig = px.scatter(df,
                     y = y,
                     x = x,
                     color = cluster, symbol = cluster)

    fig.update_traces(marker = dict(size = 10))

    fig.update(layout_coloraxis_showscale=False)

    fig.update_layout(title = {'text': f'<b>Clustered Scatterplot <br> <sup> {y} x {x} </sup></b>',
                              'xanchor': 'left',
                              'x': 0.05},
                     height = 600, width = 700,
                     margin = dict(t=100),
                     template = plotly_template,
                     showlegend = True)

    fig.show()

    print('Cluster Count:')
    print(f'{df[cluster].value_counts()}')


def barplot(df, feat):    
    
    '''
    This function is supposed to organize the n top value counts of any attribute and plot a Barplot
    '''
    
    counts = df[feat].value_counts()
    fig = px.bar(y=counts.values, 
                 x=counts.index, 
                 color = counts.index,
                 text=counts.values)

    fig.update_layout(title=f'<b>Frequency of values in {feat}<br> <sup> Barplot</sup></b>',
                      xaxis=dict(title=f'{feat}'),
                      yaxis=dict(title='Count'),
                      legend=dict(title=f'{feat}'),
                      showlegend=True,
                      height=600,
                      width=1000,
                      margin=dict(t=100, l=80),
                      template= plotly_template)
    fig.show()


def shapiro_wilk_test(df):
    '''
    This function performs a Shapiro-Wilk test to check if the data is normally distributed or not, as well as skewness
    '''
    print(f'\033[1mShapiro-Wilk Test & Skewness:\033[0m')
    print('\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  \n')

    numeric_columns = df.select_dtypes(include=['float', 'int']).columns

    for feature in numeric_columns:
        stats, p_value = shapiro(df[feature])

        if p_value < 0.05:
            text = f'{feature} Does Not Seem to be Normally Distributed'
        else:
            text = f'{feature} Seems to be Normally Distributed'

        print(f'{feature}')
        print(f'\n  Shapiro-Wilk Statistic: {stats:.2f}')
        print(f'\n  Shapiro-Wilk P-value: {p_value}')
        print(f'\n  Skewness: {np.round(skew(df[feature]), 2)}')
        print(f'\n  Conclusion: {text}')
        print('\n===============================================================================================')

    print('\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  \n')
    print(f'\033[1mEnd of Shapiro-Wilk Test\033[0m')


def boxplot(df, y, x, color):    
    '''
    This function plots a Y and X boxplot
    '''
    fig = px.box(df, y= y , x = x, color= color)

    fig.update_layout(title=f'<b>Boxplot<br> <sup> {y} by {x}</sup></b>',
                      showlegend=False,
                      yaxis=dict(tickangle= -45),
                      height=600,
                      width=1000,
                      margin=dict(t=100, l=80),
                      template= plotly_template)

    fig.show()


def pred_vs_true_plot(y_true, y_pred):
    '''
    This function takes values for y_true and y_val, and plots a scatterplot along with a line of best fit
    '''

    slope, intercept = np.polyfit(y_true, y_pred, 1)
    fit_line = slope * y_true + intercept

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name='Data Points'))
    fig.add_trace(go.Scatter(x=y_true, y=fit_line, mode='lines', line=dict(color='red'), name='Fit-line'))
    fig.update_traces(marker=dict(size=10, color='blue'))
    fig.update_layout(
        title={'text': f'<b>True x Predicted <br> <sup>Scatterplot</sup></b>'},
        xaxis=dict(title='True Salaries'), 
        yaxis=dict(title='Predicted Salaries'),
        height=750,
        width=850,
        margin=dict(t=250, l=80),
        template= plotly_template,
    )
    fig.show()


def three_axes_scatterplot(df, x, y, z):   
    
    '''
    This function takes a dataframe and different attributes to build a 3D scatterplot
    
    '''
    
    scatterplot = go.Scatter3d(
        x= df[x],
        y= df[y],
        z= df[z],  
        mode='markers')

    fig = go.Figure(data=scatterplot)
    fig.update_layout(
        title={'text': f'<b>3D Scatterplot <br> <sup>{x} x {y} x {z}</sup></b>',
               'xanchor': 'left',
               'x': 0.05},
        height=600,
        width=700,
        margin=dict(t=100),
        template= plotly_template,
        showlegend=True
    )

    
    fig.show()


def violin_boxplot(df, y, x, color):    
    '''
    This function plots a Y and X ridgeline plot
    '''
    
    fig = px.violin(df, y=y, x=x, color=color, box=True, points= 'all')

    fig.update_layout(title=f'<b>Violin Boxplot<br> <sup>{x} by {y}</sup></b>',
                      showlegend=False,
                      yaxis=dict(tickangle=-45),
                      height=600,
                      width=1000,
                      margin=dict(t=100, l=80),
                      template= plotly_template)

    fig.show()


def individual_boxplot(df, x):    
    fig = px.box(df, x = x)

    fig.update_layout(title=f'<b>Boxplot<br> <sup> {x}</sup></b>',
                      showlegend=False,
                      yaxis=dict(tickangle= -45),
                      height=400,
                      width=1000,
                      margin=dict(t=100, l=80),
                      template= plotly_template)

    fig.show()


def elbow_curve(wss):    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = list(range(1,10)),
                            y = wss,
                            mode = 'lines+markers',
                            marker = dict(color = 'midnightblue'),
                            name = 'WSS'))

    
    fig.update_layout(title = {'text': '<b>Elbow Curve Plot <br> <sup>Within-Cluster Sum of Squares</sup></b>'},
                     height = 400, width = 950,
                     xaxis_title = 'Number of Clusters',
                     yaxis_title = 'Within-Cluster Sum of Squares (WSS)',
                     margin = dict(t=80),
                     template = plotly_template)

    fig.show()


def split_train_test(df, test_size, seed):
    
    '''
    This function splits a dataframe for training and testing according to test_size
    '''
    
    train, test = train_test_split(df, test_size = test_size, shuffle = True, random_state = seed) # Splitting data

    print(f'\n Train shape: {train.shape}\n')
    print(f'\n {len(train)} Samples \n')
    print(f'\n {len(train.columns)} Attributes \n')
    display(train.head(10))
    print('\n' * 2)

    print(f'\n Test shape: {test.shape:}\n')
    print(f'\n {len(test)} Samples \n')
    print(f'\n {len(test.columns)} Attributes \n')
    display(test.head(10))
    
    return train, test


def X_y_split(df, target_variable):
    
    '''
    This function takes a dataframe and a target variable to create an X (predictors) dataframe and a y Series
    '''
    
    X, y = df.drop([target_variable], axis = 1), df[target_variable] 

    #Printing info on X and y
    print(f'\nX shape: {X.shape}\n')
    print(f'\n{len(X)} Samples \n')
    print(f'\n{len(X.columns)} Attributes \n')
    display(X.head(10))
    print('\n')
    print(f'\ny shape: {y.shape}\n')
    print(f'\n{len(y)} Samples \n')
    display(y.head(10))
    
    return X, y


def quadratic_weighted_kappa(y_true, y_pred):
    '''
    This function returns the evaluation metric of this competition
    '''
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def feat_eng(df):
    df.columns = df.columns.str.replace(' ', '_')
    df['total_acidity'] = df['fixed_acidity'] + df['volatile_acidity'] + df['citric_acid']
    df['acidity_to_pH_ratio'] = df['total_acidity'] / df['pH']
    df['free_sulfur_dioxide_to_total_sulfur_dioxide_ratio'] = df['free_sulfur_dioxide'] / df['total_sulfur_dioxide']
    df['alcohol_to_acidity_ratio'] = df['alcohol'] / df['total_acidity']
    df['residual_sugar_to_citric_acid_ratio'] = df['residual_sugar'] / df['citric_acid']
    df['alcohol_to_density_ratio'] = df['alcohol'] / df['density']
    df['total_alkalinity'] = df['pH'] + df['alcohol']
    df['total_minerals'] = df['chlorides'] + df['sulphates'] + df['residual_sugar']
    
    # Cleaning inf or null values that may result from the operations above
    df = df.replace([np.inf, -np.inf], 0)
    df = df.dropna()
    
    return df


# Applying QuantileTransformer to change the distribution to a gaussian-like distribution
class CustomQuantileTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=self.random_state)

    def fit(self, X_train, y=None):
        self.quantile_transformer.fit(X_train)
        return self

    def transform(self, X):
        X_transformed = self.quantile_transformer.transform(X)
        X = pd.DataFrame(X_transformed, columns=X.columns)
        return X


# Applying StandardScaler to bring every feature to the same scale
class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X_train, y=None):
        self.scaler.fit(X_train)
        return self

    def transform(self, X):
        X_transformed = self.scaler.transform(X)
        X = pd.DataFrame(X_transformed, columns=X.columns)
        return X
    

# Applying KMeans clustering with n_clusters = 3

class KMeansTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, n_clusters=3, random_state=seed):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
   
    def fit(self, X_train, y=None):
        self.kmeans.fit(X_train)
        return self
    
    def transform(self, X):
        X_clustered = pd.DataFrame(X.copy())
        cluster_labels = self.kmeans.predict(X)
        X_clustered['Cluster'] = cluster_labels
        return X_clustered
