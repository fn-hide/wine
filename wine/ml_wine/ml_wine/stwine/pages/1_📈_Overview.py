import time
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

st.set_page_config(page_title="Overview", page_icon="📊")

st.markdown("# Overview")
st.sidebar.header("Overview")
st.write(
    """This page shows an overview of Wine Quality dataset. Enjoy!"""
)


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return pd.read_csv('ml_wine/data/train.csv')


@st.cache_data
def get_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = np.round(df.corr(), 2)
    mask = np.triu(np.ones_like(corr, dtype = bool))
    c_mask = np.where(~mask, corr, 100)

    c = []
    for i in c_mask.tolist()[1:]:
        c.append([x for x in i if x != 100])

    fig = ff.create_annotated_heatmap(
        z=c[::-1],
        x=corr.index.tolist()[:-1],
        y=corr.columns.tolist()[1:][::-1],
        colorscale='bluyl',
    )

    fig.update_layout(
        title = {'text': '<b>Feature Correlation <br> <sup>Heatmap</sup></b>'},
        height = 650, width = 650,
        margin = dict(t=210, l = 80),
        template = 'simple_white',
        yaxis = dict(autorange = 'reversed'),
    )

    fig.add_trace(
        go.Heatmap(
            z=c[::-1],
            colorscale='bluyl',
            showscale=True,
            visible=False,
        )
    )
    
    fig.data[1].visible = True

    return fig


@st.cache_data
def get_3d_scatterplot(df: pd.DataFrame, x: str, y: str, z: str, opacity=.8, width_line=.1) -> go.Figure:   
    '''
    This function takes a dataframe and different attributes to build a 3D scatterplot
    
    '''
    
    fig = px.scatter_3d(
        df, 
        x=df[x], 
        y=df[y], 
        z=df[z],
        template="plotly",
        color_discrete_sequence=px.colors.qualitative.Prism,
    ).update_traces(
        marker={
            "size": 4,
            "opacity": opacity,
            "line": {
                "width": width_line,
                "color": "white",
            }
        }
    ).update_layout(
        title={
            'text': f'<b>3D Scatterplot <br> <sup>{x} x {y} x {z}</sup></b>',
            'xanchor': 'left',
            'x': 0.05,
        },
        width=1000, 
        height=800, 
        autosize=False, 
        showlegend=True,
        legend=dict(
            title_font_family="Times New Roman",
            font=dict(size= 20)
        ),
        scene=dict(
            xaxis=dict(title=x, titlefont_color='black'),
            yaxis=dict(title=y, titlefont_color='black'),
            zaxis=dict(title=z, titlefont_color='black')
        ),
        font=dict(family="Gilroy", color='black', size=15),
    )
    
    return fig


# status_text = st.sidebar.empty()
# progress_bar = st.sidebar.progress(0)

# status_text.text("%i%% Complete" % i)
# progress_bar.progress(i)

df = get_dataset()
df = df.drop(columns='Id')
n_samples, n_features = df.shape

col1, col2 = st.columns(2)
col1.metric(label="Samples", value=f"{n_samples}")
col2.metric(label="Features", value=f"{n_features}")

st.dataframe(df.head(), hide_index=True)

'''## Statistics'''
st.dataframe(df.describe().T)
'''📝 We have different scales across the attributes. We may benefit from using rescaling methods, such as StandardScaling.'''

'''## Visualization'''
fig = get_correlation_heatmap(df)
st.plotly_chart(fig, use_container_width=True)
'''📝 The highest correlated feature with the target variable quality is alcohol, with 0.48 correlation.'''
'''📝 The highest positive correlation is between citric acid and fixed acidity, at 0.7.'''
'''📝 The highest negative correlation is between ph and fixed acidity, at -0.67.'''

st.divider()
col1, col2, col3 = st.columns(3)
x = col1.selectbox('X', options=df.columns.tolist(), index=0)
y = col2.selectbox('Y', options=df.columns.tolist(), index=1)
z = col3.selectbox('Z', options=df.columns.tolist(), index=2)

fig = get_3d_scatterplot(df, x, y, z)
st.plotly_chart(fig, use_container_width=True)
'''📝 In the 3D Scatterplot above, we can see the relationships between citric acid, fixed acidity, and density, which are highy-correlated features.'''
