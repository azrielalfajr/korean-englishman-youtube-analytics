import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import re
import streamlit as st
import altair as alt
from datetime import datetime
from googleapiclient.discovery import build
from googletrans import Translator
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from scipy.stats import pearsonr, ttest_ind
from collections import Counter
import nltk
from nltk.corpus import stopwords
st.set_option('deprecation.showPyplotGlobalUse', False)

import warnings
warnings.filterwarnings("ignore")

@st.cache_data
################ DEF FUNCTIONS ################
# function to load data
def load_data(file_path):
    return pd.read_csv(file_path)
    
# Function to ensure valid date range selection
def get_valid_date_range(min_date, max_date, date_range):
    start_date = min_date if min_date in date_range else date_range[0]
    end_date = max_date if max_date in date_range else date_range[-1]
    return start_date, end_date

def style_negative(v, props=''):
    """ Style negative values in dataframe"""
    try: 
        return props if v < 0 else None
    except:
        pass
    
def style_positive(v, props=''):
    """Style positive values in dataframe"""
    try: 
        return props if v > 0 else None
    except:
        pass  

def set_transparent_background(ax):
    """Set transparent background and white axes for a given axis."""
    fig = ax.get_figure()
    fig.patch.set_alpha(0)  # Transparent figure background
    ax.patch.set_alpha(0)   # Transparent axis background

    # Set axis color to white
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

################ SET PAGE CONFIGURATION ################
st.set_page_config(
    page_title = "Korean Englishman Youtube Analytics Dashboard",
    page_icon = "📈",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# alt.themes.enable('urbaninstitute')

# create custom palette
custom_palette = ['#FABEC0','#F85C70','#F37970','#E43D40']
    
################ LOAD DATA ################
kre_df = load_data('KoreanEnglishman-YoutubeData-Final.csv')
kre_df['publishedAt'] = pd.to_datetime(kre_df['publishedAt'])
kre_df['EngageRatio'] = kre_df['EngageRatio'] * 100


####################################################################################################################
######################################### START BUILDING DASHBOARD IN STREAMLIT ####################################
####################################################################################################################

# Add sidebar
with st.sidebar:
    st.title("Korean Englishman Youtube Analytics Dashboard")
    st.image('channels4_banner.jpg')
    add_sidebar = st.selectbox('Content Type', ('All', 'Longs', 'Shorts'))
    
    # Generate date range for the slider
    min_date = pd.Timestamp('2019-07-13').date()
    max_date = pd.to_datetime(kre_df['publishedAt']).max().date()
    date_range = pd.date_range(start=min_date, end=max_date, freq='D').date
    
    # Initial date range selection
    start_date, end_date = get_valid_date_range(min_date, max_date, date_range)
    
    # Menambahkan slider untuk memilih rentang tanggal berdasarkan jenis konten
    if add_sidebar in ['All', 'Longs']:
        start_date, end_date = st.sidebar.select_slider(
            'Select Date Range',
            options=date_range,
            value=(start_date, end_date)
        )
    elif add_sidebar == 'Shorts':
        min_date = pd.Timestamp('2023-07-01').date()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D').date
        start_date, end_date = get_valid_date_range(min_date, max_date, date_range)
        start_date, end_date = st.sidebar.select_slider(
            'Select Date Range',
            options=date_range,
            value=(start_date, end_date)
        )
    
    st.sidebar.write('Date Selected', start_date, 'to', end_date)

    # Filter data based on the selected date range and content type
    kre_filtered_df = kre_df[(pd.to_datetime(kre_df['publishedAt']).dt.date >= start_date) & 
                             (pd.to_datetime(kre_df['publishedAt']).dt.date <= end_date)]
    
    if add_sidebar == 'All':
        kre_filtered_df = kre_filtered_df
    elif add_sidebar == 'Longs':
        kre_filtered_df = kre_filtered_df[kre_filtered_df['ContentType'] == 'Longs']
    elif add_sidebar == 'Shorts':
        kre_filtered_df = kre_filtered_df[(pd.to_datetime(kre_filtered_df['publishedAt']).dt.date >= start_date) & 
                                          (pd.to_datetime(kre_filtered_df['publishedAt']).dt.date <= end_date) & 
                                          (kre_filtered_df['ContentType'] == 'Shorts')]

# Display metrics based on the filtered data
df_agg_metrics_tosum = kre_filtered_df[['publishedAt', 'Views', 'Likes', 'Comments']]
df_agg_metrics_tomean = kre_filtered_df[['publishedAt', 'DurationSec', 'EngageRatio']]

metric_tosum = df_agg_metrics_tosum.sum(numeric_only=True)
metric_tomean = df_agg_metrics_tomean.mean(numeric_only=True)
metric_combined = pd.concat([metric_tosum, metric_tomean])

# Display metrics in Streamlit
col1, col2, col3, col4, col5 = st.columns(5)
columns = [col1, col2, col3, col4, col5]

labels = ['Total Views', 'Total Likes', 'Total Comments', 'Avg Duration (s)', 'Avg EngageRatio (%)']
metrics = ['Views', 'Likes', 'Comments', 'DurationSec', 'EngageRatio']
units = ['B', 'M', 'M', '', '']

for col, label, metric, unit in zip(columns, labels, metrics, units):
    value = metric_combined[metric]
    if unit == 'B':
        value = f"{value / 1_000_000_000:.2f}B"
    elif unit == 'M':
        value = f"{value / 1_000_000:.2f}M"
    else:
        value = f"{value:.1f}"
    with col:
        with st.container(border=True, height=110):
            st.metric(label=label, value=value)

# define the layout of the dashboard
col = st.columns((4.5, 4.5), gap='medium')

with col[0]:
    with st.container(border=True):
        st.write("Most Covered Topics on This Channel")
        all_titles = ''.join(title for title in kre_filtered_df['TranslatedTitle'])
        wordcloud = WordCloud(width=800, height=600, colormap='Reds').generate(all_titles)
    
        plt.figure(figsize=(8, 6))  # Create a new figure for wordcloud
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

    with st.container(border=True):
        st.write('Trends in Views Over Time')
        views_df = kre_filtered_df.query("publishedAt >= '2021-01-01'")
        fig, ax = plt.subplots(figsize=(12, 3))  # Create a new figure for line plot
        sns.lineplot(data=views_df, x=views_df['publishedAt'].dt.month, y='Views', hue=views_df['publishedAt'].dt.year, errorbar=None, palette=custom_palette)
        plt.ylabel('Views (in 10 Millions)')
        plt.xlabel('Month')
        plt.legend(title='Year')
        set_transparent_background(ax)
        st.pyplot(plt)
    with st.container(border=True):
        st.write('Most Engaged Topics')
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        clean_text = re.sub(r'[!?+.]+', '', all_titles)
        words = clean_text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        # Count the words
        word_freq = Counter(filtered_words)
        
        # Extract the top 20 most common words
        top_words = word_freq.most_common(20)
        top_words = [(word.title(), count) for word, count in top_words]
    
        for word, _ in top_words:
            kre_filtered_df[word] = kre_filtered_df['TranslatedTitle'].apply(lambda x: word.lower() in x.lower())
    
        word_engagement = {}
        for word, _ in top_words:
            word_df = kre_filtered_df[kre_filtered_df[word] == True]
            word_engagement[word] = word_df['EngageRatio'].mean()
    
        word_engagement_df = pd.DataFrame(list(word_engagement.items()), columns=['Word', 'EngageRatio'])
        word_engagement_df = word_engagement_df.sort_values(by='EngageRatio', ascending=False)
    
        # Create a new figure for bar plot
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(data=word_engagement_df, x='Word', y='EngageRatio', palette='Reds')
        ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
        set_transparent_background(ax)
        st.pyplot(plt)


with col[1]:
    with st.container(border=True, height=425):
        st.write('Component of Videos')
        pie_df = kre_filtered_df['ContentType'].value_counts()
        fig,ax = plt.subplots(figsize=(4,4))
        plt.pie(pie_df, labels=pie_df.index, colors=['#cb181d','#fcbba1'], autopct='%.2f%%', textprops={'color': 'white'}, startangle=90, pctdistance=0.6, labeldistance=1.389)
        set_transparent_background(ax)
        # plt.tight_layout()
        st.pyplot(plt)
    
    with st.container(border=True):
        st.write('Engagement Ratio Over Time')
        engage_df = kre_filtered_df.query("publishedAt >= '2021-01-01'")
        fig, ax = plt.subplots(figsize=(12,3))
        sns.lineplot(data=engage_df, x=engage_df['publishedAt'].dt.month, y='EngageRatio', hue=engage_df['publishedAt'].dt.year, errorbar=None, palette=custom_palette)
        plt.ylabel('Engagement Ratio (%)')
        plt.xlabel('Month')
        plt.legend(title='year')
        set_transparent_background(ax)
        st.pyplot(plt)
    
    with st.container(border=True, height=320):
        st.write('About')
        st.write('''
        This dashboard provides insights and analytics on the YouTube channel **Korean Englishman**. 
        The data used in this dashboard is obtained using the **YouTube DATA API v3**. 
        You can explore various metrics and trends related to the channel's performance over time.
        
        - **Data Source**: [Korean Englishman Final Dataset](<https://drive.google.com/file/d/1VpbFf4yAZb6K8fS5uTRd4JIjUelW3mKM/view?usp=sharing>).
        - **Metrics**: Views, Likes, Comments, Engagement Ratio, etc.
        - **Analysis**: Trends in views and engagement, word cloud of common topics, etc.
    ''')