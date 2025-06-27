import streamlit as st
import pandas as pd # type: ignore
import plotly.express as px # type: ignore
import ast
from wordcloud import WordCloud, STOPWORDS # type: ignore
import matplotlib.pyplot as plt # type: ignore
import requests # type: ignore
from io import StringIO

# --- Color Palette ---
# A more professional and visually appealing color sequence
PALETTE = px.colors.qualitative.Vivid
# Specific color mapping for sentiment for consistency across charts
SENTIMENT_COLOR_MAP = {
    'Very Positive': '#00A000',
    'Positive': '#66C2A5',
    'Neutral': '#BDBDBD',
    'Negative': '#F4A582',
    'Very Negative': '#D6604D'
}
SENTIMENT_ORDER = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]


# --- Page Config ---
st.set_page_config(layout="wide", page_title="News Analysis Report (Jan-Apr 2025)", initial_sidebar_state="collapsed")

# --- Helper Functions ---
def safe_literal_eval(val):
    """Safely evaluate a string representation of a Python literal (list, dict)."""
    if pd.isna(val):
        return [] # Return empty list for NaN/None, suitable for tag columns
    if isinstance(val, (list, dict)):
        return val # Already parsed
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return [] # Return empty list if parsing fails

def safe_literal_eval_dict(val):
    """Safely evaluate a string representation of a Python dictionary."""
    if pd.isna(val):
        return {} # Return empty dict for NaN/None
    if isinstance(val, dict):
        return val # Already parsed
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return {} # Return empty dict if parsing fails

@st.cache_data # Cache the data loading and preprocessing
def load_and_preprocess_data(file_url): # Renamed parameter for clarity
    """Loads and preprocesses the news analysis data from a URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        
        response = requests.get(file_url, headers=headers, timeout=30) # Added timeout
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        
        # Decode response text with UTF-8 and handle errors
        response_text = response.content.decode('utf-8', 'replace')

        # Use StringIO to treat the string data as a file
        csv_data = StringIO(response_text)
        df = pd.read_csv(csv_data)
        
        
    except requests.exceptions.HTTPError as errh:
        st.error(f"Http Error connecting to data source: {errh}")
        st.error(f"URL: {file_url}")
        st.error(f"Response status code: {errh.response.status_code if errh.response else 'N/A'}")
        st.error(f"Response text: {errh.response.text[:500] if errh.response else 'No response text'}...")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except requests.exceptions.ConnectionError as errc:
        st.error(f"Error Connecting to data source: {errc}")
        st.error(f"URL: {file_url}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except requests.exceptions.Timeout as errt:
        st.error(f"Timeout Error connecting to data source: {errt}")
        st.error(f"URL: {file_url}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except requests.exceptions.RequestException as err:
        st.error(f"An error occurred with the data request: {err}")
        st.error(f"URL: {file_url}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except pd.errors.EmptyDataError:
        st.error(f"The CSV file from {file_url} is empty or not valid CSV.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        # Catch any other errors during pandas processing or initial data load
        st.error(f"An unexpected error occurred while loading data from {file_url}: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # --- Start of your existing preprocessing logic ---
    try:
        # Convert date columns
        df['sentiment_created_at'] = pd.to_datetime(df['sentiment_created_at'], errors='coerce')
        df['article_publish_date'] = pd.to_datetime(df['article_publish_date'], errors='coerce')

        # Parse stringified lists/dicts
        # Ensure your safe_literal_eval functions are robust to NaNs or unexpected inputs
        df['secondary_category_tag_names_list'] = df['secondary_category_tag_names'].apply(safe_literal_eval)
        df['secondary_emotional_impact_tag_names_list'] = df['secondary_emotional_impact_tag_names'].apply(safe_literal_eval)
        df['version_info_dict'] = df['version_info'].apply(safe_literal_eval_dict)

        # Handle potential NaNs in primary tags (replace with 'Unknown' or drop)
        df['primary_category_tag_name'] = df['primary_category_tag_name'].fillna('Unknown')
        df['primary_emotional_impact_tag_name'] = df['primary_emotional_impact_tag_name'].fillna('Unknown')
        
        # Map sentiment labels to numerical values for trend analysis
        sentiment_mapping = {
            'Very Negative': -2, 'Negative': -1, 'Neutral': 0, 'Positive': 1, 'Very Positive': 2
        }
        df['sentiment_score'] = df['sentiment_label'].map(sentiment_mapping)

        # Explode secondary categories for co-occurrence analysis
        df_secondary_categories_exploded = df.explode('secondary_category_tag_names_list')
        # Ensure the column exists before renaming, or use try-except for robustness
        if 'secondary_category_tag_names_list' in df_secondary_categories_exploded.columns:
            df_secondary_categories_exploded.rename(columns={'secondary_category_tag_names_list': 'secondary_category'}, inplace=True)
        else: # Handle case where explode might not create the column as expected (e.g. if original was all NaN/empty)
            df_secondary_categories_exploded['secondary_category'] = pd.NA 
        df_secondary_categories_exploded['secondary_category'] = df_secondary_categories_exploded['secondary_category'].fillna('None')


        # Explode secondary emotions for co-occurrence analysis
        df_secondary_emotions_exploded = df.explode('secondary_emotional_impact_tag_names_list')
        if 'secondary_emotional_impact_tag_names_list' in df_secondary_emotions_exploded.columns:
            df_secondary_emotions_exploded.rename(columns={'secondary_emotional_impact_tag_names_list': 'secondary_emotion'}, inplace=True)
        else:
            df_secondary_emotions_exploded['secondary_emotion'] = pd.NA
        df_secondary_emotions_exploded['secondary_emotion'] = df_secondary_emotions_exploded['secondary_emotion'].fillna('None')
        
        
        return df, df_secondary_categories_exploded, df_secondary_emotions_exploded

    except Exception as e:
        st.error(f"An error occurred during data preprocessing: {e}")
        # Log the full traceback for debugging if needed:
        # import traceback
        # st.error(traceback.format_exc())
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Return empty DFs on preprocessing error

# --- Load Data ---
data_url = "https://r2.garmeeh.com/input_file_0.csv"
df, df_secondary_categories_exploded, df_secondary_emotions_exploded = load_and_preprocess_data(data_url)

if df.empty:
    st.stop() # Stop execution if data loading failed

# Get all sources from the original dataframe and store total count
all_sources = sorted(df['article_source'].unique())
total_articles = len(df)

# Initialize selected_sources in session state if it's not already present.
if 'selected_sources' not in st.session_state:
    # Exclude 'The Irish Sun' by default
    st.session_state.selected_sources = [source for source in all_sources if source != 'The Irish Sun']

# --- App Title ---
st.title("📰 The Emotional Pulse of Early 2025 News")
st.markdown("An Analysis of the Emotional Impact the news has on us along with a look at Sentiment, Clickbait and Categories (Jan-Apr 2025)")

# --- Creator Section ---
st.markdown("""
<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0;'>
    <div style='text-align: center;'>
        <h4>📊 Created by Gary Meehan</h4>
        <div style='display: flex; justify-content: center; gap: 20px; margin-top: 15px;'>
            <a href='https://www.linkedin.com/in/gary-meehan-a5948747/' target='_blank' style='text-decoration: none; color: #0077B5; font-weight: bold;'>
                🔗 LinkedIn
            </a>
            <a href='https://github.com/garmeeh' target='_blank' style='text-decoration: none; color: #333; font-weight: bold;'>
                💻 GitHub
            </a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Filter UI ---
st.info("💡 By default we have omitted The Irish Sun source as it has over 30k articles which skews the data. Below you can add/remove sources to your liking.")

with st.expander("Filter by News Source", expanded=True):
    # Callbacks to manage select/deselect all functionality.
    def select_all_sources():
        for source in all_sources:
            st.session_state[f"cb_{source}"] = True

    def deselect_all_sources():
        for source in all_sources:
            st.session_state[f"cb_{source}"] = False
            
    # Initialize checkbox states on the first run.
    if 'filter_init' not in st.session_state:
        # Set all sources to checked except 'The Irish Sun'
        for source in all_sources:
            if source == 'The Irish Sun':
                st.session_state[f"cb_{source}"] = False
            else:
                st.session_state[f"cb_{source}"] = True
        st.session_state['filter_init'] = True

    # Arrange "Select All" / "Deselect All" buttons side-by-side.
    st.button("Select All Sources", on_click=select_all_sources, use_container_width=True)
    st.button("Deselect All Sources", on_click=deselect_all_sources, use_container_width=True)
    
    st.markdown("---") # Visual separator

    # Create columns for the checkboxes for a cleaner layout.
    num_columns = 5
    cols = st.columns(num_columns)
    
    selected_sources = []
    
    for i, source in enumerate(all_sources):
        with cols[i % num_columns]:
            # The widget's state is the source of truth, managed via its key in st.session_state.
            if st.checkbox(source, key=f"cb_{source}"):
                selected_sources.append(source)
    
# Filter the dataframes based on the selected sources.
df = df[df['article_source'].isin(selected_sources)]
df_secondary_categories_exploded = df_secondary_categories_exploded[df_secondary_categories_exploded['article_source'].isin(selected_sources)]
df_secondary_emotions_exploded = df_secondary_emotions_exploded[df_secondary_emotions_exploded['article_source'].isin(selected_sources)]

# If the filtered dataframe is empty, show a message and stop.
if df.empty:
    st.warning("No articles found for the selected sources. Please select at least one source to continue.")
    st.stop()

st.markdown(f"*Analysis based on {len(df)} articles out of {total_articles} total articles in dataset.*")
st.markdown("---")

# --- Executive Summary / TL;DR ---
st.header("🗞️ Summary")
if not df.empty:
    avg_sentiment_score = df['sentiment_score'].mean()
    avg_clickbait_level = df['clickbait_level'].mean()
    
    sentiment_desc = "Neutral"
    if avg_sentiment_score > 0.5: sentiment_desc = "Slightly Positive"
    elif avg_sentiment_score < -0.5: sentiment_desc = "Slightly Negative"
    if avg_sentiment_score > 1.0: sentiment_desc = "Positive"
    elif avg_sentiment_score < -1.0: sentiment_desc = "Negative"

    top_category = df['primary_category_tag_name'].mode()[0] if not df['primary_category_tag_name'].mode().empty else "N/A"
    top_emotion = df['primary_emotional_impact_tag_name'].mode()[0] if not df['primary_emotional_impact_tag_name'].mode().empty else "N/A"

    # First row of metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall Sentiment", sentiment_desc, f"{avg_sentiment_score:.2f} avg score")
    with col2:
        st.metric("Avg. Clickbait Level", f"{avg_clickbait_level:.2f}/5")
    
    # Second row of metrics
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Most Frequent Category", top_category)
    with col4:
        st.metric("Most Frequent Emotion", top_emotion)
st.markdown("Dive deeper into the sections below for detailed insights.")
st.markdown("---")


# --- Headline Word Cloud ---
st.header("☁️ Headline Word Cloud")
st.markdown("These word clouds show the most frequently used words in headlines, separated by the overall sentiment of the article. The larger the word, the more often it appeared. This helps to quickly identify the key themes and topics driving different sentiments in the news.")
if not df.empty:
    

    # Create a set of stopwords
    stopwords = set(STOPWORDS)
    # Add custom stopwords relevant to news that might not be in the default list
    custom_stopwords = {
        'says', 'said', 'new', 'U', 'S', 'will', 'report', 'one', 'year', 'years', 'told', 'according', 'also', 'like', 'people',
    }
    stopwords.update(custom_stopwords)

    # Define a function to generate and display a word cloud for a given sentiment
    def generate_wordcloud_for_sentiment(sentiment_label, color):
        st.write(f"**{sentiment_label} Headlines**")
        text_data = " ".join(title for title in df[df['sentiment_label'] == sentiment_label]['article_title'])

        if text_data:
            wordcloud = WordCloud(
                width=1920, height=1080, 
                background_color='white', 
                colormap=color, 
                stopwords=stopwords
            ).generate(text_data)
            st.image(wordcloud.to_array())
        else:
            st.write(f"No headlines to display for '{sentiment_label}' sentiment.")

    # Generate word clouds for each sentiment
    generate_wordcloud_for_sentiment("Very Positive", "Greens")
    generate_wordcloud_for_sentiment("Positive", "GnBu")
    generate_wordcloud_for_sentiment("Neutral", "gray")
    generate_wordcloud_for_sentiment("Negative", "OrRd")
    generate_wordcloud_for_sentiment("Very Negative", "Reds")
else:
    st.write("Headline data not available.")
st.markdown("---")


# --- News Source Comparison ---
st.header("⚖️ News Source Comparison")
if not df.empty:
    st.subheader("Performance Metrics by News Source")
    st.markdown("This table provides a high-level comparison of the selected news sources. You can see how many articles were analyzed from each source, their average sentiment and clickbait scores, and the topics and emotions they most frequently cover. The colors help to quickly spot outliers: red indicates more negative sentiment or higher clickbait.")

    def get_mode(series):
        mode = series.mode()
        return mode[0] if not mode.empty else "N/A"

    source_analysis = df.groupby('article_source').agg(
        total_articles=('article_title', 'count'),
        avg_sentiment_score=('sentiment_score', 'mean'),
        avg_clickbait_level=('clickbait_level', 'mean'),
        most_common_category=('primary_category_tag_name', get_mode),
        most_common_emotion=('primary_emotional_impact_tag_name', get_mode)
    ).reset_index()
    
    source_analysis = source_analysis.sort_values(by='total_articles', ascending=False)

    source_analysis.rename(columns={
        'article_source': 'News Source',
        'total_articles': 'Total Articles',
        'avg_sentiment_score': 'Avg. Sentiment',
        'avg_clickbait_level': 'Avg. Clickbait',
        'most_common_category': 'Top Category',
        'most_common_emotion': 'Top Emotion'
    }, inplace=True)

    st.dataframe(source_analysis.style
                  .format({
                      'Avg. Sentiment': '{:.2f}', 
                      'Avg. Clickbait': '{:.2f}',
                      'Total Articles': '{:,}'
                  })
                  .background_gradient(cmap='RdYlGn', subset=['Avg. Sentiment'], vmin=-2, vmax=2)
                  .background_gradient(cmap='Reds', subset=['Avg. Clickbait'], vmin=1, vmax=5)
                  .bar(subset=['Total Articles'], color='#66C2A5'),
                  use_container_width=True,
                  hide_index=True)

    st.caption("Comparison of news sources by article count, average sentiment, and average clickbait level.")
else:
    st.write("News source data not available.")
st.markdown("---")


# --- Introduction ---
with st.expander("Introduction: Understanding the News Landscape", expanded=True):
    st.markdown(f"""
    This report analyzes {len(df):,} news articles published between January and April 2025.
    
    The primary goal wsa to uncover how the news makes us feel. In other words, what is the emotional impact of the news on us. 

    Also analyzed as part of this was:
    - **Sentiment:** The overall tone (Positive, Negative, Neutral) of the news.
    - **Clickbait:** Just what is the level of clickbait in the news these days?
    - **Categorization:** Standardized categories for the news.
    
    This analysis aims to provide a snapshot of the media landscape during this period.
    """)

# --- Methodology ---
with st.expander("Methodology: How This Analysis Was Done", expanded=False):
    st.markdown("""
    - **Data Source:** CSV file containing news article metadata and LLM-generated analysis.
    - **Time Period:** January 2025 – April 2025.
    - **Analysis Points:**
        - **Sentiment:** Classified as Very Negative, Negative, Neutral, Positive, Very Positive.
        - **Clickbait Score:** Rated 1 (low) to 5 (high).
        - **Category Tagging:** One primary and up to two secondary categories.
        - **Emotional Impact Tagging:** One primary and up to two secondary emotional impacts.
    - **Code:** Can be found in the [GitHub repository](https://github.com/garmeeh/news-emotional-impact).
    - **Open Dataset:** Can be found in the [R2 bucket](https://r2.garmeeh.com/input_file_0.csv) and on [Kaggle](https://www.kaggle.com/datasets/garmeeh/news-emotional-impact-and-sentiment)
    """)

st.markdown("---")

# --- Overall News Landscape ---
st.header("📊 Overall News Landscape: A Bird's-Eye View")
st.markdown("High-level trends observed in the dataset.")

st.subheader("Sentiment Distribution")
st.markdown("This chart shows the breakdown of all analyzed articles by their sentiment. It gives a quick overview of whether the news landscape during this period was predominantly positive, negative, or neutral.")
if not df.empty:
    sentiment_counts = df['sentiment_label'].value_counts().reindex(SENTIMENT_ORDER).reset_index()
    sentiment_counts.columns = ['sentiment_label', 'count']
    fig_sentiment_dist = px.bar(sentiment_counts, x='sentiment_label', y='count', 
                                title="Distribution of Sentiment Labels",
                                color='sentiment_label',
                                color_discrete_map=SENTIMENT_COLOR_MAP,
                                labels={'sentiment_label': 'Sentiment', 'count': 'Number of Articles'})
    fig_sentiment_dist.update_layout(xaxis_title="Sentiment", yaxis_title="Number of Articles", showlegend=False)
    st.plotly_chart(fig_sentiment_dist, use_container_width=True)
    
    with st.expander("Sample Headlines by Sentiment", expanded=False):
        st.markdown("Here are 5 sample headlines from each sentiment category to give you a qualitative feel for how the sentiment classification works.")
        for sentiment_val in SENTIMENT_ORDER:
            sentiment_headlines = df[df['sentiment_label'] == sentiment_val]['article_title']
            if len(sentiment_headlines) > 0:
                st.write(f"**{sentiment_val} Headlines:**")
                sample_count = min(5, len(sentiment_headlines))
                sample_headlines = sentiment_headlines.sample(sample_count, random_state=42).tolist()
                for headline in sample_headlines:
                    st.markdown(f"- {headline}")
            else:
                st.write(f"**{sentiment_val} Headlines:**")
                st.write("No headlines available for this sentiment category.")
else:
    st.write("No data to display for sentiment distribution.")

st.subheader("Article Publication Volume Over Time")
st.markdown("This chart shows the number of news articles published each week during the analysis period. It helps to identify weeks with high or low news output.")
if not df.empty:
    df_weekly_counts = df.set_index('article_publish_date').resample('W').size().reset_index(name='count')
    fig_weekly_counts = px.line(df_weekly_counts, x='article_publish_date', y='count',
                                 title="Number of Articles Published Per Week",
                                 labels={'article_publish_date': 'Week', 'count': 'Number of Articles'})
    fig_weekly_counts.update_layout(xaxis_title="Week", yaxis_title="Number of Articles")
    st.plotly_chart(fig_weekly_counts, use_container_width=True)
else:
    st.write("No data to display for publication volume.")

st.subheader("Dominant News Categories (Primary)")
st.markdown("This chart highlights the top 10 most common topics covered in the news. It helps us understand what subjects were most prevalent in the media during this period.")
if not df.empty:
    category_counts = df['primary_category_tag_name'].value_counts().nlargest(10).reset_index()
    category_counts.columns = ['primary_category_tag_name', 'count']
    fig_category_dist = px.bar(category_counts, x='count', y='primary_category_tag_name',
                               orientation='h',
                               title="Top 10 Primary News Categories",
                               color='primary_category_tag_name',
                               color_discrete_sequence=PALETTE,
                               labels={'primary_category_tag_name': 'Category', 'count': 'Number of Articles'})
    fig_category_dist.update_layout(yaxis_title="Category", xaxis_title="Number of Articles", yaxis={'categoryorder':'total ascending'}, showlegend=False)
    st.plotly_chart(fig_category_dist, use_container_width=True)
    
    with st.expander("Sample Headlines by News Category", expanded=False):
        st.markdown("Here are 5 sample headlines from each of the top 10 news categories to illustrate the types of stories that fall under each topic.")
        
        # Get the top 10 categories (same as used in the chart)
        top_10_categories = df['primary_category_tag_name'].value_counts().nlargest(10).index
        
        for category in top_10_categories:
            category_headlines = df[df['primary_category_tag_name'] == category]['article_title']
            if len(category_headlines) > 0:
                st.write(f"**{category}:**")
                sample_count = min(5, len(category_headlines))
                sample_headlines = category_headlines.sample(sample_count, random_state=42).tolist()
                for headline in sample_headlines:
                    st.markdown(f"- {headline}")
            else:
                st.write(f"**{category}:**")
                st.write("No headlines available for this category.")
else:
    st.write("No data to display for news categories.")

st.subheader("Clickbait Levels")
st.markdown("This chart shows the distribution of clickbait scores, rated on a scale from 1 (low) to 5 (high). It reveals how frequently articles used sensationalized or misleading headlines to attract clicks.")
if not df.empty:
    clickbait_counts = df['clickbait_level'].value_counts().sort_index().reset_index()
    clickbait_counts.columns = ['clickbait_level', 'count']
    fig_clickbait_dist = px.bar(clickbait_counts, x='clickbait_level', y='count', 
                                 title="Distribution of Clickbait Scores (1-5)",
                                 color='clickbait_level',
                                 color_continuous_scale=px.colors.sequential.Reds,
                                 labels={'clickbait_level': 'Clickbait Score', 'count': 'Number of Articles'})
    fig_clickbait_dist.update_layout(xaxis_title="Clickbait Score", yaxis_title="Number of Articles", coloraxis_showscale=False)
    st.plotly_chart(fig_clickbait_dist, use_container_width=True)
    
    with st.expander("Sample Headlines by Clickbait Level", expanded=False):
        st.markdown("Here are 5 sample headlines from each clickbait level to illustrate how the clickbait scoring works. Level 1 represents low clickbait, while Level 5 represents high clickbait.")
        for clickbait_level in sorted(df['clickbait_level'].unique()):
            clickbait_headlines = df[df['clickbait_level'] == clickbait_level]['article_title']
            if len(clickbait_headlines) > 0:
                st.write(f"**Level {clickbait_level} Headlines:**")
                sample_count = min(5, len(clickbait_headlines))
                sample_headlines = clickbait_headlines.sample(sample_count, random_state=42).tolist()
                for headline in sample_headlines:
                    st.markdown(f"- {headline}")
            else:
                st.write(f"**Level {clickbait_level} Headlines:**")
                st.write("No headlines available for this clickbait level.")
else:
    st.write("No data to display for clickbait levels.")

st.subheader("Distribution of Emotional Impacts")
st.markdown("This chart breaks down the emotional impact of news articles, separating them into primary (the main emotion) and secondary (other emotions present). It helps us understand not just *what* emotions the news evokes, but also the complexity of those feelings.")
if not df.empty:
    # Get counts for primary emotions
    primary_counts = df[df['primary_emotional_impact_tag_name'] != 'Unknown']['primary_emotional_impact_tag_name'].value_counts().reset_index()
    primary_counts.columns = ['emotion', 'count']
    primary_counts['type'] = 'Primary'

    # Get counts for secondary emotions
    secondary_counts = df_secondary_emotions_exploded[df_secondary_emotions_exploded['secondary_emotion'] != 'None']['secondary_emotion'].value_counts().reset_index()
    secondary_counts.columns = ['emotion', 'count']
    secondary_counts['type'] = 'Secondary'

    # Combine the data
    combined_breakdown = pd.concat([primary_counts, secondary_counts])

    if not combined_breakdown.empty:
        # Create the stacked bar chart
        fig_stacked_emotion = px.bar(combined_breakdown, 
                                        x='count', 
                                        y='emotion', 
                                        color='type',
                                        orientation='h',
                                        title='Primary vs. Secondary Emotional Impact Breakdown',
                                        labels={'count': 'Number of Mentions', 'emotion': 'Emotional Impact', 'type': 'Impact Type'},
                                        color_discrete_map={'Primary': '#66C2A5', 'Secondary': '#F4A582'})
        fig_stacked_emotion.update_layout(
            yaxis_title="Emotional Impact",
            xaxis_title="Number of Mentions",
            yaxis={'categoryorder':'total ascending'},
            height=600,
            barmode='stack'
        )
        st.plotly_chart(fig_stacked_emotion, use_container_width=True)
        st.caption("This chart shows the breakdown of each emotional impact into its primary and secondary mentions.")
        
        with st.expander("Sample Headlines by Emotional Impact", expanded=False):
            st.markdown("Here are 5 sample headlines from each emotional impact category to illustrate how articles evoke different feelings and emotional responses.")
            
            # Get all unique emotional impacts from primary emotions (excluding 'Unknown')
            unique_emotions = df[df['primary_emotional_impact_tag_name'] != 'Unknown']['primary_emotional_impact_tag_name'].value_counts().index
            
            for emotion in unique_emotions:
                emotion_headlines = df[df['primary_emotional_impact_tag_name'] == emotion]['article_title']
                if len(emotion_headlines) > 0:
                    st.write(f"**{emotion}:**")
                    sample_count = min(5, len(emotion_headlines))
                    sample_headlines = emotion_headlines.sample(sample_count, random_state=42).tolist()
                    for headline in sample_headlines:
                        st.markdown(f"- {headline}")
                else:
                    st.write(f"**{emotion}:**")
                    st.write("No headlines available for this emotional impact.")
    else:
        st.write("Not enough data for a stacked breakdown visualization.")

else:
    st.write("No data to display for combined emotional impacts.")
    
st.markdown("---")

# --- Deep Dive: Sentiment Analysis ---
st.header("🧐 Deep Dive: Sentiment Analysis")
if not df.empty and 'article_publish_date' in df.columns and 'sentiment_score' in df.columns:
    st.subheader("Sentiment Over Time")
    st.markdown("This line chart tracks the average sentiment of news articles on a weekly basis. It helps visualize whether the news became more positive or negative over the four-month period.")
    df_time_sentiment = df.set_index('article_publish_date').resample('W')['sentiment_score'].mean().reset_index()
    fig_sentiment_time = px.line(df_time_sentiment, x='article_publish_date', y='sentiment_score',
                                 title="Average Sentiment Score Over Time (Weekly)",
                                 labels={'article_publish_date': 'Date', 'sentiment_score': 'Average Sentiment Score (-2 to +2)'})
    fig_sentiment_time.update_layout(xaxis_title="Week", yaxis_title="Average Sentiment Score")
    st.plotly_chart(fig_sentiment_time, use_container_width=True)

    st.subheader("Sentiment by Primary News Category (Top 10 Categories)")
    st.markdown("This chart breaks down the sentiment for the top 10 news categories. It allows us to see which topics are associated with more positive, negative, or neutral coverage.")
    top_categories = df['primary_category_tag_name'].value_counts().nlargest(10).index
    df_sentiment_category = df[df['primary_category_tag_name'].isin(top_categories)]
    sentiment_by_category = df_sentiment_category.groupby(['primary_category_tag_name', 'sentiment_label']).size().reset_index(name='count')
    
    fig_sentiment_by_cat = px.bar(sentiment_by_category, x='primary_category_tag_name', y='count', color='sentiment_label',
                                  title="Sentiment Distribution within Top 10 News Categories",
                                  color_discrete_map=SENTIMENT_COLOR_MAP,
                                  labels={'primary_category_tag_name': 'Category', 'count': 'Number of Articles', 'sentiment_label': 'Sentiment'},
                                  category_orders={"sentiment_label": SENTIMENT_ORDER})
    fig_sentiment_by_cat.update_layout(xaxis_title="Category", yaxis_title="Number of Articles")
    st.plotly_chart(fig_sentiment_by_cat, use_container_width=True)

    st.subheader("Sentiment by News Source (Top 5 Sources)")
    st.markdown("This chart compares the sentiment distribution for the top 5 news sources, showing the proportion of positive, negative, and neutral articles each one publishes.")
    top_sources = df['article_source'].value_counts().nlargest(5).index
    df_sentiment_source = df[df['article_source'].isin(top_sources)]
    sentiment_by_source = df_sentiment_source.groupby(['article_source', 'sentiment_label']).size().reset_index(name='count')

    fig_sentiment_by_source = px.bar(sentiment_by_source, x='article_source', y='count', color='sentiment_label',
                                     title="Sentiment Distribution for Top 5 News Sources",
                                     color_discrete_map=SENTIMENT_COLOR_MAP,
                                     labels={'article_source': 'News Source', 'count': 'Number of Articles', 'sentiment_label': 'Sentiment'},
                                     category_orders={"sentiment_label": SENTIMENT_ORDER})
    fig_sentiment_by_source.update_layout(xaxis_title="News Source", yaxis_title="Number of Articles")
    st.plotly_chart(fig_sentiment_by_source, use_container_width=True)

else:
    st.write("Sentiment analysis data not fully available for deep dive.")
st.markdown("---")


# --- Deep Dive: Clickbait Uncovered ---
st.header("🎣 Deep Dive: Clickbait Uncovered")
if not df.empty:
    st.subheader("Average Clickbait Score by Primary News Category")
    st.markdown("This chart shows the average clickbait score for each news category. It helps identify which topics tend to have more sensationalized headlines.")
    avg_clickbait_category = df.groupby('primary_category_tag_name')['clickbait_level'].mean().reset_index()
    fig_clickbait_cat = px.bar(avg_clickbait_category, x='clickbait_level', y='primary_category_tag_name', orientation='h',
                               title="Average Clickbait Score for All Categories",
                               color='primary_category_tag_name',
                               color_discrete_sequence=PALETTE,
                               labels={'primary_category_tag_name': 'Category', 'clickbait_level': 'Average Clickbait Score (1-5)'})
    fig_clickbait_cat.update_layout(yaxis_title="Category", xaxis_title="Average Clickbait Score", yaxis={'categoryorder':'total ascending'}, showlegend=False, height=2000)
    st.plotly_chart(fig_clickbait_cat, use_container_width=True)
    
    with st.expander("Sample Headlines by Category (Clickbait Analysis)", expanded=False):
        st.markdown("Here are 5 sample headlines from each news category to illustrate how clickbait levels vary across different topics. Categories are ordered by their average clickbait score.")
        
        # Get categories ordered by average clickbait score (highest first)
        avg_clickbait_by_category = df.groupby('primary_category_tag_name')['clickbait_level'].mean().sort_values(ascending=False)
        
        for category in avg_clickbait_by_category.index:
            category_headlines = df[df['primary_category_tag_name'] == category]['article_title']
            avg_score = avg_clickbait_by_category[category]
            
            if len(category_headlines) > 0:
                st.write(f"**{category}** (Avg. Clickbait Score: {avg_score:.2f}):")
                sample_count = min(5, len(category_headlines))
                sample_headlines = category_headlines.sample(sample_count, random_state=42).tolist()
                for headline in sample_headlines:
                    st.markdown(f"- {headline}")
            else:
                st.write(f"**{category}** (Avg. Clickbait Score: {avg_score:.2f}):")
                st.write("No headlines available for this category.")

    st.subheader("Clickbait Level vs. Sentiment Score")
    st.markdown("This box plot explores the relationship between clickbait and sentiment. Each box shows the range of sentiment scores for a given clickbait level. This helps answer the question: are more clickbaity headlines more positive or negative?")
    fig_clickbait_sentiment = px.box(df, x='clickbait_level', y='sentiment_score', color='clickbait_level',
                                     title="Clickbait Level vs. Sentiment Score",
                                     labels={'clickbait_level': 'Clickbait Level (1-5)', 'sentiment_score': 'Sentiment Score (-2 to +2)'})
    fig_clickbait_sentiment.update_layout(xaxis_title="Clickbait Level", yaxis_title="Sentiment Score", coloraxis_showscale=False)
    st.plotly_chart(fig_clickbait_sentiment, use_container_width=True)
    st.caption("This box plot shows the distribution of sentiment scores for each clickbait level. Higher clickbait headlines do not always correlate with extremely negative or positive sentiment in this sample.")

else:
    st.write("Clickbait analysis data not available.")
st.markdown("---")


# --- Deep Dive: Categorical Analysis ---
st.header("🏷️ Deep Dive: Categorical Analysis")
if not df_secondary_categories_exploded.empty:
    st.subheader("Top Primary & Secondary Category Combinations")
    st.markdown("News stories are often about more than one thing. This section reveals the most common sub-categories that appear alongside the top 5 main news categories. Each chart shows the top 3 secondary topics for a major primary topic.")
    # Focus on top primary categories first
    top_primary_cats_for_secondary = df['primary_category_tag_name'].value_counts().nlargest(5).index
    df_filtered_primary_cats = df_secondary_categories_exploded[
        df_secondary_categories_exploded['primary_category_tag_name'].isin(top_primary_cats_for_secondary)
    ]
    
    # Count secondary categories within these top primary categories
    category_combinations = df_filtered_primary_cats.groupby(['primary_category_tag_name', 'secondary_category']).size().reset_index(name='count')
    category_combinations = category_combinations[category_combinations['secondary_category'] != 'None'] # Exclude if no secondary
    category_combinations = category_combinations.sort_values(['primary_category_tag_name', 'count'], ascending=[True, False])
    
    # Get top N secondary for each primary
    top_n_secondary_per_primary = category_combinations.groupby('primary_category_tag_name').head(3)

    if not top_n_secondary_per_primary.empty:
        st.subheader("Top 3 Secondary Categories for Each of the Top 5 Primary Categories")

        # Get a color map for the top primary categories for consistent coloring
        primary_cat_colors = dict(zip(top_primary_cats_for_secondary, PALETTE))

        for category in top_primary_cats_for_secondary:
            subset = top_n_secondary_per_primary[top_n_secondary_per_primary['primary_category_tag_name'] == category]

            if not subset.empty:
                fig_cat_combo = px.bar(subset,
                               x='count',
                               y='secondary_category',
                               orientation='h',
                               color='primary_category_tag_name',
                               color_discrete_map={category: primary_cat_colors.get(category)},
                               labels={'secondary_category': '', 'count': 'Number of Articles'})
                fig_cat_combo.update_layout(
                    title=f"Primary Category: {category}",
                    xaxis_title="Number of Articles",
                    yaxis={'categoryorder':'total ascending'},
                    showlegend=False,
                    height=200,
                    margin=dict(l=0, r=0, t=40, b=20), # Adjust margin for title
                    plot_bgcolor='rgba(0,0,0,0)', # Transparent background
                    paper_bgcolor='rgba(0,0,0,0)' # Transparent background
                )
                st.plotly_chart(fig_cat_combo, use_container_width=True)

        st.caption("Shows the most frequent secondary categories associated with the top 5 primary news categories.")
    else:
        st.write("Not enough data for category combination visualization.")

    st.subheader("Category Focus for Top 5 News Sources")
    st.markdown("This chart shows the editorial focus of the top 5 news sources by revealing which of the top 5 news categories they publish most often. It helps to understand the reporting priorities and specialties of each source.")
    # First, find the top 5 news sources by article count
    top_sources_cat = df['article_source'].value_counts().nlargest(5).index
    # Filter the dataframe to only include these sources
    df_top_sources = df[df['article_source'].isin(top_sources_cat)]
    
    # Now, find the top 5 categories *within this subset*
    top_categories_in_top_sources = df_top_sources['primary_category_tag_name'].value_counts().nlargest(5).index
    
    # Filter the subset further to only these top categories
    df_source_cat_focus = df_top_sources[df_top_sources['primary_category_tag_name'].isin(top_categories_in_top_sources)]
    
    # Group and count for the plot
    source_cat_counts = df_source_cat_focus.groupby(['article_source', 'primary_category_tag_name']).size().reset_index(name='count')

    if not source_cat_counts.empty:
        fig_source_cat = px.bar(source_cat_counts, x='article_source', y='count', color='primary_category_tag_name',
                                height=500,
                                color_discrete_sequence=PALETTE,
                                title="Category Focus for Top 5 News Sources (Showing Top 5 Categories Within These Sources)",
                                labels={'article_source': 'News Source', 'count': 'Number of Articles', 'primary_category_tag_name': 'Primary Category'})
        st.plotly_chart(fig_source_cat, use_container_width=True)
        st.caption("This chart shows the most common categories published by the top 5 news sources.")
    else:
        st.write("Not enough data for source category focus visualization.")
else:
    st.write("Categorical analysis data not fully available.")
st.markdown("---")

# --- Deep Dive: The Emotional Landscape of News ---
st.header("🎭 Deep Dive: The Emotional Landscape of News")
if not df.empty and not df_secondary_emotions_exploded.empty:
    st.subheader("Top Primary & Secondary Emotion Combinations")
    st.markdown("Emotions are complex. This section shows which secondary emotions are most often paired with the top 5 primary emotions, revealing the nuanced emotional texture of news stories.")
    # Focus on top primary emotions first
    top_primary_emotions_for_secondary = df['primary_emotional_impact_tag_name'].value_counts().nlargest(5).index
    df_filtered_primary_emotions = df_secondary_emotions_exploded[
        df_secondary_emotions_exploded['primary_emotional_impact_tag_name'].isin(top_primary_emotions_for_secondary)
    ]
    
    # Count secondary emotions within these top primary emotions
    emotion_combinations = df_filtered_primary_emotions.groupby(['primary_emotional_impact_tag_name', 'secondary_emotion']).size().reset_index(name='count')
    emotion_combinations = emotion_combinations[emotion_combinations['secondary_emotion'] != 'None'] # Exclude if no secondary
    emotion_combinations = emotion_combinations.sort_values(['primary_emotional_impact_tag_name', 'count'], ascending=[True, False])
    
    # Get top N secondary for each primary
    top_n_secondary_per_primary_emotion = emotion_combinations.groupby('primary_emotional_impact_tag_name').head(3)

    if not top_n_secondary_per_primary_emotion.empty:
        st.subheader("Top 3 Secondary Emotions for Each of the Top 5 Primary Emotions")

        # Get a color map for the top primary emotions for consistent coloring
        primary_emotion_colors = dict(zip(top_primary_emotions_for_secondary, PALETTE))

        for emotion in top_primary_emotions_for_secondary:
            subset = top_n_secondary_per_primary_emotion[top_n_secondary_per_primary_emotion['primary_emotional_impact_tag_name'] == emotion]

            if not subset.empty:
                fig_emotion_combo = px.bar(subset,
                               x='count',
                               y='secondary_emotion',
                               orientation='h',
                               color='primary_emotional_impact_tag_name',
                               color_discrete_map={emotion: primary_emotion_colors.get(emotion)},
                               labels={'secondary_emotion': '', 'count': 'Number of Articles'})
                fig_emotion_combo.update_layout(
                    title=f"Primary Emotion: {emotion}",
                    xaxis_title="Number of Articles",
                    yaxis={'categoryorder':'total ascending'},
                    showlegend=False,
                    height=150,
                    margin=dict(l=0, r=0, t=40, b=20), # Adjust margin for title
                    plot_bgcolor='rgba(0,0,0,0)', # Transparent background
                    paper_bgcolor='rgba(0,0,0,0)' # Transparent background
                )
                st.plotly_chart(fig_emotion_combo, use_container_width=True)

        st.caption("Shows the most frequent secondary emotions associated with the top 5 primary news emotions.")
    else:
        st.write("Not enough data for emotion combination visualization.")
        
    st.subheader("Emotional Impact by Primary News Category")
    st.markdown("Different topics evoke different feelings. These charts show the top 3 emotions associated with each of the top 5 news categories, helping to illustrate the typical emotional response to different subjects.")
    # Identify Top Categories
    N_TOP_CATEGORIES = 5
    top_categories_for_analysis = df['primary_category_tag_name'].value_counts().nlargest(N_TOP_CATEGORIES).index.tolist()
    # Optional: Exclude 'Unknown' if it appears in top categories and you don't want to analyze it
    if 'Unknown' in top_categories_for_analysis:
        top_categories_for_analysis.remove('Unknown')

    # Prepare the Data
    data_for_plotting = []
    N_TOP_EMOTIONS_PER_CAT = 3
    for category_val in top_categories_for_analysis:
        # Filter the DataFrame for the current category
        df_current_category = df[df['primary_category_tag_name'] == category_val]

        # Optional: Exclude 'Unknown' emotions for cleaner results within each category
        df_current_category = df_current_category[df_current_category['primary_emotional_impact_tag_name'] != 'Unknown']

        if df_current_category.empty:
            continue # Skip if no articles for this category (after filtering unknown emotions)

        # Get the value counts of emotions for THIS category
        emotion_counts = df_current_category['primary_emotional_impact_tag_name'].value_counts()

        # Get the top N emotions for THIS category
        top_n_emotions_for_category = emotion_counts.nlargest(N_TOP_EMOTIONS_PER_CAT).reset_index()
        top_n_emotions_for_category.columns = ['primary_emotional_impact_tag_name', 'count']

        # Add the category name itself as a column (for plotting)
        top_n_emotions_for_category['primary_category_tag_name'] = category_val

        # Append to our list
        data_for_plotting.append(top_n_emotions_for_category)

    # Concatenate the list of DataFrames into a single DataFrame
    if data_for_plotting:
        df_plot_ready_cat_emotion = pd.concat(data_for_plotting, ignore_index=True)
    else:
        df_plot_ready_cat_emotion = pd.DataFrame(columns=['primary_category_tag_name', 'primary_emotional_impact_tag_name', 'count']) # Empty df if no data

    # Visualize the Prepared Data
    if not df_plot_ready_cat_emotion.empty:
        fig_emotion_by_cat_revised = px.bar(
            df_plot_ready_cat_emotion,
            x='count',
            y='primary_emotional_impact_tag_name',
            orientation='h',
            color='primary_emotional_impact_tag_name',
            color_discrete_sequence=PALETTE,
            facet_row='primary_category_tag_name',
            labels={'count': 'Number of Articles', 'primary_emotional_impact_tag_name': ''}, # Cleared Y-axis title
            title=f"Top {N_TOP_EMOTIONS_PER_CAT} Emotional Impacts for Top {N_TOP_CATEGORIES} Categories",
            category_orders={"primary_category_tag_name": top_categories_for_analysis},
            height=800
        )
        # Ensure each facet y-axis is sorted independently and remove legend
        fig_emotion_by_cat_revised.update_layout(showlegend=False)
        fig_emotion_by_cat_revised.update_yaxes(categoryorder="total ascending", matches=None)
        # Clean up facet titles
        fig_emotion_by_cat_revised.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig_emotion_by_cat_revised, use_container_width=True)
    else:
        st.write("Not enough data for emotion by category visualization (revised).")

    st.subheader("Primary Emotional Impact by Sentiment Label")
    st.markdown("This series of charts connects sentiment with specific emotions. For each sentiment level (from 'Very Positive' to 'Very Negative'), it shows the top 3 emotions that were most frequently identified, clarifying what drives the overall tone.")
    # Refined logic to find top emotions PER sentiment label
    data_for_plotting = []
    N_TOP_EMOTIONS = 3  # Or 5, etc.

    for sentiment_val in SENTIMENT_ORDER:
        # Filter the DataFrame for the current sentiment label
        df_current_sentiment = df[df['sentiment_label'] == sentiment_val]

        # Optional: Exclude 'Unknown' emotions for cleaner results
        df_current_sentiment = df_current_sentiment[df_current_sentiment['primary_emotional_impact_tag_name'] != 'Unknown']

        if df_current_sentiment.empty:
            continue  # Skip if no articles for this sentiment

        # Get the value counts of emotions for THIS sentiment label
        emotion_counts = df_current_sentiment['primary_emotional_impact_tag_name'].value_counts()

        # Get the top N emotions for THIS sentiment label
        top_n_emotions_for_sentiment = emotion_counts.nlargest(N_TOP_EMOTIONS).reset_index()
        top_n_emotions_for_sentiment.columns = ['primary_emotional_impact_tag_name', 'count']

        # Add the sentiment label itself as a column (for plotting)
        top_n_emotions_for_sentiment['sentiment_label'] = sentiment_val

        # Append to our list
        data_for_plotting.append(top_n_emotions_for_sentiment)

    # Concatenate the list of DataFrames into a single DataFrame
    if data_for_plotting:
        df_plot_ready = pd.concat(data_for_plotting, ignore_index=True)
    else:
        df_plot_ready = pd.DataFrame(columns=['sentiment_label', 'primary_emotional_impact_tag_name', 'count'])

    # Visualize the prepared data
    if not df_plot_ready.empty:
        fig_emotion_by_sentiment_revised = px.bar(
            df_plot_ready,
            x='count',
            y='primary_emotional_impact_tag_name',
            orientation='h',
            color='primary_emotional_impact_tag_name',
            color_discrete_sequence=PALETTE,
            facet_row='sentiment_label',
            labels={'count': 'Number of Articles', 'primary_emotional_impact_tag_name': ''}, # Cleared Y-axis title for cleaner look
            title=f"Top {N_TOP_EMOTIONS} Emotional Impacts per Sentiment Label",
            category_orders={"sentiment_label": SENTIMENT_ORDER},
            height=800 # Increased height for vertical stacking
        )
        fig_emotion_by_sentiment_revised.update_layout(showlegend=False)
        fig_emotion_by_sentiment_revised.update_yaxes(categoryorder="total ascending", matches=None)
        # Clean up facet titles (e.g., "sentiment_label=Positive" -> "Positive")
        fig_emotion_by_sentiment_revised.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig_emotion_by_sentiment_revised, use_container_width=True)
    else:
        st.write("Not enough data for emotion by sentiment visualization (revised).")
else:
    st.write("Emotional landscape data not fully available.")
st.markdown("---")


# --- Intersections & Nuances ---
st.header("🔗 Intersections & Nuances")
if not df.empty:
    st.subheader("Clickbait Level vs. Primary Emotional Impact (Top 5 Emotions)")
    st.markdown("This chart investigates whether certain emotions are more likely to be used in clickbait headlines. It shows the average clickbait score for articles that have one of the top 5 most common primary emotions.")
    top_emotions_clickbait = df['primary_emotional_impact_tag_name'].value_counts().nlargest(5).index
    df_clickbait_emotion = df[df['primary_emotional_impact_tag_name'].isin(top_emotions_clickbait)]
    
    # Create a summary: average clickbait score for each top emotion
    avg_clickbait_per_emotion = df_clickbait_emotion.groupby('primary_emotional_impact_tag_name')['clickbait_level'].mean().reset_index()
    avg_clickbait_per_emotion = avg_clickbait_per_emotion.sort_values('clickbait_level', ascending=False)

    if not avg_clickbait_per_emotion.empty:
        fig_clickbait_emotion_avg = px.bar(avg_clickbait_per_emotion, 
                                       x='primary_emotional_impact_tag_name', 
                                       y='clickbait_level',
                                       color='primary_emotional_impact_tag_name',
                                       color_discrete_sequence=PALETTE,
                                       title="Average Clickbait Score for Top 5 Primary Emotional Impacts",
                                       labels={'primary_emotional_impact_tag_name': 'Emotional Impact', 'clickbait_level': 'Average Clickbait Score (1-5)'})
        st.plotly_chart(fig_clickbait_emotion_avg, use_container_width=True)
        st.caption("This chart shows if certain dominant emotions tend to be associated with higher or lower average clickbait scores.")
    else:
        st.write("Not enough data for clickbait vs. emotion visualization.")
else:
    st.write("Data for intersections not fully available.")
st.markdown("---")


# --- Emotional Impact-Category Heatmap ---
st.header("🔥 Emotional Impact-Category Heatmap")
st.markdown("This heatmap provides a bird's-eye view of the entire emotional landscape of the news. It shows which emotions are most frequently associated with each of the top 10 news categories. The darker the red, the more articles in that category evoked that specific emotion.")
if not df.empty:
    st.subheader("Emotional Impact Profile of Top 10 News Categories")
    
    # Get top 10 categories, excluding 'Unknown' if it exists
    top_10_categories = df[df['primary_category_tag_name'] != 'Unknown']['primary_category_tag_name'].value_counts().nlargest(10).index
    
    # Filter dataframe to only include these top categories
    df_heatmap = df[df['primary_category_tag_name'].isin(top_10_categories)]
    # Exclude 'Unknown' emotions for a cleaner heatmap
    df_heatmap = df_heatmap[df_heatmap['primary_emotional_impact_tag_name'] != 'Unknown']

    # Create pivot table showing all emotions for the top categories
    heatmap_data = pd.crosstab(df_heatmap['primary_category_tag_name'], df_heatmap['primary_emotional_impact_tag_name'])
    
    if not heatmap_data.empty:
        fig_heatmap = px.imshow(heatmap_data,
                                text_auto=True,
                                aspect="auto",
                                color_continuous_scale="Reds",
                                title="Emotional Impact Profile of Top 10 News Categories")
        fig_heatmap.update_layout(
            xaxis_title="Emotional Impact",
            yaxis_title="News Category",
            xaxis_tickangle=-45,
            height=600 # Increase height for better readability
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.caption("This heatmap shows the complete emotional profile for the 10 most frequent news categories.")
    else:
        st.write("Not enough data for a heatmap visualization.")
else:
    st.write("Data for heatmap not available.")
st.markdown("---")

# --- XIII. Appendix ---
# --- Appendix ---
with st.expander("Appendix: Tag Definitions & Prompts", expanded=True):
    st.subheader("Category Tags Used in Analysis")
    category_tags = [
        'Cost of Living', 'Housing Crisis', 'Taxation Policy', 'Public Spending & Budgets', 'Elections & Campaigns', 
        'Diplomacy & International Relations', 'Social Justice & Equality', 'Immigration & Refugees', 'Education Policy', 
        'Business & Corporate News', 'Banking & Financial Services', 'Startups & Entrepreneurship', 
        'Financial Markets & Investments', 'Digital Economy & Cryptocurrencies', 'Trade & Exports', 'Tourism & Travel', 
        'Artificial Intelligence & Automation', 'Cybersecurity & Data Privacy', 'Digital Privacy & Ethics', 
        'Technology & Gadgets', 'Space Exploration', 'Scientific Research & Discovery', 'Climate Change', 
        'Sustainability & Green Tech', 'Natural Disasters', 'Pollution & Waste', 'Conservation & Wildlife', 
        'Public Health & Outbreaks', 'Mental Health', 'Nutrition & Fitness', 'Healthcare Policy', 'Celebrity Gossip', 
        'Reality Shows', 'Arts & Culture', 'Awards & Festivals', 'Major Leagues & Tournaments', 
        'Athlete Scandals & Contract News', 'Labor Action & Disputes', 'Employment & Job Market', 
        'Consumer Rights & Product Recalls', 'Inventions & Patents', 'Transportation', 'Infrastructure & Development', 
        'Agriculture & Farming', 'Food & Dining', 'Family & Relationships', 'Personal Finance & Money-Saving', 
        'Personal Development & Self-Help', 'Fashion & Trends', 'Demographics & Population', 
        'Urban Development & Housing', 'Religion & Spirituality', 'Political Scandals & Corruption', 
        'Legal & Justice System', 'Organized Crime & Gangs', 'Court Cases & Trials', 'Terrorism & Extremism', 
        'Energy Crisis', 'Weapons & Military Tech', 'Esports', 'Product Reviews & Comparisons', 
        'Shopping Guides & Deals', 'Sponsored Content & Promotions', 'Product Launches', 'Subscription Services', 
        'Home & Garden Products', 'Beauty & Cosmetics Reviews', 'Gaming Hardware & Software', 'Lifestyle Products', 
        'Affiliate Content', 'Tech Reviews', 'Automotive Reviews'
    ]
    st.json(category_tags)

    st.subheader("Emotional Impact Tags Used in Analysis")
    emotional_impact_tags = [
        'Relief / Reassurance', 'Anxiety / Fear', 'Anger / Outrage', 'Moral Outrage / Ethical Conflict', 
        'Stress / Overwhelm', 'Sadness / Grief', 'Triggering / Traumatic', 'Hopelessness / Despair', 
        'Frustration / Helplessness', 'Cynicism / Distrust', 'Nostalgia / Loss of Past', 
        'Social Connection / Belonging', 'Guilt / Shame', 'Confusion / Uncertainty', 'Curiosity / Wonder', 
        'Uplifting / Inspiring', 'Compassion / Empathy', 'Empowerment / Motivation'
    ]
    st.json(emotional_impact_tags)

st.subheader("Prompts Used in Analysis")
with st.expander("Category prompt"):
    st.code("""
Analyze the given news article headline and description to determine a primary category (the single best match) and up to two optional secondary categories, if strongly relevant.

## Topic Tags & Guidelines

{category_tags_list}

# Steps

1. **Read and Understand**  
- Carefully read the headline and description to capture the main topic or focus of the article.

2. **Primary Tag Selection**  
- Choose exactly one tag that best categorizes the article’s core theme.
- If no tags apply, choose the single closest match.

3. **Secondary Tag Selection** (Optional)  
- If one or two other tags strongly apply, include them as secondary tags.
- Never include more than two secondary tags.
- If in doubt, stick with only the primary tag.

4. **Formatting**  
- Return a JSON object with two fields: `"primary_tag"` and `"secondary_tags"`.
- Example:
    ```json
    {{
    "primary_tag": "<SINGLE MOST APPROPRIATE TAG>",
    "secondary_tags": ["<OPTIONAL TAG 1>", "<OPTIONAL TAG 2>"]
    }}
    ```
- Do not add extra keys.

# Examples

## Example 1
**Headline**: "Inflation Hits New High, Worsening Cost of Living for Families"  
**Description**: "Families struggle to cope with escalating prices due to a sudden rise in inflation rates."

**Reasoning**:  
- Primary tag: **Cost of Living** (the article specifically addresses families’ financial burden).  
- Secondary tag: **Financial Markets & Investments** (there is a broader economic impact, albeit secondary).

**Final Output**:
```json
{{
"primary_tag": "Cost of Living",
"secondary_tags": ["Financial Markets & Investments"]
}}
```

## Example 2
**Headline**: “Historic Summit Strengthens International Trade Relations”
**Description**: “World leaders convened to forge new trade agreements, boosting global economic cooperation.”

**Reasoning**:
- Primary tag: **Diplomacy & International Relations** (core focus is world leaders collaborating).
- Secondary tag: **Trade & Exports** (the summit centers on new trade agreements).

**Final Output**:
```json
{{
"primary_tag": "Diplomacy & International Relations",
"secondary_tags": ["Trade & Exports"]
}}
```

# Notes

- Select more than one tag only when there is a “very high match” for each.
- If the article seems to fit only one tag, do not add secondary tags.
- Keep the JSON structure exactly as shown—no extra fields or formatting.
- Only include tags from the provided list of topic tags.

# Input

Headline: {headline}
Description: {description}
            """, language='text')
with st.expander("Clickbait prompt"):
    st.code("""
You are an AI assistant trained to classify the "clickbait" level of a given news article title. 

Evaluate the title based on the following factors:

- **Sensationalism**: Presence of exaggerated language, all-caps, or strong emotional words meant to provoke reactions.
- **Curiosity Gap**: If it teases the reader by withholding key information ("You won't believe what happened next", "This one trick...").
- **Urgency or Shock**: Use of words like "shocking", "unbelievable", "must-see".
- **Misleading or Over promising**: Whether the headline promises something unlikely or deceptive.

Assign a rating from **1** to **5**:

- **1 = Not Clickbait**: Straightforward, factual title; no exaggeration or withheld details.
- **2 = Slightly Clickbait**: Minor use of emotional language or mild curiosity gap.
- **3 = Moderately Clickbait**: Clear but not extreme sensationalism; some teasing phrases.
- **4 = Very Clickbait**: Strong emotional language, significant curiosity gap, possibly misleading.
- **5 = Extremely Clickbait**: Over-the-top sensational wording, heavy curiosity gap, highly misleading or “shocking” claims.

# Steps

1. Analyze the given title for any indications of sensationalism, curiosity gap, urgency, shock, misleading language, or over promising.
2. Determine the degree of clickbait using the provided rating scale.
3. Reason with a brief one-sentence explanation for the chosen rating.
4. Rate the given title and return a rating score

# Examples

**Example 1:**
- *Input*: "Shocking Discovery: Scientists Unveil Secret to Everlasting Youth!"
- *Output*: rating: 5

**Example 2:**
- *Input*: "Local Man Wins Lottery After Playing Same Numbers for 20 Years"
- *Output*: rating: 1


# Input

Headline: {headline}
            """, language='text')
with st.expander("Emotional impact prompt"):
    st.code("""
Analyze the given news article headline and categorize its emotional impact using the specified tags.  

You must designate exactly one PRIMARY TAG (the single strongest emotional impact) and up to two OPTIONAL SECONDARY TAGS (additional strong impacts, if any).  

# Emotional Impact Tags

{emotional_impact_tags_list}

# Steps

1. **Headline Analysis**  
   - Read and interpret the headline to understand the core subject matter and how it might affect a reader emotionally.

2. **Primary Tag Selection**  
   - Identify one tag that best represents the single strongest emotional response elicited by the headline.  
   - If no tags apply, default to the single tag that is most closely aligned with any possible reaction (or “Confusion / Uncertainty” if truly unclear).

3. **Secondary Tag Selection** (optional)  
   - If one or two additional tags also strongly apply—i.e., they would likely resonate with many readers—list them here.  
   - If in doubt, do not include secondary tags. Never include more than two secondary tags.

4. **Reasoning**  
   - Briefly explain, for each selected tag (primary and secondary), the elements in the headline that support its emotional impact. 
   - Keep this explanation concise.

5. **Classification**  
   - Format your final answer in valid JSON with fields “primary_tag” and “secondary_tags.”  
   - Example output:  
     ```json
     {{
       "primary_tag": "Anger / Outrage",
       "secondary_tags": ["Cynicism / Distrust"]
     }}
     ```

# Examples

## Example 1
**Headline**: "Local Shelter Saves Dozens of Abandoned Puppies with Community Effort"

**Reasoning**:  
  - Primary Tag: **Compassion / Empathy** (the community’s rescue effort and focus on puppies can evoke empathy)  
  - Secondary Tag: **Uplifting / Inspiring** (successful outcome provides a positive, uplifting feeling)  

**Output**:
```json
{{
  "primary_tag": "Compassion / Empathy",
  "secondary_tags": ["Uplifting / Inspiring"]
}}
```

## Example 2

Headline: “Government Scandal Unveils Extensive Misuse of Public Funds”

**Reasoning**:  
  - Primary Tag: **Anger / Outrage** (misuse of public funds provokes a sense of injustice)
  - Secondary Tags: **Cynicism / Distrust** (erodes trust in government) and **Frustration / Helplessness** (feeling powerless to effect change)

**Output**:
```json
{{
  "primary_tag": "Anger / Outrage",
  "secondary_tags": ["Cynicism / Distrust", "Frustration / Helplessness"]
}}
```

## Example 3

Headline: “Small Town’s Last Factory Closes After 80 Years”

**Reasoning**:
  - Primary Tag: **Sadness / Grief** (loss of jobs and tradition)
  - Secondary Tag: **Nostalgia / Loss of Past** (the end of a longstanding era)

**Output**:
```json
{{
  "primary_tag": "Sadness / Grief",
  "secondary_tags": ["Nostalgia / Loss of Past"]
}}
```

# Notes
  - Strictly adhere to one primary tag, and up to two secondary tags only if strongly applicable.
  - Use exact tag names from the provided list of emotional impact tags.
  - If no secondary tag strongly applies, leave “secondary_tags” empty.
  - Return your final classification in valid JSON format as shown.

# Input

Headline: {headline}
            """, language='text')
with st.expander("Headline sentiment prompt"):
    st.code("""
Analyze the sentiment of a given news article headline by assigning one category from the following set: 

{categories}

Assign a confidence score to your analysis.

# Steps

1. **Analyze the Headline**: Examine the words and structure of the headline to determine its overall tone.
2. **Determine Sentiment**: Choose the most appropriate sentiment category ({categories}).
3. **Confidence Score**: Assign a confidence score between 0 to 100, indicating how sure you are about your analysis.

Analyze the sentiment of a given news article headline by assigning one of the provided categories. 

# Steps

1. **Analyze the Headline**: Examine the words and structure of the headline to determine its overall tone.
2. **Determine Sentiment**: Choose the most appropriate sentiment category
3. **Confidence Score**: Assign a confidence score between 0 to 100, indicating how sure you are about your analysis.

# Examples

1. "Company profits soar to record heights"
   - Sentiment: Very Positive
   - Confidence: 90
   Reason: Contains strongly positive words ("soar", "record") and indicates financial success

2. "Scientists raise concerns over new medication"
   - Sentiment: Negative
   - Confidence: 80
   Reason: "Concerns" indicates problems or issues, suggesting negative implications

3. "Local council schedules monthly meeting"
   - Sentiment: Neutral
   - Confidence: 95
   Reason: Purely factual statement without emotional content

4. "Devastating earthquake leaves thousands homeless"
   - Sentiment: Very Negative
   - Confidence: 95
   Reason: Contains explicitly negative words ("devastating") and describes tragic outcome

5. "Small improvements noted in economic indicators"
   - Sentiment: Positive
   - Confidence: 75
   Reason: Shows progress but modest scope ("small") reduces confidence and intensity

# Notes

- Carefully consider emotional cues within the headline, such as adjectives and verbs that convey sentiment.
- If the sentiment is unclear, the 'Neutral' category can be used.


# Input

Headline: {headline}
            """, language='text')

st.markdown("---")
st.markdown("<div style='text-align: center;'>Created by <a href='https://www.linkedin.com/in/gary-meehan-a5948747/'>Gary Meehan</a></div>", unsafe_allow_html=True)
st.markdown("---")