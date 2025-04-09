import streamlit as st
import pandas as pd
import time
from openai import OpenAI
import altair as alt

st.set_page_config(layout='wide')
st.title('City/Category Pair Trends and Analysis')

######### DEFINE VARS #########
import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
#REMOVE KEY BEFORE PUSHING
openai_key = st.secrets['openai_key']
client = OpenAI(api_key=openai_key)

######### LOAD DATA ######### 
@st.cache_data
def load_data():
    keywords = pd.read_csv('data/all/citycategory/keyword_trends_by_category_city.csv', usecols=['category_city', 'top_positive_keywords', 'top_negative_keywords', 'top_positive_coefs', 'top_negative_coefs'])
    similarity = pd.read_csv('data/all/citycategory/most_similar_cities.csv')
    topics = pd.read_csv('data/all/citycategory/topic_data.csv', usecols=['City_Category', 'Sentiment', 'Words', 'topic_words'])
    return keywords, similarity, topics

keywords, similarity, topics = load_data()

######### SESSION STATES #########
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
    st.session_state['messages'].append({'role': 'assistant', 'content': 'Hi! Please enter your abstract.'})

######### DEFINE FUNCTIONS ######### 
def load_categories(df):
    if 'categories' not in st.session_state:
        st.session_state.categories = df['category_city'].apply(lambda x: x.split(' - ')[0]).unique().tolist()
    return st.session_state.categories

def load_cities(df):
    if 'cities' not in st.session_state:
        st.session_state.cities = df['category_city'].apply(lambda x: x.split(' - ')[1]).unique().tolist()
    return st.session_state.cities

def process_keywords(selected_category, selected_city):
    selected_pair = f"{selected_category} - {selected_city}"
    filtered_df = keywords[keywords['category_city'] == selected_pair]
    
    pos_words = filtered_df['top_positive_keywords'].str.split(", ").explode().reset_index(drop=True)
    neg_words = filtered_df['top_negative_keywords'].str.split(", ").explode().reset_index(drop=True)
    pos_coefs = filtered_df['top_positive_coefs'].str.split(", ").apply(lambda x: list(map(float, x))).explode().reset_index(drop=True)
    neg_coefs = filtered_df['top_negative_coefs'].str.split(", ").apply(lambda x: list(map(float, x))).explode().reset_index(drop=True)
    
    return pos_words, pos_coefs, neg_words, neg_coefs

@st.cache_data
def get_keyword_df(selected_category, selected_city):
    pos_words, pos_coefs, neg_words, neg_coefs = process_keywords(selected_category, selected_city)
    pos_df = pd.DataFrame({"Word": pos_words, "Impact": pos_coefs})
    neg_df = pd.DataFrame({"Word": neg_words, "Impact": neg_coefs})

    return pos_df, neg_df

def get_topics(selected_category, selected_city):
    selected_pair = f"{selected_category} - {selected_city}"

    filtered_df_pos = topics[(topics['City_Category'] == selected_pair) & (topics['Sentiment'] == 'Positive')]
    filtered_df_neg = topics[(topics['City_Category'] == selected_pair) & (topics['Sentiment'] == 'Negative')]

    top_positive_topics = filtered_df_pos['Words'].tolist()
    top_negative_topics = filtered_df_neg['Words'].tolist()

    return top_positive_topics, top_negative_topics

def get_topic_words(selected_category, selected_city):
    selected_pair = f"{selected_category} - {selected_city}"

    filtered_df_pos = topics[(topics['City_Category'] == selected_pair) & (topics['Sentiment'] == 'Positive')]
    filtered_df_neg = topics[(topics['City_Category'] == selected_pair) & (topics['Sentiment'] == 'Negative')]

    top_positive_topic_words = filtered_df_pos['topic_words'].tolist()
    top_negative_topic_words = filtered_df_neg['topic_words'].tolist()

    return top_positive_topic_words, top_negative_topic_words

def get_most_similar_city(selected_city_category):
    similarity_row = similarity[similarity['City'] == selected_city_category]
    
    most_similar_pair = similarity_row['Most_Similar_City'].values[0]
    
    similar_category, similar_city = most_similar_pair.split(' - ')
    
    return similar_category, similar_city

def query_llm(category, city, similar_cat, similar_city, top_positive_keywords, top_negative_keywords, similar_pos_words, similar_neg_words):
    prompt = f"""
    Give a general analysis of what customers like and dislike:

    ### City and Category Analysis:
    - **City**: {city}
    - **Category**: {category}

    #### Top Positive Words for {city} - {category}
    {top_positive_keywords}

    #### Top Negative Words for {city} - {category}
    {top_negative_keywords}

    ### Strategy for Success:
    Based on the analysis above, suggest the best strategy to succeed within this city and category to open my own place and be successful.

    ### Most Similar City-Category Pair Analysis:
    Now, compare the most similar city-category pair to {city} and {category}:

    - **Most Similar City-Category Pair**: {similar_city} - {similar_cat}

    #### Top Positive Words for the Most Similar City-Category Pair {similar_city} - {similar_cat}
    {similar_pos_words}

    #### Top Negative Words for the Most Similar City-Category Pair {similar_city} - {similar_cat}
    {similar_neg_words}

    """

    completion = client.chat.completions.create(
        model='gpt-4o-2024-11-20',
        # model="o3-mini-2025-01-31",
        # model='gpt-4o-mini-2024-07-18',
        messages=[
            {'role': 'system', "content": "You are an authority on food and beverage establishments. Help me anaylze review trends."},
            {"role": "user", "content": prompt},
        ],
        stream=True
    )

    for chunk in completion:
        if hasattr(chunk.choices[0].delta, "content"): 
            yield chunk.choices[0].delta.content
            time.sleep(0.02)

def query_llm_topics(category, city, similar_cat, similar_city, top_positive_keywords, top_negative_keywords, similar_pos_words, similar_neg_words, top_positive_topics, top_negative_topics):
    prompt = f"""
    Give a general analysis of what customers like and dislike:

    ### City and Category Analysis:
    - **City**: {city}
    - **Category**: {category}

    #### Top Positive Words for {city} - {category}
    {top_positive_keywords}

    #### Top Negative Words for {city} - {category}
    {top_negative_keywords}

    #### Topic Insights for {city} - {category}
    Below are the topics extracted from customer reviews for {city} in the {category} industry. Analyze these topics and provide an interpretation of how they contribute to the customer experience:
    
    #### Top Positive Topic words and the importance of each word for {city} - {category}
    {top_positive_topics}

    #### Top Negative Topic words and the importance of each word for {city} - {category}
    {top_negative_topics}

    ### Strategy for Success:
    Based on the analysis above, suggest the best strategy to succeed within this city and category to open my own place and be successful.

    ### Most Similar City-Category Pair Analysis:
    Now, compare the most similar city-category pair to {city} and {category}:

    - **Most Similar City-Category Pair**: {similar_city} - {similar_cat}

    #### Top Positive Words for the Most Similar City-Category Pair {similar_city} - {similar_cat}
    {similar_pos_words}

    #### Top Negative Words for the Most Similar City-Category Pair {similar_city} - {similar_cat}
    {similar_neg_words}

    """

    completion = client.chat.completions.create(
        model='gpt-4o-2024-11-20',
        # model="o3-mini-2025-01-31",
        # model='gpt-4o-mini-2024-07-18',
        messages=[
            {'role': 'system', "content": "You are an authority on food and beverage establishments. Help me anaylze review trends."},
            {"role": "user", "content": prompt},
        ],
        stream=True
    )

    for chunk in completion:
        if hasattr(chunk.choices[0].delta, "content"): 
            yield chunk.choices[0].delta.content
            time.sleep(0.02)

######### LAYOUT #########
agg_col = st.columns([12])

######### AGGREGATIONS #########
with agg_col[0]:
    with st.popover('Choose Category/City', use_container_width=True):
        filter_col1, filter_col2= st.columns([6, 6])
        with filter_col1:
            categories = st.pills(label='Choose category', options=load_categories(keywords))
        with filter_col2:
            cities = st.pills(label='Choose city', options=load_cities(keywords))

    tab1, tab2 = st.tabs([":star: Key Words", ":left_speech_bubble: DeepInsights"])

    with tab1:
        ######### REVIEWS #########
        if categories and cities:
            pos_df, neg_df = get_keyword_df(categories, cities)
            
            # Display the results
            st.subheader(f"Top Positive Words for {categories} - {cities}")
            chart = alt.Chart(pos_df).mark_bar().encode(
                x=alt.X('Impact:Q', title='Impact'),        
                y=alt.Y('Word:N', title='Word', sort=None),  
                color=alt.value("#00FF00")                  
            ).properties(
                width=600,
                height=400
            )
            st.altair_chart(chart)         

            st.subheader(f"Top Negative Words for {categories} - {cities}")
            chart = alt.Chart(neg_df).mark_bar().encode(
                x=alt.X('Impact:Q', title='Impact'),        
                y=alt.Y('Word:N', title='Word', sort=None),  
                color=alt.value("#FF0000")                  
            ).properties(
                width=600,
                height=400
            )
            st.altair_chart(chart)      
             
            similar_cat, similar_city = get_most_similar_city(f"{categories} - {cities}")
            
            # Get the good and bad words for the most similar city-category pair
            pos_df, neg_df = get_keyword_df(similar_cat, similar_city)
            # similar_pos_words, similar_neg_words = get_similar_keywords(most_similar_pair)
            
            # Display the good and bad words for the most similar city-category pair
            st.subheader(f"Top Positive Words for the Most Similar City-Category Pair ({similar_cat} - {similar_city})")
            chart = alt.Chart(pos_df).mark_bar().encode(
                x=alt.X('Impact:Q', title='Impact'),        
                y=alt.Y('Word:N', title='Word', sort=None),  
                color=alt.value("#00FF00")                  
            ).properties(
                width=600,
                height=400
            )
            st.altair_chart(chart)             
            
            st.subheader(f"Top Negative Words for the Most Similar City-Category Pair ({similar_cat} - {similar_city})")
            chart = alt.Chart(neg_df).mark_bar().encode(
                x=alt.X('Impact:Q', title='Impact'),        
                y=alt.Y('Word:N', title='Word', sort=None),  
                color=alt.value("#FF0000")                  
            ).properties(
                width=600,
                height=400
            )
            st.altair_chart(chart)                

            selected_pair = f"{categories} - {cities}"
            if selected_pair in topics['City_Category'].values:
                top_positive_topics, top_negative_topics = get_topic_words(categories, cities)

                st.subheader(f"Top Positive Topics for {categories} - {cities} including the topic words")
                st.markdown(f"- " + "\n- ".join(top_positive_topics))

                st.subheader(f"Top Negative Topics for {categories} - {cities} including the topic words")
                st.markdown(f"- " + "\n- ".join(top_negative_topics)) 

    with tab2:
        ######### CHAT #########
        if categories and cities:
            top_positive_keywords, _, top_negative_keywords, _ = process_keywords(categories, cities)

            similar_cat, similar_city = get_most_similar_city(f"{categories} - {cities}")

            similar_pos_words, _, similar_neg_words, _ = process_keywords(similar_cat, similar_city)

            response_container = st.container(height=600)

            if selected_pair in topics['City_Category'].values:
                top_positive_topics, top_negative_topics = get_topics(categories, cities)
                with response_container:
                    st.write_stream(query_llm_topics(categories, cities, similar_cat, similar_city, top_positive_keywords, top_negative_keywords, similar_pos_words, similar_neg_words, top_positive_topics, top_negative_topics))
            else: 
                with response_container:
                    st.write_stream(query_llm(categories, cities, similar_cat, similar_city, top_positive_keywords, top_negative_keywords, similar_pos_words, similar_neg_words))