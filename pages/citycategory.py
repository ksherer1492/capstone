import streamlit as st
import pandas as pd
import time
from openai import OpenAI
import altair as alt
from PIL import Image

st.set_page_config(layout='wide')
st.title('City/Category Pair Trends and Analysis')
st.subheader('Select a City and Category pair from the drop down below')
st.text('This page displays the most predictive terms/phrases/topics of high and low performing restaurants for your chosen pair')
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
    topics = pd.read_csv('data/all/citycategory/topic_data.csv', usecols=['City_Category', 'Sentiment', 'Words', 'topic_words', 'Coefficient'])
    return keywords, similarity, topics

keywords, similarity, topics = load_data()

@st.cache_resource
def load_image(path):
    return Image.open(path)

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

    top_positive_topic_words = filtered_df_pos[['topic_words', 'Coefficient']]
    top_negative_topic_words = filtered_df_neg[['topic_words', 'Coefficient']]

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

    tab1, tab2, tab3 = st.tabs([":star: Key Words", ":left_speech_bubble: Deep Insights", ":left_speech_bubble: About"])

    with tab1:
        ######### REVIEWS #########
        if categories and cities:
            st.subheader("Predictive Words/Phrases Derived from TF-IDF and Ridge Regression Coefficients")
            pos_df, neg_df = get_keyword_df(categories, cities)
            
            # Display the results
            st.text(f"Top Positive Words for {categories} - {cities}")
            chart = alt.Chart(pos_df).mark_bar().encode(
                x=alt.X('Impact:Q', title='Impact'),        
                y=alt.Y('Word:N', title='Word', sort=None),  
                color=alt.value("#00FF00")                  
            ).properties(
                width=600,
                height=400
            )
            st.altair_chart(chart)         

            st.text(f"Top Negative Words for {categories} - {cities}")
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
            st.subheader("Predictive Words/Phrases Derived from TF-IDF and Ridge Regression Coefficients of the most similar City based on Cosine Similarity")
            # Display the good and bad words for the most similar city-category pair
            st.text(f"Top Positive Words for {similar_cat} - {similar_city}")
            chart = alt.Chart(pos_df).mark_bar().encode(
                x=alt.X('Impact:Q', title='Impact'),        
                y=alt.Y('Word:N', title='Word', sort=None),  
                color=alt.value("#00FF00")                  
            ).properties(
                width=600,
                height=400
            )
            st.altair_chart(chart)             
            
            st.text(f"Top Negative Words for {similar_cat} - {similar_city}")
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
                st.subheader("Predictive Topics Derived from BERTopic and Ridge Regression Coefficients")
                top_positive_topics, top_negative_topics = get_topic_words(categories, cities)

                st.text(f"Top Positive Topics for {categories} - {cities} including the topic words")
                chart = alt.Chart(top_positive_topics).mark_bar().encode(
                    x=alt.X('Coefficient:Q', title='Impact'),        
                    y=alt.Y('topic_words:N', title='Topic Words', sort=None),  
                    color=alt.value("#00FF00")                  
                ).properties(
                    width=600,
                    height=400
                )
                st.altair_chart(chart) 
                # st.markdown(f"- " + "\n- ".join(top_positive_topics))

                st.text(f"Top Negative Topics for {categories} - {cities} including the topic words")
                chart = alt.Chart(top_negative_topics).mark_bar().encode(
                    x=alt.X('Coefficient:Q', title='Impact'),        
                    y=alt.Y('topic_words:N', title='Topic Words', sort=None),  
                    color=alt.value("#FF0000")                  
                ).properties(
                    width=600,
                    height=400
                )
                st.altair_chart(chart) 
                # st.markdown(f"- " + "\n- ".join(top_negative_topics)) 
            else:
                st.text("There are no topics for this city/category pair.")

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

    with tab3:
        ######### ABOUT #########
        st.title("Analysis of Average Restaurant Rating Prediction Using Review from City/Category Pairs")
        st.subheader("This tab will cover a description of our analytical approach and example results that reflect how this area of analysis was able to help answer our key questions. For additional results use the first two tabs of this page to generate city/category pair results of your choosing by selecting them from the above dropdown. Additionally, check out the deep insights tab to get a GPT generated report based on the insights from your selected pair.")
        st.subheader("Overview")
        st.text("We refined our data by standardizing restaurant categories by ensuring we only focus on restaurants that have been assigned a category and drop any that have not. This allows us to use the same grouping of restaurants across each area of analysis. This plays a big role in this particular page as it relies on the different restaurant categories.")
        st.text("This analysis like the other review analysis page focuses on reviews from the past 5 years. This ensures we analyze only the most relevant and actionable information and reflects what was obtainable with our web scraping strategy.")
        st.text("Since our review analysis is made up of two separate forms of analysis this page covers the city/category based analysis which focuses on identifying localized strengths, weaknesses, and preferences for city/category pairs in our dataset. This allows us to give a broad look at these markets and compare them.")
        st.text("For our review trend analysis by city and category we grouped our data into the proper city/category pairs then refined our data by removing stop words like and, but, and or and lemmatizing the text which brings words to their base form for example turning running and ran into just run. Next reviews were broken down into words/phrases through the use of TF-IDF which creates importance scores for each word/phrase based on how often it appears in a specific review compared to all reviews allowing us to identify the most frequently relevant/important words in reviews. We then measured how similar city/category groupings were based on these term frequency vectors through a cosine similarity analysis. At the same time we used BERTopic alongside TF-IDF to identify topics tied to city/category pairs. Using both the term frequency and BERTopic results as inputs in separate models we uncovered city/category specific keywords and topics that drive high average review scores.")
        st.text("The categories featured on this page were created to ensure a substantial sample size by defining macro class categories through restaurant names and existing google categories of which a majority of our reviews can be classified into the top 7 shown below.")
        st.image(load_image("images/categorydist.png"))
        st.subheader("Term Frequency Analysis")
        st.text("For our TF-IDF strategies we decided to go with TF-IDF as it helps us understand how important a word or phrase is in one review, compared to all other reviews in the dataset. This means we’re not just looking at what words are common, but what words are distinctive and meaningful. Before applying TF-IDF, we pre-processed the text. That included removing stop words and using lemmatization to reduce words to their base forms. We used both unigrams and bigrams so single words and two-word combinations to capture more context and nuance. To keep things both manageable and detailed, we focused on the top 5,000 features based on their TF-IDF scores. Finally, by averaging those scores across all reviews, we were able to highlight which terms consistently stood out as important across each dataset.")
        st.image(load_image("images/tfchi.png"))
        st.text("We can see above our term frequency results for the American and fast food category in the city of Chicago. We can see that important terms include popular menu items like burgers and chicken. Another interesting insight is the frequent mention of Chicago. This tells us there may be a strong local identity, whether that’s in the style of food or location. Atmosphere also appeared to be an important factor for this pairing.")
        st.text("Taking the results from the previous image we can see in the below image that through our cosine similarity analysis that is based on the term frequency results for each city/category pair Austin is the most similar city to Chicago in the American and fast food category. Based on these visualizations we see that frequent terms remain the same across these pairs like service, burger, chicken, and friendly which suggests that customer expectations and experiences in these two cities align closely within this category.")
        st.image(load_image("images/cosine.png"))
        st.subheader("Term Frequency Modeling Results")
        st.text("Above we identified imporant words/phrases for the American and fast food category in the city of Chicago, but which of them are actually predictive of restaurant ratings. For this part of the analysis we target average restaurant rating using ridge regression which works well with high-dimensional data which comes in the form of our term frequency data by using regularization which penalizes high coefficients to minimize overfitting. We tuned these models through the alpha hyperparameter which affects this regularization through the use of GridSearchCV with mean squared error as our performance metric. We split our data 80/20 using train test split to give gridsearch a large enough subset of data to do its own splits for optimization.")
        st.text("Below is a result from these models for the American and fast food category in the city of Chicago. Here we are looking at the top predictive words/phrases based on coefficients for each word or phrase gathered from the trained models. The higher the coefficient the more influence it has in predicting a high average rating. We can see popular menu items such as Donuts and steak sandwiches along with brunch offering and reservation options are predictors in this particular market.")
        st.image(load_image("images/tfchimodel.png"))
        st.text("Below is another look at a different set of results for the city/category pair we identified as the most similar in the earlier slide. Here we can see that along with similarities in important words/phrases there are similarities in predictors as brunch offerings stand out in both markets. Despite this there are still a lot of unique predictors across similar markets in this case we see burgers as an important menu item and specific establishments listed such as In n Out. We also see the focus on Austin now instead of Chicago.")
        st.image(load_image("images/tfaustinmodel.png"))
        st.subheader("Topic Analysis")
        st.text("Now that we’ve studied how individual terms or phrases can be used in predicting average restaurant rating we also looked at how higher-level themes or topics performed in comparison. For this portion we used BERTopic to extract themes from reviews then represented each review as a probability distribution over these generated themes/topics. We were able to use these topic probabilities as inputs in additional ridge regression models while still targeting average restaurant rating.")
        st.text("For our BERTopic strategies first, during feature extraction, we again focused on unigrams and bigrams while filtering out stop words and very low-frequency terms to reduce noise. We let BERTopic automatically determine the number of topics allowing the model to adapt based on the structure of the data rather than a fixed number of topics. We also set a minimum cluster size on top of this to ensure that the topics created were substantial and informative enough. To keep topic interpretations clear and focused, we concentrated on the top 10 words per topic. This helped ensure that each topic label remained relevant and easy to understand.")
        st.subheader("Topic Modeling Results")
        st.image(load_image("images/tfvstopic.png"))
        st.text("As you can see in the image above when comparing our two types of Ridge regression models, both of which we scored based on mean squared error and tuned through gridsearch we found that TF-IDF consistently outperformed BERTopic in predictive accuracy.")
        st.text("One limitation we faced was that only the top 7 categories which were identified above had enough data to support meaningful topic modeling with BERTopic. This limited the scope of comparison.")
        st.image(load_image("images/topicchimodel.png"))
        st.text("Despite not performing as strongly as our term frequency models we can still derive insights by viewing the topics with high coefficient scores like we did for our term frequency models. In the image above we can see that some of the predictive topics for the fast food market in Chicago focus on quality food and service, celebration offerings, and popular menu items like donuts. Interestingly, donuts also appeared as a predictive term in our term frequency model results, which points to consistent trends across both of our modeling approaches.")
        st.subheader("Final Results")
        st.image(load_image("images/naive.png"))
        st.text("The results detailed above which are only a subset of what this page can show reveal a clear relationship between the language used in reviews whether it’s specific words, phrases, or broader topics and a restaurant's overall rating as can be seen in the visualization above where our average MSE for both the term frequency and topic models beat out the respective average MSE of Naive mean baseline models.")
        st.text("By analyzing this data, we’re able to confidently identify localized predictors that drive restaurant ratings in different cities and food categories. For example, we saw how terms like donuts were consistently linked to higher ratings in Chicago’s fast food market. This helps answer the question \"What separates highly reviewed restaurants from poorly reviewed ones?\"")
        st.text("Additionally, by comparing these city/category pairs we are able to identify similarities between markets that may assist new or existing restaurant owners in determining the best cities for expansion which helps answer the question \"What key factors can help restaurant owners determine the best locations for business among New York, Chicago, Austin, and Los Angeles?\"")
        st.text("What makes this approach especially valuable is that it’s both scalable and repeatable. These strategies can be applied to new cities and categories not currently found on this dashboard helping businesses stay on top of customer trends and make data-informed decisions.")





