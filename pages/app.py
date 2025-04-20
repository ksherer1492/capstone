import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import polars as pl
import glob
import time
from openai import OpenAI


st.set_page_config(layout='wide')


keywords = ['worst',
'horrible',
'amazing',
'great',
 'delicious',
 'bad',
 'rude',
 'best',
 'love',
 'order',
 'friendly',
 'terrible',
 'reviews',
 'never',
 'highly',
 'ordered',
 'ok',
 'money',
 'sickening',
 'said',
 'spot',
 'charge',
 'robbed',
 'asked',
 'fast',
 'minutes',
 'favorite',
 'nice',
 'gem',
 'told',
 'perfect',
 'stolen',
 'fresh',
 'recommend',
 'overpriced',
 'phone',
 'really',
 'delivery',
 'waste',
 'super',
 'staff',
 'poor',
 'manager',
 'restaurant',
 'definitely',
 'called',
 'dinero',
 'excellent',
 'garbage',
 'try',
 'okay',
 'disgusting',
 'welcoming',
 'shame',
 'expensive',
 'also',
 'waited',
 'customer']


word_combinations = [('ever', 'worst'),
 ('food', 'worst'),
 ('service', 'worst'),
 ('order', 'worst'),
 ('service', 'horrible'),
 ('food', 'horrible'),
 ('customer', 'horrible'),
 ('order', 'horrible'),
 ('food', 'amazing'),
 ('service', 'amazing'),
 ('great', 'amazing'),
 ('place', 'amazing'),
 ('food', 'great'),
 ('service', 'great'),
 ('place', 'great'),
 ('good', 'great'),
 ('food', 'delicious'),
 ('service', 'delicious'),
 ('great', 'delicious'),
 ('friendly', 'delicious'),
 ('service', 'bad'),
 ('food', 'bad'),
 ('good', 'bad'),
 ('place', 'bad'),
 ('service', 'rude'),
 ('order', 'rude'),
 ('food', 'rude'),
 ('customer', 'rude'),
 ('ever', 'best'),
 ('food', 'best'),
 ('place', 'best'),
 ('service', 'best'),
 ('place', 'love'),
 ('food', 'love'),
 ('great', 'love'),
 ('always', 'love'),
 ('food', 'order'),
 ('time', 'order'),
 ('get', 'order'),
 ('service', 'order'),
 ('staff', 'friendly'),
 ('food', 'friendly'),
 ('great', 'friendly'),
 ('service', 'friendly'),
 ('service', 'terrible'),
 ('food', 'terrible'),
 ('order', 'terrible'),
 ('customer', 'terrible'),
 ('food', 'reviews'),
 ('place', 'reviews'),
 ('good', 'reviews'),
 ('bad', 'reviews'),
 ('food', 'never'),
 ('order', 'never'),
 ('always', 'never'),
 ('place', 'never'),
 ('recommend', 'highly'),
 ('food', 'highly'),
 ('great', 'highly'),
 ('recommended', 'highly'),
 ('food', 'ordered'),
 ('good', 'ordered'),
 ('order', 'ordered'),
 ('chicken', 'ordered'),
 ('food', 'ok'),
 ('good', 'ok'),
 ('service', 'ok'),
 ('place', 'ok'),
 ('food', 'money'),
 ('waste', 'money'),
 ('worth', 'money'),
 ('order', 'money'),
 ('food', 'sickening'),
 ('place', 'sickening'),
 ('get', 'sickening'),
 ('it', 'sickening'),
 ('order', 'said'),
 ('asked', 'said'),
 ('food', 'said'),
 ('back', 'said'),
 ('great', 'spot'),
 ('food', 'spot'),
 ('good', 'spot'),
 ('service', 'spot'),
 ('extra', 'charge'),
 ('food', 'charge'),
 ('order', 'charge'),
 ('service', 'charge'),
 ('got', 'robbed'),
 ('feel', 'robbed'),
 ('like', 'robbed'),
 ('order', 'robbed'),
 ('order', 'asked'),
 ('said', 'asked'),
 ('food', 'asked'),
 ('us', 'asked'),
 ('service', 'fast'),
 ('food', 'fast'),
 ('good', 'fast'),
 ('friendly', 'fast'),
 ('order', 'minutes'),
 ('food', 'minutes'),
 ('wait', 'minutes'),
 ('waited', 'minutes'),
 ('place', 'favorite'),
 ('one', 'favorite'),
 ('food', 'favorite'),
 ('great', 'favorite'),
 ('food', 'nice'),
 ('good', 'nice'),
 ('place', 'nice'),
 ('great', 'nice'),
 ('hidden', 'gem'),
 ('food', 'gem'),
 ('great', 'gem'),
 ('place', 'gem'),
 ('order', 'told'),
 ('us', 'told'),
 ('asked', 'told'),
 ('food', 'told'),
 ('food', 'perfect'),
 ('great', 'perfect'),
 ('place', 'perfect'),
 ('service', 'perfect'),
 ('phone', 'stolen'),
 ('card', 'stolen'),
 ('got', 'stolen'),
 ('order', 'stolen'),
 ('food', 'fresh'),
 ('always', 'fresh'),
 ('delicious', 'fresh'),
 ('great', 'fresh'),
 ('highly', 'recommend'),
 ('food', 'recommend'),
 ('great', 'recommend'),
 ('place', 'recommend'),
 ('food', 'overpriced'),
 ('good', 'overpriced'),
 ('service', 'overpriced'),
 ('place', 'overpriced'),
 ('order', 'phone'),
 ('answer', 'phone'),
 ('food', 'phone'),
 ('called', 'phone'),
 ('good', 'really'),
 ('food', 'really'),
 ('great', 'really'),
 ('place', 'really'),
 ('order', 'delivery'),
 ('food', 'delivery'),
 ('pizza', 'delivery'),
 ('fast', 'delivery'),
 ('money', 'waste'),
 ('time', 'waste'),
 ('dont', 'waste'),
 ('food', 'waste'),
 ('friendly', 'super'),
 ('food', 'super'),
 ('staff', 'super'),
 ('great', 'super'),
 ('friendly', 'staff'),
 ('great', 'staff'),
 ('food', 'staff'),
 ('good', 'staff'),
 ('service', 'poor'),
 ('customer', 'poor'),
 ('food', 'poor'),
 ('order', 'poor'),
 ('order', 'manager'),
 ('food', 'manager'),
 ('service', 'manager'),
 ('us', 'manager'),
 ('food', 'restaurant'),
 ('great', 'restaurant'),
 ('good', 'restaurant'),
 ('service', 'restaurant'),
 ('back', 'definitely'),
 ('food', 'definitely'),
 ('great', 'definitely'),
 ('service', 'definitely'),
 ('order', 'called'),
 ('said', 'called'),
 ('told', 'called'),
 ('food', 'called'),
 ('service', 'excellent'),
 ('food', 'excellent'),
 ('great', 'excellent'),
 ('good', 'excellent'),
 ('food', 'garbage'),
 ('like', 'garbage'),
 ('place', 'garbage'),
 ('pizza', 'garbage'),
 ('food', 'try'),
 ('place', 'try'),
 ('good', 'try'),
 ('great', 'try'),
 ('food', 'okay'),
 ('good', 'okay'),
 ('service', 'okay'),
 ('place', 'okay'),
 ('food', 'disgusting'),
 ('never', 'disgusting'),
 ('place', 'disgusting'),
 ('like', 'disgusting'),
 ('staff', 'welcoming'),
 ('food', 'welcoming'),
 ('great', 'welcoming'),
 ('friendly', 'welcoming'),
 ('food', 'shame'),
 ('place', 'shame'),
 ('order', 'shame'),
 ('its', 'shame'),
 ('good', 'expensive'),
 ('food', 'expensive'),
 ('little', 'expensive'),
 ('place', 'expensive'),
 ('good', 'also'),
 ('food', 'also'),
 ('great', 'also'),
 ('place', 'also'),
 ('minutes', 'waited'),
 ('order', 'waited'),
 ('food', 'waited'),
 ('hour', 'waited'),
 ('service', 'customer'),
 ('great', 'customer'),
 ('food', 'customer'),
 ('good', 'customer')]



######### DEFINE VARS #########
import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
openai_key = st.secrets['openai_key']
client = OpenAI(api_key=openai_key)


patterns = [rf"\b{word1}\b.*\b{word2}\b" for word1, word2 in word_combinations]
keywords = '|'.join(keywords)
review_paths = glob.glob('data/all/reviews/*.parquet')

######### LOAD DATA ######### 
establishments = (
    pl.scan_parquet('data/all/all_establishments.parquet')
    # .filter(pl.col('latitude').is_not_null())
    # .unique()
)
reviews = pl.concat([pl.scan_parquet(path) for path in review_paths])

# only return establishments that have reviews available
valid_fac_ids = reviews.unique('facility_id').select('facility_id').collect()
nyc_establishments = (
    establishments
    .filter(
        (True==True)
        & (pl.col('state') == "NY")
        & (pl.col('longitude').is_not_null())
        & (pl.col('average_rating').is_not_null())
        & (pl.col('facility_id').is_in(valid_fac_ids))
        )
    )


######### SESSION STATES #########
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
    st.session_state['messages'].append({'role': 'assistant', 'content': 'Hi! Please enter your abstract.'})


######### DEFINE FUNCTIONS ######### 

def load_categories():
    query = """
    with category_counts as (
    select
        category,
        count(category) as count
    from read_parquet('data/all/all_establishments.parquet') 
    where state = 'NY'
    group by category
    )

    select distinct category
    FROM category_counts
    where count >= 50
    order by category asc
    """ 
    return duckdb.query(query).df()


def load_filtered_reviews(fac_ids):
    query = f"""
    SELECT
        facility_id,
        text,
        rating
    FROM read_parquet('data/all/reviews/*.parquet')
    WHERE 
        True
        AND facility_id IN {fac_ids}    
        -- AND REGEXP_MATCHES(text, '{keywords}')
        AND text NOT NULL
    """
    return duckdb.query(query).df()


def single_query_llm(review_str):
    prompt = f"""
    Tell me the strengths and weaknesses of this places.
    Give a analysis of what customers like and dislike: {review_str}.
    """
    completion = client.chat.completions.create(
        model='gpt-4o-2024-11-20',
        # model="o3-mini-2025-01-31",
        # model='gpt-4o-mini-2024-07-18',
        messages=[
            {'role': 'system', "content": "You are an authority on food and beverage establishments. Help me anaylze review texts."},
            {"role": "user", "content": prompt},
        ],
        stream=True
    )

    for chunk in completion:
        if hasattr(chunk.choices[0].delta, "content"): 
            yield chunk.choices[0].delta.content
            time.sleep(0.02)



def query_llm(review_str):
    prompt = f"""
    Tell me the strengths and weaknesses of these places. Don't mention specific places. 
    Give a general analysis of what customers like and dislike: {review_str}.
    Then tell me the speicifc best place by name and why it's the best. Then the specific worst place by name and why it's the worst.
    Then tell me the best strategy to succeed among these places such that I can open my own place nearby and be successful.
    """

    completion = client.chat.completions.create(
        model='gpt-4o-2024-11-20',
        # model="o3-mini-2025-01-31",
        # model='gpt-4o-mini-2024-07-18',
        messages=[
            {'role': 'system', "content": "You are an authority on food and beverage establishments. Help me anaylze review texts."},
            {"role": "user", "content": prompt},
        ],
        stream=True
    )

    for chunk in completion:
        if hasattr(chunk.choices[0].delta, "content"): 
            yield chunk.choices[0].delta.content
            time.sleep(0.02)


######### LAYOUT #########
main_filter_col, _ = st.columns([6, 6])

map_col, agg_col = st.columns([6, 6])

######### FILTERS #########
with main_filter_col:
    with st.popover('Filters', use_container_width=True):
        filter_col1, filter_col2= st.columns([3, 3])
        with filter_col1:
            # categories = st.multiselect(label='Choose categories', options=load_categories())
            categories = st.pills(label='Choose categories', options=load_categories(), selection_mode='multi')
        with filter_col2:
            st.write('placeholder')



######### MAP #########
with map_col:
    map_df = nyc_establishments.collect().to_pandas()
    if categories:
        map_df = map_df.query('category.isin(@categories)')

    color_min = map_df['average_rating'].quantile(0.025)
    color_max = map_df['average_rating'].quantile(0.85)
    map_selection = st.plotly_chart(
        px.scatter_map(
            data_frame=map_df, 
            lat='latitude', 
            lon='longitude',
            zoom=12,
            center=dict(lat=40.7473666, lon=-73.9902979),
            color='average_rating',
            color_continuous_scale="RdYlGn",
            range_color=[color_min, color_max],
            opacity=0.75,
            map_style='carto-darkmatter',
            hover_name='restaurant_name',
            custom_data=['restaurant_name', 'average_rating', 'score']
            ).update_traces(
                hovertemplate=('%{customdata[0]}<br>'
                                'GoogleMaps Rating: %{customdata[1]}<br>'
                                ),
                marker=dict(size=10)
            ).update_layout(
                width=800,
                height=800
            ).update_coloraxes(
                showscale=False
            )
            , 
        on_select='rerun',
        use_container_width=True
        )

######### AGGREGATIONS #########
with agg_col:
    tab1, tab2, tab3 = st.tabs(["📈 Ratings Over Time", ":star: Reviews", ":left_speech_bubble: DeepInsights"])

    with tab1:
        ######### LINE CHART #########
        if map_selection.selection['point_indices']:
            map_selection_idx = map_selection.selection['point_indices']
            fac_ids = map_df.iloc[map_selection_idx]['facility_id'].unique()
        else:
            fac_ids = map_df.facility_id.unique()
        fac_ids = tuple(fac_ids)
        
        ratings_timeseries = (nyc_establishments
                            .filter(pl.col('facility_id').is_in(fac_ids))
                            .select('facility_id')
                            .join(reviews, on='facility_id')
                            .with_columns(pl.col('timestamp').cast(pl.Datetime('us')))
                            .with_columns(pl.col('timestamp').dt.strftime('%Y-%m').alias('year_month'))
                            .filter(pl.col('timestamp').dt.year() >= 2020)
                            .group_by('year_month')
                            .agg(pl.col('rating').mean().alias('monthly_rating'))
                            .sort(by='year_month')
                            .with_columns(rolling_mean=pl.col('monthly_rating').rolling_mean(window_size=6))
                            .collect()
                            )

        st.plotly_chart(
            px.line(
                data_frame=ratings_timeseries,
                x='year_month',
                y='rolling_mean'
            ).update_layout(
                yaxis=dict(range=[2, 5]),
                height=600
            )
        )

    with tab2:
        ######### REVIEWS #########
        filtered_reviews = load_filtered_reviews(fac_ids)
        if map_selection.selection['point_indices']:
            agg_df = map_df.iloc[map_selection_idx].sort_values(by='average_rating', ascending=False)

            agg_df = pd.merge(agg_df[['facility_id', 'google_name']], 
                              filtered_reviews, 
                              left_on='facility_id', 
                              right_on='facility_id')
            st.dataframe(agg_df, hide_index=False, column_config={'facility_id': None, 'google_name': 'Restaurant', 'text': 'review'})

            concise_reviews = (
                reviews
                .join(establishments.select('facility_id', 'restaurant_name', 'average_rating'), on='facility_id')
                .filter(
                    (True == True)
                    & pl.col('facility_id').is_in(agg_df['facility_id'].to_list())
                ) 
                .head(100000)
                .with_columns(
                    pl.sum_horizontal([pl.col("text").str.count_matches(p, literal=False) for p in patterns]).alias("match_count")
                    )
                .with_columns((pl.col('match_count') / pl.col('text').str.len_chars()).alias('match_ratio'))
                .filter(pl.col("match_count") > 0)
                .sort(by='match_ratio', descending=True)
                .collect()
                .unique(['facility_id', 'text', 'timestamp', 'rating'])
                )

            st.dataframe(
                concise_reviews
                .filter(
                    (True == True)
                    & (pl.col.match_ratio.is_between(0.0001, 0.02)) 
                )
                .unique(['facility_id', 'text', 'timestamp', 'rating'])
                .with_row_index('id')
            )
            st.dataframe(
                concise_reviews
                .filter(
                    (True == True)
                    & (pl.col.match_ratio.is_between(0.0001, 0.01)) 
                )
                .unique(['facility_id', 'text', 'timestamp', 'rating'])
                .select(pl.col('rating').value_counts())
            )

            # else:
            #     st.dataframe(filtered_reviews[['text', 'rating']], hide_index=True)
        else:
            st.markdown('# Please make selections on the map.')

        

    with tab3:
        ######### CHAT #########
        filtered_reviews = load_filtered_reviews(fac_ids)
        if map_selection.selection['point_indices']:
            agg_df = map_df.iloc[map_selection_idx].sort_values(by='average_rating', ascending=False)
 

            agg_df = pd.merge(agg_df[['facility_id', 'google_name']], 
                                filtered_reviews, 
                                left_on='facility_id', 
                                right_on='facility_id')

            tab2_reviews = (
                reviews
                .join(establishments.select('facility_id', 'restaurant_name', 'average_rating'), on='facility_id')
                .filter(
                    (True == True)
                    & pl.col('facility_id').is_in(agg_df['facility_id'].to_list())
                ) 
                .head(100000)
                .with_columns(
                    pl.sum_horizontal([pl.col("text").str.count_matches(p, literal=False) for p in patterns]).alias("match_count")
                    )
                .with_columns((pl.col('match_count') / pl.col('text').str.len_chars()).alias('match_ratio'))
                .filter(pl.col("match_count") > 0)
                .sort(by='match_ratio', descending=True)
                .collect()
                .unique()
                )

            reviews_str = (
                tab2_reviews
                .filter(
                    (True == True)
                    & (pl.col.match_ratio.is_between(0.001, 0.01)) 
                )
                .with_columns(reviews_str=pl.concat_str([pl.col('restaurant_name'), pl.col('text')], separator='\n'))
                .sort(by='rating')
                .unique()
                .head(700)
                .select(pl.col('reviews_str').str.join('\n\n').alias('reviews_str'))
            )

            reviews_str = (
                tab2_reviews
                .filter(
                    (True == True)
                    & (pl.col.match_ratio.is_between(0.001, 0.01)) 
                )
                .with_row_index('id')
            )

            response_container = st.container(height=600)
            with response_container:
                if len(map_selection.selection['point_indices']) == 1:
                    st.write_stream(single_query_llm(reviews_str))
                else:
                    st.write_stream(query_llm(reviews_str))
                # with st.chat_message('Gordon Ramsay'):
                #     st.markdown(openai_response)

        # response_container = st.container(height=600)
        # input_container = st.container()


        # with input_container:
        #     if prompt := st.chat_input('Enter your abstract', max_chars=3000):
        #         # with st.chat_message('user'):
        #         #     st.write(prompt)
        #         st.session_state['messages'].append({'role': 'user', 'content': prompt})
        #         # with st.chat_message("assistant"):
        #         #     response = st.write(prompt)
        # with response_container:
        #     for i, message in enumerate(st.session_state['messages']):
        #         with st.chat_message(message['role']):
        #             st.markdown(message['content'])