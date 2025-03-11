import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import polars as pl
import glob
import pickle


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
 ('tortillas', 'corn'),
 ('good', 'corn'),
 ('tacos', 'corn'),
 ('food', 'corn'),
 ('food', 'sickening'),
 ('place', 'sickening'),
 ('get', 'sickening'),
 ('it', 'sickening')]

patterns = [rf"\b{word1}\b.*\b{word2}\b" for word1, word2 in word_combinations]

review_paths = glob.glob('data/all/reviews/*.parquet')

######### LOAD DATA ######### 
establishments = pl.scan_parquet('data/all/all_establishments.parquet')
reviews = pl.concat([pl.scan_parquet(path) for path in review_paths])
nyc_establishments = (establishments
                      .filter(
                          (True==True)
                          & (pl.col('state') == "NY")
                          & (pl.col('longitude').is_not_null())
                          & (pl.col('average_rating').is_not_null())
                          )
                      )


concise_reviews = (
    reviews
    .join(establishments.select('facility_id', 'restaurant_name', 'average_rating'), on='facility_id')
    .filter(
        (True == True)
        & (pl.col("text").is_not_null())
        & (pl.col('average_rating') <= 3.8)
    ) 
    .head(100000)
    .with_columns(
        pl.sum_horizontal([pl.col("text").str.count_matches(p, literal=False) for p in patterns]).alias("match_count")
        )
    .with_columns((pl.col('match_count') / pl.col('text').str.len_chars()).alias('match_ratio'))
    .filter(pl.col("match_count") > 0)
    .sort(by='match_ratio', descending=True)
    
    .collect()
    )


st.dataframe(
    concise_reviews
    .filter(
        (True == True)
        & (pl.col.rating != 3)
        & (pl.col.match_ratio >= 0.01)
    )
    .with_row_index('id')
)