import pandas as pd
import numpy as np
import streamlit as st

def run_dot_product_similarity(selected_geoid, income_path, race_path):
    @st.cache_data
    def load_data():
        income_df = pd.read_csv(income_path, dtype={'income_census_tract_FIPS': str})
        race_df = pd.read_csv(race_path, dtype={'race_census_tract_FIPS': str})
        return income_df, race_df

    income_df, race_df = load_data()

    # merge on FIPS codes
    merged_df = pd.merge(
        income_df, race_df,
        left_on='income_census_tract_FIPS',
        right_on='race_census_tract_FIPS'
    )

    # create unified FIPS key
    merged_df['FIPS'] = merged_df['income_census_tract_FIPS']

    # define city from income dataset state column
    state_to_city = {
        'NY': 'NYC',
        'CA': 'LA',
        'IL': 'Chicago',
        'TX': 'Austin'
    }

    # use state_x for city assignment
    if 'state_x' in merged_df.columns:
        merged_df['city'] = merged_df['state_x'].map(state_to_city).fillna("Unknown")
    else:
        merged_df['city'] = "Unknown"

    # select only numeric features for similarity calculation
    numeric = merged_df.select_dtypes(include=np.number)

    # standardize features
    scaled = (numeric - numeric.mean()) / numeric.std()

    # verify selected GEOID exists
    if selected_geoid in merged_df['FIPS'].values:
        vec = scaled[merged_df['FIPS'] == selected_geoid].values[0]

        # compute dot product similarity
        dot_scores = scaled.dot(vec)

        # normalize similarity scores to 0–100
        min_score = dot_scores.min()
        max_score = dot_scores.max()
        normalized_scores = 100 * (dot_scores - min_score) / (max_score - min_score)

        merged_df['dot_similarity'] = normalized_scores

        # filter top 10 most similar tracts (excluding self)
        top10 = merged_df[merged_df['FIPS'] != selected_geoid]\
            .sort_values(by='dot_similarity', ascending=False)\
            .query("dot_similarity < 100")\
            .head(10)

        st.subheader("Top 10 Most Similar Tracts (Dot Product Similarity)")
        st.dataframe(top10[['FIPS', 'city', 'dot_similarity']])
    else:
        st.error("Selected GEOID not found in merged dataset.")




