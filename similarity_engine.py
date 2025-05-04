import pandas as pd
import numpy as np
import streamlit as st

def run_dot_product_similarity(selected_geoid, income_path, race_path, return_df=False):
    @st.cache_data
    def load_data():
        income_df = pd.read_csv(income_path, dtype={'income_census_tract_FIPS': str})
        race_df = pd.read_csv(race_path, dtype={'race_census_tract_FIPS': str})
        return income_df, race_df

    @st.cache_data
    def load_dictionary():
        return pd.read_csv("data/all/census/full_census_dictionary.csv")

    income_df, race_df = load_data()
    dictionary_df = load_dictionary()

    # Merge on FIPS codes
    merged_df = pd.merge(
        income_df, race_df,
        left_on='income_census_tract_FIPS',
        right_on='race_census_tract_FIPS'
    )

    merged_df['FIPS'] = merged_df['income_census_tract_FIPS']
    merged_df['city'] = merged_df.get('state_x', '').map({
        'NY': 'NYC', 'CA': 'LA', 'IL': 'Chicago', 'TX': 'Austin'
    }).fillna("Unknown")

    numeric = merged_df.select_dtypes(include=np.number)
    scaled = (numeric - numeric.mean()) / numeric.std()

    if selected_geoid in merged_df['FIPS'].values:
        vec = scaled[merged_df['FIPS'] == selected_geoid].values[0]
        dot_scores = scaled.dot(vec)
        min_score, max_score = dot_scores.min(), dot_scores.max()
        normalized_scores = 100 * (dot_scores - min_score) / (max_score - min_score)
        merged_df['dot_similarity'] = normalized_scores

        top10 = merged_df[merged_df['FIPS'] != selected_geoid]\
            .sort_values(by='dot_similarity', ascending=False)\
            .query("dot_similarity < 100")\
            .head(10)

        # Contributions
        contributions = pd.Series(vec, index=scaled.columns)
        contributions_abs = contributions.abs().sort_values(ascending=False)
        top_features = contributions.loc[contributions_abs.index[:10]]

        contrib_df = pd.DataFrame({
            'feature': top_features.index,
            'contribution': top_features.values
        })
        contrib_df = contrib_df.merge(dictionary_df[['field_name', 'description']],
                                      left_on='feature', right_on='field_name', how='left')
        contrib_df = contrib_df[['description', 'contribution']].rename(
            columns={'description': 'Feature Description'}
        )

        if return_df:
            return top10[['FIPS', 'dot_similarity']], contrib_df

        # If not returning, show directly
        st.subheader("Top 10 Most Similar Tracts (Dot Product Similarity)")
        st.dataframe(top10[['FIPS', 'city', 'dot_similarity']])

        st.subheader("Top 10 Contributing Features to Similarity")
        st.dataframe(contrib_df)

    else:
        st.error("Selected GEOID not found in merged dataset.")
        return pd.DataFrame(), pd.DataFrame()
