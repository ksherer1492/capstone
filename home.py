import streamlit as st

apps = {
    'Home Page': [st.Page('apps/actual_home.py', title="Home Page")],
    "Text Analysis": [
        st.Page("apps/City Category Analysis.py", title="City Category Analysis"),
        st.Page("apps/app.py", title="Customer Sentiment"),
    ],
    "Census Tract Analysis": [
        st.Page("apps/similarity_measure.py", title="Similarity Measure"),
        st.Page("apps/Census Tract Clustering Analysis.py", title="Census Tract Clustering Analysis"),
    ],
}

pg = st.navigation(apps)
pg.run()