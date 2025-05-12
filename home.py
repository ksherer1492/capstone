import streamlit as st

pages = {
    'Home Page': [st.Page('pages/actual_home.py', title="Home Page")],
    "Census Tract Analysis": [
        st.Page("pages/Census Tract Clustering Analysis.py", title="Census Tract Clustering Analysis"),
        st.Page("pages/similarity_measure.py", title="Similarity Measure"),
    ],
    "Text Analysis": [
        st.Page("pages/app.py", title="Customer Sentiment"),
        st.Page("pages/City Category Analysis.py", title="City Category Analysis"),
    ],
}

pg = st.navigation(pages)
pg.run()