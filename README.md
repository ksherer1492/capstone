# So, You Want To Be A Restaurant Owner?

**Capstone Project – Master of Science in Data Science & Analytics**

This Streamlit dashboard is designed for aspiring or current restaurant owners seeking **data-driven insights** into optimal locations and business strategies in four major U.S. cities: **Chicago, Austin, New York, and Los Angeles**.

## Objective

To answer these key business questions:

- What separates highly reviewed restaurants from poorly reviewed ones?
- What key factors help determine the best restaurant locations across different cities?
- How can restaurant owners leverage customer sentiment and competition data to make smarter decisions?

## What the Dashboard Provides

This interactive dashboard provides **location-specific business intelligence** and **strategic guidance** to help restaurant owners:

- Identify high-potential areas using income and demographic data.
- Understand local competition and customer sentiment from Google reviews.
- Tailor offerings and positioning based on neighborhood characteristics.
- Compare cities and spot underserved markets.

## Dashboard Pages & Features

### 1. **City Category Analysis**
- Identifies local strengths, weaknesses, and consumer preferences.
- Allows cross-city market comparisons by cuisine and price tier.

### 2. **Similarity Measure**
- Select a census tract and view the top 10 most demographically similar tracts.
- Uses dot product similarity across standardized features.
- Visualizes the most important traits contributing to similarity.

### 3. **Census Tract Clustering Analysis**
- Groups tracts using a precomputed similarity matrix.
- Explore clusters of neighborhoods with shared economic and racial characteristics.
- Gain regional insights into community-level market behavior.

### 4. **Search Census Tract**
- Enter any U.S. address to find the corresponding census tract.
- Get real-time guidance based on tract-level data.
- Highlight areas on the map and receive targeted **strategic recommendations**:
  - Competition density and pricing
  - Google review sentiment trends
  - Suggestions on how to best **position your restaurant** to align with local demand

Each page includes an **About** section explaining the underlying methods and how they support data-driven decision-making.

## Data Sources

- [Google Maps API](https://developers.google.com/maps/documentation)
- [U.S. Census Bureau](https://www.census.gov/data.html)
- Public restaurant inspection datasets
- Google Places & Review Data
- GeoJSON & shapefiles for spatial analysis

##  Disclaimer of Data Bias

- **Survivorship Bias**: Data may favor restaurants that have survived long enough to accumulate reviews.
- **Nonresponse Bias**: Census data reflects only respondents; weights and margins of error are applied to adjust.

## Getting Started

### Run Locally

1. Clone the repo:
    ```bash
    git clone https://github.com/ksher
