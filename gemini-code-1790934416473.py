import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load Datasets
clients = pd.read_csv('clients.csv')
properties = pd.read_csv('properties.csv')

# 2. Data Cleaning & Feature Engineering
def parse_dob(val):
    for fmt in ('%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y'):
        try:
            return pd.to_datetime(val, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(val, errors='coerce')

clients['dob_dt'] = clients['date_of_birth'].apply(parse_dob)
today = pd.to_datetime('2026-01-01')
clients['age'] = (today - clients['dob_dt']).dt.days // 365

properties['clean_price'] = properties['sale_price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
client_prop = properties.groupby('client_ref').agg(
    total_spent=('clean_price', 'sum'),
    units_purchased=('listing_id', 'count'),
    avg_floor_area=('floor_area_sqft', 'mean')
).reset_index()

merged = pd.merge(clients, client_prop, left_on='client_id', right_on='client_ref', how='left')
merged['total_spent'] = merged['total_spent'].fillna(0)
merged['units_purchased'] = merged['units_purchased'].fillna(0)

# 3. Feature Scaling & K-Means Clustering
features = merged[['age', 'satisfaction_score', 'total_spent', 'units_purchased']].copy()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=4, random_state=42)
merged['cluster'] = kmeans.fit_predict(scaled_features)

# Cluster Summary
cluster_summary = merged.groupby('cluster').agg({
    'age': 'mean',
    'satisfaction_score': 'mean',
    'total_spent': 'mean',
    'units_purchased': 'mean',
    'client_id': 'count'
}).rename(columns={'client_id': 'client_count'})

print("Cluster Summary:\n", cluster_summary)
merged.to_csv('segmented_clients_output.csv', index=False)