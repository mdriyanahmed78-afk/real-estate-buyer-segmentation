import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Parcl Real Estate Market Intelligence", layout="wide")

st.title("🏢 Real Estate Market Intelligence: Buyer Segmentation & Investment Profiling")
st.markdown("AI-driven clustering platform for Parcl Co. Limited and Unified Mentor.")

@st.cache_data
def load_data():
    return pd.read_csv('segmented_clients_output.csv')

try:
    df = load_data()
except Exception as e:
    st.error("Please ensure segmented_clients_output.csv is uploaded to your repository.")
    st.stop()

st.sidebar.header("Filter Controls")
selected_country = st.sidebar.selectbox("Country", ["All"] + list(df['country'].unique()))
selected_type = st.sidebar.selectbox("Client Type", ["All"] + list(df['client_type'].unique()))
selected_purpose = st.sidebar.selectbox("Acquisition Purpose", ["All"] + list(df['acquisition_purpose'].unique()))

filtered_df = df.copy()
if selected_country != "All":
    filtered_df = filtered_df[filtered_df['country'] == selected_country]
if selected_type != "All":
    filtered_df = filtered_df[filtered_df['client_type'] == selected_type]
if selected_purpose != "All":
    filtered_df = filtered_df[filtered_df['acquisition_purpose'] == selected_purpose]

st.subheader("📊 Buyer Segmentation Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Clients", len(filtered_df))
col2.metric("Avg Investment ($)", f"${filtered_df['total_spent'].mean():,.2f}")
col3.metric("Avg Satisfaction", f"{filtered_df['satisfaction_score'].mean():.2f} / 5")
col4.metric("Avg Units Purchased", f"{filtered_df['units_purchased'].mean():.1f}")

st.markdown("---")
st.subheader("📈 Investor Behavior Dashboard")
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=filtered_df, x='age', y='total_spent', hue='cluster', palette='Set1', ax=ax, s=60)
ax.set_title("Age vs. Total Investment by Cluster Segment")
st.pyplot(fig)

st.subheader("📝 Segment Insights Panel")
st.dataframe(filtered_df[['client_id', 'client_type', 'country', 'acquisition_purpose', 'satisfaction_score', 'total_spent', 'cluster']].head(50))
