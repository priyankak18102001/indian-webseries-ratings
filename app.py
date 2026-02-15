import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Web Series Dashboard", layout="wide")

st.title("Indian Web Series Recommendation Dashboard")

# Load dataset
df = pd.read_csv("indian_webseries_ratings.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Convert to long format
df_melted = df.melt(id_vars=["User_ID"], 
                    var_name="Series", 
                    value_name="Rating")

df_melted = df_melted.dropna()

# ---- Average Ratings Chart ----
st.subheader("Average Ratings of Web Series")

avg_rating = df_melted.groupby("Series")["Rating"].mean().sort_values(ascending=False)

fig, ax = plt.subplots()
avg_rating.plot(kind="bar", ax=ax)
st.pyplot(fig)

# ---- Top Rated Series ----
st.subheader("Top Rated Series")
st.write(avg_rating.head(5))

# ---- Recommendation Section ----
st.subheader("Get Recommendations")

user_list = df["User_ID"].unique()
selected_user = st.selectbox("Select User", user_list)

# Recommendation logic (simple similarity)
ratings_matrix = df.set_index("User_ID").fillna(0)
similarity = cosine_similarity(ratings_matrix)
similarity_df = pd.DataFrame(similarity, 
                             index=ratings_matrix.index, 
                             columns=ratings_matrix.index)

similar_users = similarity_df[selected_user].sort_values(ascending=False)[1:3]

recommended_series = ratings_matrix.loc[similar_users.index].mean().sort_values(ascending=False)

st.write("Recommended Series:")
st.write(recommended_series.head(5))
