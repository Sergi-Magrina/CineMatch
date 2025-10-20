import streamlit as st
from src.recommender import ContentRecommender

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")
st.title("🎬 CineMatch — Content-Based Movie Recommender")

@st.cache_resource
def load_model():
    # genres dominate, titles lightly weighted
    return ContentRecommender(title_weight=0.15, use_titles=True).fit_from_movies_csv("data/movies.csv")

rec = load_model()

left, right = st.columns([2,1])
with left:
    title = st.text_input("Movie title", "Toy Story")
with right:
    k = st.slider("How many recommendations?", 5, 20, 10)

mode = st.radio("Ranking mode", ["Pure similarity", "Diversified (MMR)"], index=1, horizontal=True)
exclude = st.checkbox("Exclude sequels / same franchise (for Pure)", value=True)
lambda_val = st.slider("MMR λ (higher = more similar, lower = more diverse)", 0.5, 0.95, 0.80)

if st.button("Recommend"):
    if mode == "Diversified (MMR)":
        out = rec.recommend_mmr(title, k=k, lambda_val=lambda_val)
    else:
        out = rec.recommend(title, k=k, include_scores=True, exclude_franchise=exclude)
    if isinstance(out, str):
        st.warning(out)
    else:
        st.dataframe(out, use_container_width=True)

