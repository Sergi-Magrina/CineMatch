from .recommender import ContentRecommender


rec = ContentRecommender(title_weight=0.15, use_titles=True).fit_from_movies_csv("data/movies.csv")

print(rec.recommend("Toy Story", k=10, exclude_franchise=True))
print(rec.recommend_mmr("Toy Story", k=10, lambda_val=0.75))
