import pandas as pd
from pathlib import Path

# Try to read data/movies_sample.csv; fallback to movies.csv if user replaces it
data_path = Path(__file__).parent / "data" / "movies_sample.csv"
if not data_path.exists():
    data_path = Path(__file__).parent / "data" / "movies.csv"

try:
    df = pd.read_csv(data_path)
    print("\n✅ Loaded:", data_path.name)
    print("\nFirst 10 rows:\n", df.head(10))
    print("\nColumns:", list(df.columns))
    print("\nRow count:", len(df))
except Exception as e:
    print("⚠️ Could not load CSV:", e)