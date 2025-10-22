# CineMatch — A Movie Recommendation System

A smart, content-based movie recommender built with Python  
You can explore it **directly on the web** or **run it locally**.

---

##  Two Ways to Use CineMatch

### 1️ Web App (Recommended)
CineMatch is deployed with **Streamlit Cloud** for easy access — no installation required.

**Steps:**
1. Go to the live app link: [https://cinematch123.streamlit.app/](https://cinematch123.streamlit.app/) 
2. Type any movie title in the input box (e.g., `Toy Story`).
3. Choose:
   - Number of recommendations
   - Mode: **Pure similarity** or **Diversified (MMR)**
   - Whether to exclude sequels or same-franchise films  
4. Click **“Recommend”** to instantly see the top similar movies!

Each result includes:
-  Movie title  
-  Similarity score  
-  Genres  

---

### 2️⃣ Local Compiler (Developer Mode)
If you want to test or modify the algorithm locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/Sergi-Magrina/CineMatch.git
   cd CineMatch
