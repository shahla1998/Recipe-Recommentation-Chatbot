import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('/Users/mishabulhaq/Desktop/food/Ifood_new.csv')

df = df.dropna(subset=['name', 'ingredients'])

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['ingredients'].astype(str))

def get_recipe(user_ingredients, veg_only=False):
    filtered_df = df.copy()
    

    if veg_only:
        filtered_df = filtered_df[filtered_df['diet'].str.lower() == 'vegetarian']
    
    temp_tfidf = vectorizer.transform(filtered_df['ingredients'].astype(str))
    user_vec = vectorizer.transform([user_ingredients])
    
    similarity = cosine_similarity(user_vec, temp_tfidf)
    
    score = similarity.max()
    if score < 0.15:
        return "Sorry, I didn't find any recipes that use these ingredients."
        
    index = similarity.argmax()
    recipe = filtered_df.iloc[index]
    
    return f"""
 *Dishes:* {recipe['name']}
 *Part:* {recipe['diet'].title()}
 *Preparation time:* {recipe['prep_time']} mint
 *Cooking time:* {recipe['cook_time']} mint

 *Required ingredients:*
{recipe['ingredients']}
"""

print("--- Indian cooking bot is ready ---")

while True:
    user_input = input("\Type the ingredients (exit 'exit'): ")
    if user_input.lower() in ['exit', 'bye']:
        print("Good Bye!")
        break
        
    print(get_recipe(user_input))