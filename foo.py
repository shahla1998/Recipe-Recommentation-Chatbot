import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('/Users/mishabulhaq/Desktop/food/Ifood_new.csv')
df = df.dropna(subset=['name', 'ingredients'])

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['ingredients'].astype(str))

def recommend_recipe(user_ingredients, veg_only):
    if not user_ingredients.strip():
        return "ദയവായി കുറച്ച് ഇൻഗ്രീഡിയൻസ് ടൈപ്പ് ചെയ്യുക!"
    
    filtered_df = df.copy()
    
    if veg_only and 'diet' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['diet'].astype(str).str.lower() == 'vegetarian']
    
    if filtered_df.empty:
        return "ഈ ഫിൽട്ടറിന് അനുയോജ്യമായ വിഭവങ്ങൾ ഒന്നും കണ്ടില്ല."

    temp_tfidf = vectorizer.transform(filtered_df['ingredients'].astype(str))
    user_vec = vectorizer.transform([user_ingredients])
    
    similarity = cosine_similarity(user_vec, temp_tfidf)
    score = similarity.max()
    
    if score < 0.15:
        return "Sorry, I didn't find any recipes that use these ingredients."
    
    index = similarity.argmax()
    recipe = filtered_df.iloc[index]
    
    diet_info = recipe.get('diet', 'N/A')
    prep_info = recipe.get('prep_time', 'N/A')
    cook_info = recipe.get('cook_time', 'N/A')
    
    result = f"""
*Dishes:* {recipe['name']}
 *Part:* {recipe['diet'].title()}
 *Preparation time:* {recipe['prep_time']} mint
 *Cooking time:* {recipe['cook_time']} mint

*Required ingredients:*
{recipe['ingredients']}
"""
    return result

demo = gr.Interface(
    fn=recommend_recipe,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter ingredients e.g., egg, onion / rice flour, ghee...", label="Ingredients"),
        gr.Checkbox(label="Vegetarian Only 🥗")
    ],
    outputs=gr.Textbox(lines=8, label="Recipe Result"),
    title="🍳 Indian Recipe Recommendation Bot",
    description="Enter the ingredients you and discover suitable indian dishes!"
)

demo.launch()