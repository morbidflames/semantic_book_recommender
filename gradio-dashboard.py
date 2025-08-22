import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader  # Converts raw text to workable format
from langchain_text_splitters import CharacterTextSplitter   # Split descriptions into meaningful chunks
from langchain_huggingface import HuggingFaceEmbeddings      # Converting chunks to doc embeddings
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800" # To fetch bigger thumbnails

# When bigger thumbnails are unavailable
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load() # Reading tag descriptions into textloader
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0.1, chunk_overlap=0) #Instantiating splitter
# Applying to all docs to give individual book descriptions
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents,
                                 HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
) # Convert to doc embeddings & store


# Function to retrieve semantic recommendations from books dataset
# Filter based on category and sort based on emotions
def retrieve_semantic_recommendations(query:str,
                                      category: str = None,
                                      tone: str = None,
                                      initial_top_k: int = 50,
                                      final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k) # Getting recommendation from vector db
    # Getting isbns by splitting from descriptions
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    # Limit books df to only matching isbns
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    # Applying filtering based on category
    if category != "All":
        book_recs = book_recs[book_recs["category"] == category][:final_top_k]
    else:
        book_recs = book_recs.head(final_top_k)

    # Sorting emotions based on highest probability
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

# Function to display what we want on the gradio dashboard
def recommend_books(
        query:str,
        category: str,
        tone: str,
):
    recommendations = retrieve_semantic_recommendations(query, category, tone) # Get recomm df
    results = []

    # Loop over every single recommendations
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split() # Splitting the description to separate words
        # If description words > 30 cut it off and continue with trailing ellipses
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        # If book has more than 1 authors combine
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{','.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        # Display all information as caption
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


categories = ["All"] + sorted(books["simple_categories"].unique()) # List containing all categories
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"] # List containing all tones

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a Category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional Tone:", value = "All")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended Books", columns = 8,  rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)

if __name__ == "__main__":
    dashboard.launch()

