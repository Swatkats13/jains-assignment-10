import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify, render_template
from scipy.spatial.distance import cosine
from open_clip import create_model_and_transforms, get_tokenizer

# Configuration
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
IMAGE_DIR = "static/coco_images_resized"
EMBEDDINGS_PATH = "data/image_embeddings.pickle"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load precomputed embeddings
df = pd.read_pickle(EMBEDDINGS_PATH)
df["embedding"] = df["embedding"].apply(lambda x: torch.tensor(x, device=DEVICE))

# Load CLIP model and tokenizer
model, _, preprocess = create_model_and_transforms("ViT-B/32", pretrained="openai")
model = model.to(DEVICE)
model.eval()
tokenizer = get_tokenizer("ViT-B-32")


def calculate_similarity(query_embedding, top_k=5):
    """Calculate cosine similarity between query embedding and dataset embeddings."""
    cosine_similarities = df["embedding"].apply(lambda emb: 1 - cosine(query_embedding.cpu().numpy(), emb.cpu().numpy()))
    top_indices = cosine_similarities.nlargest(top_k).index
    results = [
        {
            "file_name": f"/static/coco_images_resized/{df.loc[i, 'file_name']}",  # Ensure correct relative path to static
            "similarity": cosine_similarities[i],
        }
        for i in top_indices
    ]
    return results

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Render the main page and handle search requests.
    """
    if request.method == "POST":
        query_type = request.form.get("query_type")
        query_embedding = None
        image_embedding = None
        text_embedding = None

        # Handle image query
        if query_type in ["image", "hybrid"]:
            image_file = request.files.get("image_query")
            if image_file:
                image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
                image_file.save(image_path)
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
                image_embedding = F.normalize(
                    model.encode_image(image), p=2, dim=1
                ).detach()
                print("Image Embedding:", image_embedding)

        # Handle text query
        if query_type in ["text", "hybrid"]:
            query_text = request.form.get("text_query")
            if query_text:
                text_tokens = tokenizer([query_text]).to(DEVICE)
                text_embedding = F.normalize(
                    model.encode_text(text_tokens), p=2, dim=1
                ).detach()
                print("Text Embedding:", text_embedding)

        # Handle hybrid query
        if query_type == "hybrid" and image_embedding is not None and text_embedding is not None:
            weight = float(request.form.get("hybrid_weight", 0.5))
            print(f"Hybrid Query Weight: {weight}")
            query_embedding = F.normalize(
                weight * text_embedding + (1 - weight) * image_embedding, p=2, dim=1
            ).squeeze(0)
            print("Hybrid Query Embedding:", query_embedding)
        elif query_type == "image":
            query_embedding = image_embedding.squeeze(0)
        elif query_type == "text":
            query_embedding = text_embedding.squeeze(0)

        # Calculate similarity
        if query_embedding is not None:
            print("Query Embedding for Similarity Calculation:", query_embedding)
            results = calculate_similarity(query_embedding)
            print("Results:", results)
            return jsonify(results)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=3000)
