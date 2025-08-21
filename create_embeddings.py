from sentence_transformers import SentenceTransformer

# 1. Load a pre-trained multilingual model.
# We choose a multilingual model because the sample data contains Vietnamese.
# 'all-MiniLM-L6-v2' is great but primarily for English.
# 'paraphrase-multilingual-MiniLM-L12-v2' is a good choice for multiple languages.
print("Loading sentence transformer model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Model loaded.")

# 2. Your raw data
data = [
    {"keyframe_id": "L01_V001_00000000", "ocr_text": ["HTV9 HD", "06:29:59", "60 giây"], "object": ["cityscape", "river", "buoy", "boat"]},
    {"keyframe_id": "L01_V001_00000016", "ocr_text": ["HTV9", "HD", "06:29:59", "60", "giây"], "object": ["buildings", "city skyline"]},
    {"keyframe_id": "L01_V001_00000049", "ocr_text": ["HTV9", "HD", "06:30:01", "60 giây"], "object": ["person", "news desk", "studio set"]}
]

# 3. Process the data to create embeddings
data_to_insert = []
for item in data:
    # Combine the text fields into a single, meaningful string
    ocr_content = " ".join(item["ocr_text"])
    object_content = ", ".join(item["object"])
    combined_text = f"Ocr text: {ocr_content}. Detected objects: {object_content}."

    # Generate the embedding for the combined text
    # The .tolist() converts the numpy array to a standard Python list
    embedding = model.encode(combined_text).tolist()

    # Structure the data for Milvus insertion
    data_to_insert.append({
        "keyframe_id": item["keyframe_id"],
        "embedding": embedding,
        "ocr_text_raw": ocr_content, # Storing the original text is highly recommended
        "objects_raw": object_content
    })

# 4. Print the results to show the final structure
for item in data_to_insert:
    print("\n--------------------")
    print(f"Keyframe ID: {item['keyframe_id']}")
    print(f"Raw OCR Text: {item['ocr_text_raw']}")
    print(f"Raw Objects: {item['objects_raw']}")
    # Print the first 5 dimensions of the embedding vector for brevity
    print(f"Embedding Vector (first 5 dims): {item['embedding'][:5]}...")
    print(f"Embedding Vector Dimension: {len(item['embedding'])}")

print("\nProcessing complete. The 'data_to_insert' list is now ready.")
print("Each item contains the keyframe ID, the original text, and the generated vector embedding.")
print("You can now use a Milvus client to insert this data into your collection.")
