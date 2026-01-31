from sentence_transformers import SentenceTransformer, util

def extract_keywords(ticket_content):
    keywords = ["login", "auth", "refunds", "deployment", "preview", "user"]
    extracted = [word for word in keywords if word in ticket_content.lower()]
    return extracted

def semantic_match(ticket_content, document_keys):
    # Load pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')
  
    # Encode sentences
    content_embedding = model.encode(ticket_content, convert_to_tensor=True)
    keys_embedding = model.encode(document_keys, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(content_embedding, keys_embedding)
    
    best_match = None
    if similarities.size(0) > 0:
        idx = similarities.argmax()
        best_match = document_keys[idx]

    return best_match

def process_ticket(ticket_data, document_keys):
    title = ticket_data.get("title", "")
    content = ticket_data.get("content", "")
    keywords = extract_keywords(content)
    best_match = semantic_match(content, document_keys)
    
    print("Processed Ticket:")
    print("Title:", title)
    print("Content:", content)
    print("Keywords:", keywords)
    if best_match:
        print("Best Semantic Match:", best_match)

def main():
    document_keys = ["login failure", "authentication problem", "refund process", "deployment issues"]
    ticket = {
        "id": "001",
        "title": "Account Access",
        "content": "Having issues with login and authentication."
    }
    process_ticket(ticket, document_keys)

if __name__ == "__main__":
    main()
