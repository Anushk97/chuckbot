def extract_keywords(ticket_content):
    # Enhanced keyword extraction
    keywords = ["login", "auth", "refunds", "deployment", "preview", "user"]
    extracted = [word for word in keywords if word in ticket_content.lower()]
    return extracted

def process_ticket(ticket_data):
    title = ticket_data.get("title", "")
    content = ticket_data.get("content", "")
    keywords = extract_keywords(content)
    print("Processed Ticket:")
    print("Title:", title)
    print("Content:", content)
    print("Keywords:", keywords)

def main():
    # Example ticket data
    ticket = {
        "id": "001",
        "title": "Account Access",
        "content": "Having issues with login, authentication, and user access."
    }
    process_ticket(ticket)

if __name__ == "__main__":
    main()
