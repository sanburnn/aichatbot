def needs_web_search(answer: str) -> bool:
    if not answer:
        return True

    keywords = [
        "i don't know",
        "i am not sure",
        "no information",
        "I don't have information",
        "I'm not aware of any information",
        "as an ai",
        "cannot",
        "not sure",
        "unknown",
        "sorry",
        "no data",
        "not available",
        "I don't have real-time information",
        "real-time information"
    ]

    a = answer.lower()
    return any(k in a for k in keywords)
