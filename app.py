from model import load_model
from chat import generate_text

def main():
    # Change model here later (e.g. "qwen-7b")
    tokenizer, model = load_model("qwen-1.5b")

    print("\nQwen Chatbot (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Bye 👋")
            break

        response = generate_text(tokenizer, model, user_input)
        print(f"AI: {response}\n")

if __name__ == "__main__":
    main()
