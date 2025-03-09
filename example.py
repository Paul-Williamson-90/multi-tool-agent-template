import logging
import warnings

from working_example.router import get_agent, invoke

logging.basicConfig(level=logging.INFO)  # NOTE: comment this line to disable logging
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    agent = get_agent()

    while True:
        user_input = input("\nUser (type 'exit' to exit): ")
        if user_input == "exit":
            break
        response = invoke(agent, user_input)
        print("Assistant: ", end="")
        for words in response.chat_stream:
            if words.delta:
                print(words.delta, end="", flush=True)
