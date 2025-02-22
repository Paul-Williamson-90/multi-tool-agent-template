from textwrap import dedent

from llama_index.core import PromptTemplate

SYSTEM_PROMPT = """You are a helpful assistant that chooses a tool to call based on the user's request. Today's date is {date}."""

ACTION_DECISION_INSTRUCTIONS = """# Instructions
- **Your available actions are either to generate a response back to the user or use a tool provided. \
All of your responses should be a tool call or text using the JSON JSON schema provided at the bottom of this prompt.**
"""

TOOL_DECISION_INSTRUCTIONS = """# Instructions
- **Select the tools using the JSON schema provided at the bottom of this prompt as according to your prior thought process.**
"""

RESPONSE_INSTRUCTIONS = PromptTemplate(
    dedent(
        """# SYSTEM\n
        <system>{system}</system>\n\n

        # CHAT HISTORY AND PREVIOUS TOOL CALLS\n
        <chat history>{chat_history}</chat history>\n\n

        {thoughts}\n\n

        # Instructions\n
        - Now generate the response to the user based on your prior thoughts.\n
        - You are encourages to use the following formatting to provide clarity in your response back to the user:\n
        \t- # Headers and ## Subheaders, using '#' markers.\n
        \t- **Bold** and *italic* text, using either single '*' for italics or double '**' for bold.\n
        \t- New lines, tabs, bullet points and numbered lists.\n
        \t- No other formatting is allowed.\n
        """
    )
)

ROUTER_AGENT_PROMPT_TEMPLATE = PromptTemplate(
    dedent(
        """# SYSTEM\n
        <system>{system}</system>\n
        {instructions}\n\n
        
        # TOOLS AVAILABLE\n
        <tools>{tools}</tools>\n\n
        
        # CHAT HISTORY AND PREVIOUS TOOL CALLS\n
        <chat history>{chat_history}</chat history>\n\n
        
        {thoughts}\n\n
        
        **Below is the schema for which your JSON output must be formatted as (YOU MUST ensure that you are double escaping any control characters that \
        are not inside strings):**\n
        {schema}
        """
    )
)

CHAT_HISTORY_CONDENSE = PromptTemplate(
    dedent(
        """You are an agentic ChatBot currently in conversation with a user, however your context window is too small \
        to fit the entire chat history into the prompt. Your task is to condense the chat history so that only the most relevant information \
        is retained. Relevance should be determined on how useful the information is to the user's current query. \
        You must use concise bullet points for each piece of information to ensure the chat history is easy to read and understand. \
        If the current messages to be condensed are irrelevant, simply output a blank string.\n\n
        
        # USER'S LAST MESSAGE FOR CHECKING RELEVANCE AGAINST\n
        <user last message>{user_last_message}</user last message>\n\n
        
        # CONDENSED CHAT HISTORY SO FAR...\n
        <condensed>{condensed}</condensed>\n\n
        
        # CURRENT MESSAGES TO BE CONDENSED\n
        <current message>{current_message}</current message>
        """
    )
)


ESCAPE_PROMPT = PromptTemplate(
    dedent(
        """# SYSTEM\n
        <system>{system}</system>\n\n

        You are running into some issues providing the user with the information they've requested:\n
        <hint>{hint}</hint>\n\n

        Here is the last message from the user that you were trying to answer:
        <user>{user_last_msg}</user>\n\n

        Here is your internal dialogue so far:
        <internal dialogue>{internal_messages}</internal dialogue>\n\n

        Your task is to generate a response back to the user apologising, and following the hint provided. \
        Your response must be conversational and supportive.
        """
    )
)

USER_INTENT_CONDENSE = PromptTemplate(
    dedent(
        """# SYSTEM\n
        <system>{system}</system>\n\n
        
        The user has sent you a message and your task is now to use the last few messages in the chat to summarise the user's intent / what they are asking of you. \
        THe last message is the response from the user.
        
        You should re-write the user's last message from the perspective of the user but making sure that any \
        relevant information from prior messages they are referring to is included. For example when the user refers to information in a \
        prior message but does not directly state it in their following message (presupposition).\n\n
        
        If there is no prior message that contains relevant information to the user's current query, simply repeat the user's last message.\n\n
        
        # CHAT HISTORY\n
        <chat history>{chat_history}</chat history>\n\n"""
    )
)

FIX_JSON_PROMPT = PromptTemplate(
    dedent(
        """This string is meant to be JSON however is raising a JSONDecodeError. \
        Without changing any of the key-value pairs, you must fix the JSON string so that it is valid JSON. \
        Typically the issues reside in unescaped newlines that aren't inside strings, or unescaped control characters.\n\n
        
        Only output the fixed JSON string, do NOT include any other information.\n\n
        
        # JSON TO FIX\n
        {response}
        """
    )
)

CONDENSED_TEMPLATE = """**This is a condensed chat history to save space:**
{condensed}

**This is the user's last message:**
{user_last_message}

**This is your internal dialogue so far:**
{thoughts}
"""

ROUNDS_EXCEEDED_HINT = "You are struggling to find the relevant data to answer the user's query. \
Suggest the user to provide more context or rephrase their query."

ERROR_HINT = "You seem to be running into issues, ask the user to try again or provide more context."