from textwrap import dedent

from llama_index.core import PromptTemplate

SYSTEM_PROMPT = """You are a helpful AI ChatBot. Today's date is {date}."""


ACTION_DECISION_INSTRUCTIONS = """# Instructions
- **Your available actions are either to generate a response back to the user or use a tool provided. \
All of your responses should be a tool call or text using the JSON JSON schema provided at the bottom of this prompt.**
"""


TOOL_DECISION_INSTRUCTIONS = """# Instructions
- **Select the tools using the JSON schema provided at the bottom of this prompt as according to your prior thought process.**
"""


CONTEXT_SELECTION_INSTRUCTIONS = """# Instructions
- **Select the context you want to use to generate a response back to the user using the schema defined below.**
- **If you do not need any additional information to generate a useful response to the user, do not select any contexts.**
- **As part of context selection, you will need to specify the type of facts you want to extract from the context \
that will be useful for generating a response to the user.**
- **Once selected, the information extracted from the contexts will be provided to you at a later stage.**
- **You can select multiple contexts if needed.**
"""


RESPONSE_INSTRUCTIONS = PromptTemplate(
    dedent(
        """# SYSTEM\n
        <system>{system}</system>\n\n

        {chat_history}\n\n

        # ASSISTANT'S THOUGHTS AND TOOL CALLS:\n
        {thoughts}\n\n

        # Instructions\n
        <instructions>- Now generate the response to the user based on your prior thoughts.\n
        - You are encourages to use the following formatting to provide clarity in your response back to the user:\n
        \t- # Headers and ## Subheaders, using '#' markers.\n
        \t- **Bold** and *italic* text, using either single '*' for italics or double '**' for bold.\n
        \t- New lines, tabs, bullet points and numbered lists.\n
        \t- No other formatting is allowed.\n</instructions>\n\n
        """
    )
)


ROUTER_AGENT_PROMPT_TEMPLATE = PromptTemplate(
    dedent(
        """# SYSTEM:\n
        <system>{system}\n
        {instructions}</system>\n\n

        # TOOLS AVAILABLE:\n
        <tools>{tools}</tools>\n\n

        {chat_history}\n\n

        # ASSISTANT'S THOUGHTS AND TOOL CALLS:\n
        {thoughts}"""
    )
)


ESCAPE_PROMPT = PromptTemplate(
    dedent(
        """You are running into some issues providing the user with the information they've requested:\n
        <hint>{hint}</hint>\n\n

        Your task is to generate a response back to the user apologising, and following the hint provided. \
        Your response must be conversational and supportive.
        """
    )
)


ROUNDS_EXCEEDED_HINT = "You are struggling to find the relevant data to answer the user's query. \
Suggest the user to provide more context or rephrase their query."


ERROR_HINT = "You seem to be running into issues, ask the user to try again or provide more context."
