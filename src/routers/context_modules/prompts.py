from llama_index.core import PromptTemplate


EXTRACT_TEMPLATE = PromptTemplate(
    """# SYSTEM:
<system>Given a brief from the user and some retrieved contexts, you must examine the brief and extract the facts from the contexts \
that relates to the brief. The extraction should be purely factual without any embellishments. \
If the user is asking for counts of things (e.g. such as a count of occurrences of an entity) you can ignore \
this part of the brief as the information on this will automatically be provided - you must still extract facts for \
the rest of the brief. You MUST ensure that any fact that is extracted is on a per document basis rather than summarising facts across \
multiple documents (i.e. each fact is independent and can only have one reference). If multiple documents in the context \
have the same/similar fact, you must create an individual fact for each document that contains that fact.</system>

# BRIEF:
<brief>{brief}</brief>

# CONTEXTS:
<contexts>{contexts}</contexts>
"""
)


SUMMARISE_TEMPLATE = PromptTemplate(
    """# SYSTEM:
<system>Given a query from the user and a collection of documents, you must summarise the documents based on the query. \
The summary should be concise and informative, providing the user with a brief overview of the documents in relation \
ti the query. You do not need to provide any summary statistics such as counts as these are provided automatically.</system>

# QUERY:
<query>{query}</query>

# CONTENT:
<content>{content}</content>

# CONTENT SUMMARY:
<summary>{summary}</summary>
"""
)