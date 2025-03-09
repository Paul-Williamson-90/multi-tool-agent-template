# Project Template for Multi-Tool LLM Agents with Llama-Index
A multi-tool agent template with Llama-Index using the Workflow event-driven architecture. This template has been extended with some custom functionality designed for various use-cases I've had to design for.

**NOTE: This repo was recently updated with some new functionality and is yet to be thoroughly tested.**

![Workflow](basic_workflow.png)

# Example
See example.py file in the root of the project for a working demonstration.

# Template Explained
## RouterAgent (src.routers.base.RouterAgent)
This class is the router LLM that will receive a text input, and return a response. It has tools available to it via 'Skills' which are defined by the programmer and passed via the SkillMap class. Additionally, the router agent can be fitted with a condense_module for condensing the chat history (improved context window management) and ensuring the user's message has relevant context from prior messages. Finally, context_modules can be added for managing large retrieved context.

## FunctionCallSkill (src.routers.skills.base.FunctionCallSkill)
This class is a parent class for 'Skills' which can be passed to the router LLM and available for use when answering input text queries.

## SkillMap (src.routers.skills.base.SkillMap)
A class for hosting multiple skills and provided to the router LLM.

## CondenseModuleBase (src.routers.condensers.CondenseModuleBase)
The parent class for defining the process for condensing the chat history to only the relevant information to the current user query, as well as re-writing the user's query to solve issues of where the user uses presupposition.

## ContextModuleBase (src.routers.context_modules.ContextModuleBase)
Parent class for memory modules that the router agent can access during response generation. The intention of this class is for use with large context retrieval (such as documents from a database). The class ensures that the retrieved context does not impair the router agent's reasoning process by only showing the retrieved context during the response step. The router agent also has tools to verify if the context contains the information it seeks, and uses concurrent fact extraction to condense the documents to only the relevant information the router agent needs in a response.