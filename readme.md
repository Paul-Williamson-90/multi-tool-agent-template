# (WIP) Project Template for Multi-Tool LLM Agents with Llama-Index

*Note:* This template is yet to be tested with a working version, tests are yet to be written.

# Instructions
View example.py for a working example of how to use the multi-tool agent router and provide new skills to it.

# Code Base Explained
## src.agents.router.AgentFlowOpenAI
This class is the router LLM that will receive a text input, and return a response. It has tools available to it via 'Skills' which are defined by the programmer and passed via the SkillMap class (src.skills.base.SkillMap).
## src.skills.base.FunctionCallSkill
This class is a parent class for 'Skills' which can be passed to the router LLM and available for use when answering input text queries. 
- def execute(self, args) -> str: The core method which the router LLM will use when activating the tool.
## src.skills.base.SkillArgAttr
Pydantic class for handling arg attributes for a FunctionCallSkill. The attributes are used to inform the router LLM on how to use the tool available to it, and what args are required/available.
## src.skills.base.SkillMap
A class for hosting multiple skills and provided to the router LLM.
