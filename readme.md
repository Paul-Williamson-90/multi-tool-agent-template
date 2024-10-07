# (WIP) Project Template for Multi-Tool LLM Agents with Llama-Index

*Note:* This template is yet to be tested with a working version, tests are yet to be written.

# Instructions
- View example.py for a working example of how to use the multi-tool agent router and provide new skills to it.
- You will need to create a .env in the root of the project folder with the following keys:
```
OPENAI_API_KEY=...
```

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

# Creating a New Skill
- example.py has an example of defining a new skill (multiplication) and adding it to the LLM router's toolkit.
- When the LLM router chooses a skill, it will create a structured output (JSON) in string form as its written response and triggers the use of a tool. The args for the chosen tool are contained in the "input" key of the resulting dictionary (as can be seen on src.agents.router.AgentFlowOpenAI.tool_call_handler).
- The args are parsed via src.skills.base.FunctionCallSkill.handle_router_input and then passed to the execute method.
- The src.skills.base.SkillArgAttr objects passed to the src.skills.base.FunctionCallSkill class you create define what args the LLM router should add to the tool call.
- To create a new skill, create a class that inherits from the src.skills.base.FunctionCallSkill. All you need to ensure is that the src.skills.base.SkillArgAttr objects passed to your Skill match those that are required for running the execute method you define. 
