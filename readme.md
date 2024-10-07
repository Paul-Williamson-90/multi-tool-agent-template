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

# Creating a New Skill
- example.py has an example of defining a new skill (multiplication) and adding it to the LLM router's toolkit.
- When the LLM router chooses a skill, it will create a structured output (JSON) in string form as its written response and triggers the use of a tool. The args for the chosen tool are contained in the "input" key of the resulting dictionary (as can be seen on src.agents.router.AgentFlowOpenAI.tool_call_handler).
- The args for the "execute" method of a src.skills.base.FunctionCallSkill can be access from the resulting "input" dictionary as follows:
```python
def execute(self, args: dict) -> Union[int, float]:
    if isinstance(args, dict) and "input" in args:
        args = args["input"]
    else:
        return 'Invalid input: expected a dictionary with the key "input" that\'s value is a dictionary.'
    
    answer = args["a"] * args["b"]
    return f"The answer is {answer}."
```
- The src.skills.base.SkillArgAttr objects passed to the src.skills.base.FunctionCallSkill class you create define what args the LLM router should add to the tool call.
- To create a new skill, create a class that inherits from the src.skills.base.FunctionCallSkill and follow the above when constructing the execute method. The execute method can do pretty much whatever you want it to, so long as it returns a string (images can also be sent if you were to configure the router LLM to also receive images as a VLM).
