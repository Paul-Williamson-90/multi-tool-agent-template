from llama_index.utils.workflow import draw_all_possible_flows

from src.routers.base import RouterAgent

draw_all_possible_flows(RouterAgent, filename="basic_workflow.html")
