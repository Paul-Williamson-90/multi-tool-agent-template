from src.routers.skills.base import FunctionCallSkill, SkillArgAttr, SkillOutput


class Multiply(FunctionCallSkill):
    def __init__(
        self,
        name: str = "multiply",
        description: str = "Use this tool to multiply two numbers.",
        function_args=[
            SkillArgAttr(
                name="a", dtype="float", description="First number", required=True
            ),
            SkillArgAttr(
                name="b", dtype="float", description="Second number", required=True
            ),
        ],
        visible_to_human: bool = True,
    ):
        super().__init__(
            name=name,
            description=description,
            function_args=function_args,
            visible_to_human=visible_to_human,
        )

    def execute(self, a: float, b: float) -> float:  # type: ignore
        return SkillOutput(response_to_llm=f"{a} times {b} is {a*b}")
