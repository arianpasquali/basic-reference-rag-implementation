import os
from enum import Enum
from pydantic import BaseModel, Field
import openai

class SafetyAssessment(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    ERROR = "error"

class GuardrailsOutput(BaseModel):
    safety_assessment: SafetyAssessment = Field(description="The safety assessment of the content.")
    unsafe_categories: list[str] = Field(
        description="If content is unsafe, the list of unsafe categories.", default=[]
    )

class OpenAIModerator:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("OPENAI_API_KEY not set, skipping moderation")
            self.enabled = False
        else:
            self.enabled = True

    def invoke(self, text: str) -> GuardrailsOutput:
        if not self.enabled:
            return GuardrailsOutput(safety_assessment=SafetyAssessment.SAFE)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.moderations.create(input=text)
            result = response.results[0]
            flagged = result.flagged
            categories = [k for k, v in result.categories.model_dump().items() if v]
            if flagged:
                return GuardrailsOutput(
                    safety_assessment=SafetyAssessment.UNSAFE,
                    unsafe_categories=categories,
                )
            else:
                return GuardrailsOutput(safety_assessment=SafetyAssessment.SAFE)
        except Exception as e:
            print(f"OpenAI Moderation API error: {e}")
            return GuardrailsOutput(safety_assessment=SafetyAssessment.ERROR)

    async def ainvoke(self, text: str) -> GuardrailsOutput:
        if not self.enabled:
            return GuardrailsOutput(safety_assessment=SafetyAssessment.SAFE)
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key)
            response = await client.moderations.create(input=text)
            result = response.results[0]
            flagged = result.flagged
            categories = [k for k, v in result.categories.model_dump().items() if v]
            if flagged:
                return GuardrailsOutput(
                    safety_assessment=SafetyAssessment.UNSAFE,
                    unsafe_categories=categories,
                )
            else:
                return GuardrailsOutput(safety_assessment=SafetyAssessment.SAFE)
        except Exception as e:
            print(f"OpenAI Moderation API error: {e}")
            return GuardrailsOutput(safety_assessment=SafetyAssessment.ERROR)

if __name__ == "__main__":
    moderator = OpenAIModerator()
    # Example: test with unsafe content
    output = moderator.invoke("What's a good way to harm an animal?")
    print(output)
    # Example: test with safe content
    output = moderator.invoke("How do I change a tire on my car?")
    print(output)
