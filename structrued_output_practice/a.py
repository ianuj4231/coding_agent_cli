import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv()


class ContactInfo(BaseModel):
    """Contact details extracted from text."""

    name: str = Field(description="The person's full name")
    email: str = Field(description="The person's email address")


model = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

agent = create_agent(
    model=model,
    tools=[],
    response_format=ContactInfo,
)


if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Extract the contact: Anuj, anuj@example.com",
                }
            ]
        }
    )

    contact = result["structured_response"]
    print(contact)
    print(contact.name)
    print(contact.email)
