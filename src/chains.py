"""
LangChain Model Initialization, LLMChain, Message Handling, and Streaming.
Includes Python 3.14 / 3.15 compatibility for LLMChain.
"""
from typing import Dict, Any, Generator, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

# Safe import for LLMChain across Python 3.10 - 3.15+
try:
    from langchain.chains import LLMChain
except TypeError:
    class LLMChain:
        """
        Python 3.14/3.15+ compatible LLMChain wrapper using LangChain's
        Runnable pipeline (prompt | llm) when legacy Chain hits annotationlib bug.
        """
        def __init__(self, llm: ChatOpenAI, prompt: Any):
            self.llm = llm
            self.prompt = prompt
            self.chain = prompt | llm

        def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            result = self.chain.invoke(inputs)
            content = result.content if hasattr(result, "content") else str(result)
            return {**inputs, "text": content}

        def run(self, *args: Any, **kwargs: Any) -> str:
            inputs = args[0] if args and isinstance(args[0], dict) else kwargs
            return self.invoke(inputs)["text"]

from src.prompts import (
    SYSTEM_SAFETY_PROMPT,
    STRUCTURED_CHAT_PROMPT_TEMPLATE,
    NARRATIVE_CHAT_TEMPLATE,
)


def get_chat_model(
    api_key: str,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
    streaming: bool = False,
) -> ChatOpenAI:
    """Initialize and return a ChatOpenAI instance."""
    return ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        streaming=streaming,
    )


def build_financial_llm_chain(llm: ChatOpenAI) -> LLMChain:
    """
    Build a reusable LLMChain using the ChatPromptTemplate (Section 10 & 18).
    """
    return LLMChain(
        llm=llm,
        prompt=STRUCTURED_CHAT_PROMPT_TEMPLATE,
    )


def run_structured_analysis(llm: ChatOpenAI, inputs: Dict[str, Any]) -> str:
    """Run the reusable LLMChain and return the raw JSON string output."""
    chain = build_financial_llm_chain(llm)
    response = chain.invoke(inputs)
    if isinstance(response, dict):
        return response.get("text", "")
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)


def stream_recommendations(
    llm: ChatOpenAI, inputs: Dict[str, Any]
) -> Generator[str, None, None]:
    """
    Section 13 Streaming Requirement:
    Format messages with NARRATIVE_CHAT_TEMPLATE and stream chunks via llm.stream().
    """
    messages = NARRATIVE_CHAT_TEMPLATE.format_messages(**inputs)
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


def build_message_demo_conversation(
    inputs: Dict[str, Any], ai_summary: str
) -> List[BaseMessage]:
    """
    Section 10 & 18 Requirement:
    Demonstrate explicit SystemMessage, HumanMessage, and AIMessage objects
    representing a structured LangChain conversation turn.
    """
    sys_msg = SystemMessage(content=SYSTEM_SAFETY_PROMPT.strip())
    human_msg = HumanMessage(
        content=(
            f"My monthly income is {inputs['monthly_income']}, total expenses are "
            f"{inputs['total_expenses']} ({inputs['expense_ratio']}%), and I save "
            f"{inputs['savings']} ({inputs['savings_ratio']}%). My goal is '{inputs['financial_goal']}'."
        )
    )
    ai_msg = AIMessage(
        content=ai_summary
        or "Educational Summary: Focus on balancing monthly cash flow and building an emergency buffer."
    )
    return [sys_msg, human_msg, ai_msg]


def run_followup_chat(
    llm: ChatOpenAI,
    financial_context: str,
    chat_history: List[Dict[str, str]],
    user_question: str,
) -> Generator[str, None, None]:
    """
    Bonus Feature: Multi-turn follow-up Q&A using SystemMessage, HumanMessage, and AIMessage.
    """
    messages: List[BaseMessage] = [
        SystemMessage(
            content=(
                SYSTEM_SAFETY_PROMPT
                + f"\n\nUser's Current Financial Context:\n{financial_context}\n"
                "Answer the user's question concisely and educationally based on their numbers."
            )
        )
    ]
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_question))

    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content