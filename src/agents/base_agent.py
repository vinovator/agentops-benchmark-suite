# src/agents/base_agent.py
from src.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate

class BaseAgent:
    """
    Agent A: 'The Intern' (Zero-Shot)
    Architecture: Direct LLM call. No tools. No RAG.
    Behavior: Tries to answer from its training data (Hallucination Machine).
    """
    def __init__(self, model_name: str, provider: str):
        print(f"🔌 Initializing Agent A (The Intern) with model: {model_name} [{provider}]")
        # Temperature=0 makes it deterministic (less creative, more strict)
        self.llm = get_llm(provider, model_name)

    def run(self, task_input: str) -> str:
        """
        The 'run' method is the standard interface for all our agents.
        """
        print(f"🤖 Agent A is thinking about: {task_input[:50]}...")
        
        # Simple Prompt Construction
        # We give it a system instruction to be professional, but NO context files.
        messages = [
            ("system", "You are a helpful enterprise assistant. Answer the user's query directly."),
            ("human", task_input),
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"❌ Error: {str(e)}"

# A quick test block to run if executed directly
if __name__ == "__main__":
    agent = BaseAgent(model_name="llama3.2:3b", provider="ollama")
    print(agent.run("Hello, who are you?"))