# src/agents/react_agent.py
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from src.llm_factory import get_llm
import os

class ReactAgent:
    def __init__(self, model_name: str, provider: str):
        print(f"🛠️ Initializing Agent B (The Operator) with model: {model_name} [{provider}]")
        self.llm = get_llm(provider, model_name)
        
        # --- PATH SETUP ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../../"))
        
        self.accounts_path = os.path.join(project_root, "data/crm_mock/accounts.csv")
        self.contacts_path = os.path.join(project_root, "data/crm_mock/contacts.csv")
        self.deals_path = os.path.join(project_root, "data/crm_mock/deals.csv")

        # Load Dataframes
        self.df_accounts = pd.read_csv(self.accounts_path)
        self.df_contacts = pd.read_csv(self.contacts_path)
        self.df_deals = pd.read_csv(self.deals_path)

    def run(self, task_input: str) -> str:
        print(f"🤖 Agent B is starting tool loop for: {task_input[:50]}...")
        
        # --- ROBUSTNESS FIX ---
        # Llama 3 sometimes forgets to stop. We give it a specific 'prefix' instruction.
        custom_prefix = (
            "You are a pandas expert. "
            "ALWAYS run `import pandas as pd` before writing any code. "
            "You must use the provided dataframes to answer the question. "
            "IMPORTANT: When you have the answer, you MUST start your response with 'Final Answer:'. "
            "Do not explain your steps after finding the answer. Just say 'Final Answer: <the answer>'."
        )

        agent = create_pandas_dataframe_agent(
            self.llm,
            [self.df_accounts, self.df_contacts, self.df_deals],
            verbose=True, 
            allow_dangerous_code=True,
            # We treat parsing errors as a 'retry' signal, not a crash
            agent_executor_kwargs={"handle_parsing_errors": True},
            prefix=custom_prefix 
        )
        
        try:
            full_prompt = (
                f"You have access to 3 tables: Accounts, Contacts, and Deals. "
                f"Use them to answer this request: {task_input}"
            )
            response = agent.invoke(full_prompt)
            return response['output']
        except Exception as e:
            return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    agent = ReactAgent(model_name="llama3.1:latest", provider="ollama")
    print(agent.run("Find the email of the CTO of CyberDyne."))