# src/agents/react_agent.py
import pandas as pd
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from src.llm_factory import get_llm
from src.tools.file_tools import read_document, list_files

class ReactAgent:
    def __init__(self, model_name: str, provider: str):
        print(f"🛠️ Initializing Agent B ({provider}) with model: {model_name}")
        self.llm = get_llm(provider, model_name)
        
        # Path Setup
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../../"))
        
        # Load CSVs strictly
        # We fail fast if data is missing rather than letting the agent guess/hallucinate
        try:
            self.df_accounts = pd.read_csv(os.path.join(project_root, "data/crm_mock/accounts.csv"))
            self.df_contacts = pd.read_csv(os.path.join(project_root, "data/crm_mock/contacts.csv"))
            self.df_deals = pd.read_csv(os.path.join(project_root, "data/crm_mock/deals.csv"))
        except FileNotFoundError as e:
            print(f"❌ DATA ERROR: {e}")
            raise

    def run(self, task_input: str) -> str:
        # We enforce a "Stateless" mindset: Assume variables might be lost, so import everything.
        # This solves the `NameError: name 'pd' is not defined`
        prefix_prompt = (
            "You are a Data Logic Agent. You have access to pandas dataframes and file tools.\n"
            "DATA MAP:\n"
            "- `df1`: Accounts (account_id, company_name...)\n"
            "- `df2`: Contacts (contact_id, name, email...)\n"
            "- `df3`: Deals (deal_id, amount, stage...)\n\n"
            "CRITICAL EXECUTION RULES:\n"
            "1. ⚠️ ALWAYS start your python code with `import pandas as pd`.\n"
            "2. NEVER assume `pd` is already defined. Import it every time.\n"
            "3. If you need to read a file, RUN `list_files()` FIRST to see what exists.\n"
            "4. Do not guess filenames. Check them.\n"
        )
        
        agent = create_pandas_dataframe_agent(
            self.llm,
            [self.df_accounts, self.df_contacts, self.df_deals],
            extra_tools=[read_document, list_files],
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            prefix=prefix_prompt
        )
        
        try:
            return agent.invoke(task_input)['output']
        except Exception as e:
            return f"Agent Logic Failed: {str(e)}"