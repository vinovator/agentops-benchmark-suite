import pandas as pd
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.tools import tool
from src.llm_factory import get_llm

# Tool Definition (Same as before)
from src.tools.file_tools import read_document, list_files


class ReactAgent:
    def __init__(self, model_name: str, provider: str):
        print(f"🛠️ Initializing Agent B ({provider}) with model: {model_name}")
        self.llm = get_llm(provider, model_name)
        
        # Load Dataframes
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../../"))
        self.df_accounts = pd.read_csv(os.path.join(project_root, "data/crm_mock/accounts.csv"))
        self.df_contacts = pd.read_csv(os.path.join(project_root, "data/crm_mock/contacts.csv"))
        self.df_deals = pd.read_csv(os.path.join(project_root, "data/crm_mock/deals.csv"))

    def run(self, task_input: str) -> str:
        # --- THE FIX: Explicitly Map Variables in the Prompt ---
        # 1. Variable Mapping: We explicitly tell the agent: "df2 is Contacts". It stops guessing df_contacts.
        # 2. Import Enforcement: We explicitly tell it: "Always import pandas as pd". This fixes the name 'pd' is not defined error if the agent tries complex operations.
        # 3. Behavior Control: We tell it "DO NOT load CSVs manually", which stops the panic loop where it tries to read files from disk that it can't find.
        prefix_prompt = (
            "You are an expert Data Analyst Agent.\n"
            "You have access to 3 pre-loaded DataFrames. DO NOT try to load CSVs manually.\n"
            "USE THESE EXACT VARIABLE NAMES:\n"
            "- `df1`: Accounts Table (Columns: account_id, company_name, ...)\n"
            "- `df2`: Contacts Table (Columns: contact_id, name, email, role, ...)\n"
            "- `df3`: Deals Table (Columns: deal_id, amount, stage, ...)\n\n"
            "IMPORTANT RULES:\n"
            "1. Always `import pandas as pd` inside your code block before using `pd`.\n"
            "2. To join tables, use `pd.merge(df1, df2, on='account_id')`.\n"
            "3. If you need to read a file (like a policy), use the `read_document` tool.\n"
            "4. If you don't know the file name, use `list_files` first.\n"
        )
        
        agent = create_pandas_dataframe_agent(
            self.llm,
            [self.df_accounts, self.df_contacts, self.df_deals],
            extra_tools=[read_document, list_files], # <--- Added list_files to allow discovery
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            prefix=prefix_prompt  # <--- Injecting the cheat sheet here
        )
        
        try:
            return agent.invoke(task_input)['output']
        except Exception as e:
            return f"❌ Error: {str(e)}"