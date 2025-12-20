# src/agents/planner_agent.py
import pandas as pd
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from src.llm_factory import get_llm

from src.tools.file_tools import read_document, list_files

# (Removed Duplicate @tool definition)

class PlannerAgent:
    def __init__(self, model_name: str, provider: str):
        print(f"♟️ Initializing Agent C ({provider}) with model: {model_name}")
        self.llm = get_llm(provider, model_name)
        
        # Path Setup
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../../"))
        self.accounts_path = os.path.join(project_root, "data/crm_mock/accounts.csv")
        self.contacts_path = os.path.join(project_root, "data/crm_mock/contacts.csv")
        self.deals_path = os.path.join(project_root, "data/crm_mock/deals.csv")

        # Load Dataframes
        self.df_accounts = pd.read_csv(self.accounts_path)
        self.df_contacts = pd.read_csv(self.contacts_path)
        self.df_deals = pd.read_csv(self.deals_path)
        
        self.schema_summary = (
            f"1. Accounts Table (Columns: {list(self.df_accounts.columns)})\n"
            f"2. Contacts Table (Columns: {list(self.df_contacts.columns)})\n"
            f"3. Deals Table (Columns: {list(self.df_deals.columns)})\n"
            "4. FILE SYSTEM: You can use the 'read_document' tool to read files like 'security_policy.md'."
        )

        self.app = self.build_graph()

    def plan_step(self, state):
        print("   ---> 🧠 Generating Plan...")
        planner_prompt = (
            "You are a Senior Solutions Architect. "
            "Create a numbered execution plan. "
            "If the user asks about policies, meetings, or documents, include a step to 'Read the file X'. "
            f"\n\nAVAILABLE RESOURCES:\n{state['data_context']}"
        )
        messages = [SystemMessage(content=planner_prompt), HumanMessage(content=state["task"])]
        response = self.llm.invoke(messages)
        return {"plan": response.content}

    def execute_step(self, state):
        print("   ---> ⚙️ Executing Plan using Tools...")
        
        # Create Agent with Pandas + The New File Reading Tool
        executor_agent = create_pandas_dataframe_agent(
            self.llm,
            [self.df_accounts, self.df_contacts, self.df_deals],
            extra_tools=[read_document, list_files], # <--- GIVE IT THE TOOL!
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )
        
        execution_prompt = (
            "Follow this plan STRICTLY. "
            "Use 'read_document' if you need to check a policy or file. "
            "Use python for dataframe lookups. "
            f"\n\nTHE PLAN:\n{state['plan']}"
        )
        
        try:
            response = executor_agent.invoke(execution_prompt)
            return {"response": response['output']}
        except Exception as e:
            return {"response": f"Execution Failed: {str(e)}"}

    def build_graph(self):
        # (Same graph building code as before)
        workflow = StateGraph(dict) # Simply using dict for state to avoid typing issues in this snippet
        workflow.add_node("planner", self.plan_step)
        workflow.add_node("executor", self.execute_step)
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", END)
        return workflow.compile()

    def run(self, task_input):
        print(f"♟️ Agent C is strategizing for: {task_input[:50]}...")
        inputs = {"task": task_input, "data_context": self.schema_summary}
        result = self.app.invoke(inputs)
        return result["response"]

if __name__ == "__main__":
    # Manual Test
    agent = PlannerAgent(model_name="gemini-2.5-flash", provider="google")
    print(agent.run("Check the security policy for discount rules."))