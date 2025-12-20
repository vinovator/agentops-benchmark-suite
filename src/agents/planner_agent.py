# src/agents/planner_agent.py
import pandas as pd
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langgraph.graph import StateGraph, END
from src.llm_factory import get_llm
from src.tools.file_tools import read_document, list_files

class PlannerAgent:
    def __init__(self, model_name: str, provider: str):
        print(f"♟️ Initializing Agent C ({provider}) with model: {model_name}")
        self.llm = get_llm(provider, model_name)
        
        # Path Setup
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../../"))
        
        # Load Dataframes for schema context and execution
        self.df_accounts = pd.read_csv(os.path.join(project_root, "data/crm_mock/accounts.csv"))
        self.df_contacts = pd.read_csv(os.path.join(project_root, "data/crm_mock/contacts.csv"))
        self.df_deals = pd.read_csv(os.path.join(project_root, "data/crm_mock/deals.csv"))
        
        # Context summary for the planning phase
        self.schema_summary = (
            "Dataframes available via python: df1 (Accounts), df2 (Contacts), df3 (Deals).\n"
            "Files available via tools: Use list_files() to see them."
        )
        
        self.app = self.build_graph()

    def plan_step(self, state):
        # The Planner: Decides what to do
        planner_prompt = (
            f"Task: {state['task']}\n"
            f"Context: {state['data_context']}\n"
            "Generate a numbered step-by-step plan. "
            "If the task involves reading a document, Step 1 MUST be 'List files to verify name'."
        )
        response = self.llm.invoke(planner_prompt)
        return {"plan": response.content}

    def execute_step(self, state):
        # The Executor: Uses Tools to do it
        # Note: We pass the tools directly here
        executor_agent = create_pandas_dataframe_agent(
            self.llm,
            [self.df_accounts, self.df_contacts, self.df_deals],
            extra_tools=[read_document, list_files], 
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )
        
        # Strict execution prompt to avoid hallucinated filenames or missing imports
        execution_prompt = (
            "You are the Executor. Follow this plan STRICTLY.\n"
            f"THE PLAN:\n{state['plan']}\n\n"
            "RULES:\n"
            "1. If the plan says read a file, use `list_files` first to find the exact name.\n"
            "2. ALWAYS `import pandas as pd` in python blocks.\n"
            "3. Output the Final Answer as JSON if requested."
        )
        
        try:
            response = executor_agent.invoke(execution_prompt)
            return {"response": response['output']}
        except Exception as e:
            return {"response": f"Execution Failed: {str(e)}"}

    def build_graph(self):
        # Build the LangGraph workflow: Planner -> Executor -> End
        workflow = StateGraph(dict)
        workflow.add_node("planner", self.plan_step)
        workflow.add_node("executor", self.execute_step)
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", END)
        return workflow.compile()

    def run(self, task_input):
        inputs = {"task": task_input, "data_context": self.schema_summary}
        result = self.app.invoke(inputs)
        return result["response"]