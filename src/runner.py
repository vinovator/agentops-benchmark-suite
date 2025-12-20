# src/runner.py (FINAL MASTER VERSION)
import yaml
import json
import pandas as pd
import os
import time
import re
from src.agents.base_agent import BaseAgent
from src.agents.react_agent import ReactAgent
from src.agents.planner_agent import PlannerAgent
from src.llm_factory import get_llm

class BenchmarkRunner:
    """
    The Orchestrator of the AgentOps Benchmark Suite.
    
    This class is responsible for:
    1.  **Setup**: Loading configuration and initializing the 'Judge' (LLM evaluator) and 'Agents' (Participants).
    2.  **Task Management**: Loading and parsing tasks from YAML files.
    3.  **Execution Loop**: Running each agent against every task.
    4.  **Evaluation**: 
        -   **Hard Gates**: Objective pass/fail checks (regex, JSON schema, required terms).
        -   **Soft Grading**: Subjective quality assessment (1-5 score) by the LLM Judge.
    5.  **Reporting**: Aggregating results into a Pandas DataFrame and saving to CSV.
    """
    def __init__(self):
        print("🚀 Initializing Benchmark Suite...")
        
        # ---------------------------------------------------------------------
        # 1. Load Config
        # ---------------------------------------------------------------------
        # Reads the central `config.yaml` to determine model providers and API keys.
        config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError("❌ config.yaml not found!")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # ---------------------------------------------------------------------
        # 2. Initialize Judge (The Teacher)
        # ---------------------------------------------------------------------
        # The 'Judge' is an LLM used for subjective evaluation (Soft Scores).
        # We default to a high-reasoning model (e.g., Gemini-1.5/2.5 Pro or similar) for best results.
        print("   ⚖️  Initializing LLM Judge...")
        judge_conf = self.config.get('agents', {}).get('judge', {}) if 'judge' in self.config.get('agents', {}) else self.config.get('judge', {})
        # Fallback if not configured
        provider = judge_conf.get('provider', 'google')
        model = judge_conf.get('model', 'gemini-2.5-flash')
        
        self.judge_llm = get_llm(provider=provider, model_name=model)

        # ---------------------------------------------------------------------
        # 3. Initialize Agents (The Students)
        # ---------------------------------------------------------------------
        # These are the subjects being tested. 
        # - Agent A: Base/Simple Agent
        # - Agent B: ReAct/Tool-using Agent
        # - Agent C: Planning Agent
        print("   🤖 Initializing Agents...")
        self.agents = {
            "Agent A": BaseAgent(self.config['agents']['agent_a']['model'], self.config['agents']['agent_a']['provider']),
            "Agent B": ReactAgent(self.config['agents']['agent_b']['model'], self.config['agents']['agent_b']['provider']),
            "Agent C": PlannerAgent(self.config['agents']['agent_c']['model'], self.config['agents']['agent_c']['provider'])
        }
        
        # ---------------------------------------------------------------------
        # 4. Load All Task Files
        # ---------------------------------------------------------------------
        # Tasks are defined in YAML files in the `tasks/` directory.
        # They cover different domains: Sales, RFPs, Meetings, etc.
        self.tasks = []
        task_files = ["sales_tasks.yaml", "rfp_tasks.yaml", "meeting_tasks.yaml"]
        base_task_dir = os.path.join(os.path.dirname(__file__), "../tasks")
        
        for file in task_files:
            path = os.path.join(base_task_dir, file)
            if os.path.exists(path):
                print(f"   📂 Loading tasks from: {file}")
                with open(path, "r") as f:
                    loaded = yaml.safe_load(f)
                    # --- PATCH: Handle both List and Dict (wrapped) YAML formats ---
                    # Some YAMLs are a direct list `[ - task 1... ]`
                    # Others are a dict `{"tasks": [ - task 1... ]}`
                    # We normalize this to a flat list of tasks.
                    if isinstance(loaded, list):
                        self.tasks.extend(loaded)
                    elif isinstance(loaded, dict) and "tasks" in loaded:
                        self.tasks.extend(loaded["tasks"])
                    # ---------------------------------------------------------------
            else:
                print(f"   ⚠️ Warning: Task file not found: {file}")
        
        self.results = []

    def grade_quality(self, task_input, agent_response, rubric):
        """
        Soft Judge: Subjective Evaluation.
        
        Uses the Judge LLM to evaluate the quality of the response based on a rubric.
        Returns an integer score from 1 to 5.
        
        Args:
            task_input (str): The original prompt given to the agent.
            agent_response (str): The output produced by the agent.
            rubric (str): Guidelines for what constitutes a good response.
        """
        prompt = (
            "You are a strict Teacher grading an AI Agent's homework.\n"
            f"Original Task: {task_input}\n"
            f"Grading Rubric: {rubric}\n"
            "--------------------------------------------------\n"
            f"Agent Response:\n{agent_response}\n"
            "--------------------------------------------------\n"
            "Grade this response from 1 to 5 based strictly on the rubric.\n"
            "Return ONLY the number (e.g., 4)."
        )
        try:
            score = self.judge_llm.invoke(prompt).content.strip()
            # Robust extraction: look for the first digit in the response
            return int(''.join(filter(str.isdigit, score)))
        except:
            # Fallback to neutral score on API failure
            return 3 

    def evaluate_hard_gates(self, response: str, rules: dict) -> dict:
        """
        Universal Validator: Objective Evaluation (Hard Gates).
        
        Checks the response against a set of deterministic rules defined in the task YAML.
        
        Supports:
        1.  **JSON Schema Validation**: Ensures output is valid JSON and has required keys.
        2.  **Universal Containment**: checks for substrings, IDs, specific values.
        3.  **Forbidden Terms**: Ensures specific words are NOT present.
        4.  **Citation Check**: Verifies if specific source files are referenced.
        
        Args:
            response (str): The agent's raw text response.
            rules (dict): The 'eval_rules' dictionary from the task definition.
            
        Returns:
            dict: {"passed": bool, "failed_reasons": list[str]}
        """
        gates = rules.get("hard_gates", [])
        
        # --- V1 Compatibility Patch ---
        # Converts legacy V1 simple dict format to V2 list-of-dicts format.
        # V1: {"must_contain": [...], "forbidden_terms": [...]}
        # V2: [{"type": "crm_contact_match", ...}, {"type": "forbidden_terms", ...}]
        if isinstance(gates, dict):
            normalized_gates = []
            if "must_contain" in gates:
                normalized_gates.append({
                    "type": "crm_contact_match", # Re-using universal string matcher
                    "name": "legacy_must_contain",
                    "params": {"expected": gates["must_contain"]} 
                })
            if "forbidden_terms" in gates:
                normalized_gates.append({
                    "type": "forbidden_terms",
                    "name": "legacy_forbidden",
                    "params": {"terms": gates["forbidden_terms"]}
                })
            gates = normalized_gates
        # ------------------------------

        score = {"passed": True, "failed_reasons": []}
        
        # --- Helper: Extract JSON from response ---
        # Attempts to parse the whole string as JSON, or find a markdown ```json block.
        try:
            response_json = json.loads(response)
            json_valid = True
        except:
            try:
                match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
                if match:
                    response_json = json.loads(match.group(1))
                    json_valid = True
                else:
                    response_json = {}
                    json_valid = False
            except:
                response_json = {}
                json_valid = False

        # --- Gate Processing Loop ---
        for gate in gates:
            gate_type = gate.get("type")
            gate_name = gate.get("name")
            params = gate.get("params", {})

            # 1. JSON Schema Validation
            if gate_type == "json_schema_validate":
                if not json_valid:
                    score["passed"] = False
                    score["failed_reasons"].append(f"{gate_name}: Output was not valid JSON.")
                    continue
                missing = [f for f in params.get("required_fields", []) if f not in response_json]
                if missing:
                    score["passed"] = False
                    score["failed_reasons"].append(f"{gate_name}: Missing keys {missing}")

            # 2. Universal Containment (IDs, Values, Regex)
            # Checks if specific strings (or regex patterns) exist in the response.
            elif gate_type in ["crm_contact_match", "crm_deal_match", "crm_field_check", "crm_numeric_reference_check", "regex_all", "regex_any", "field_equals"]:
                targets = []
                # Collect all possible target strings from the params
                for key in ["expected_account_id", "expected_contact_id", "expected_email", "expected_deal_id", "expected_value", "expected", "patterns"]:
                    val = params.get(key)
                    if isinstance(val, list): targets.extend([str(v) for v in val])
                    elif val is not None: targets.append(str(val))
                
                # Check if they exist in the response
                for target in targets:
                    if target.lower() not in response.lower():
                        score["passed"] = False
                        score["failed_reasons"].append(f"{gate_name}: Missing '{target}'")

            # 3. Forbidden Terms
            elif gate_type == "forbidden_terms":
                for term in params.get("terms", []):
                    if term.lower() in response.lower():
                        score["passed"] = False
                        score["failed_reasons"].append(f"{gate_name}: Forbidden term '{term}' found")

            # 4. Citation Check
            elif gate_type == "citation_check":
                files = params.get("must_include_any_of", [])
                if not any(f in response for f in files):
                     score["passed"] = False
                     score["failed_reasons"].append(f"{gate_name}: No citation from {files}")

        return score

    def run_benchmark(self):
        """
        Main Execution Loop.
        
        Iterates through:
        1. All loaded Tasks
        2. All initialized Agents
        
        For each combination:
        - Runs the agent with the task prompt.
        - Captures the response and execution time.
        - Catches and logs any exceptions so the benchmark doesn't crash.
        - Runs both Hard Gates (Pass/Fail) and Soft Grading (1-5).
        - Appends results to `self.results`.
        """
        print(f"\n🏆 Starting Benchmark on {len(self.tasks)} Tasks...\n")
        
        for task in self.tasks:
            print(f"🔹 Task: {task['name']}")
            
            for agent_name, agent in self.agents.items():
                print(f"   ▶️ Running {agent_name}...")
                start_time = time.time()
                
                try:
                    response = agent.run(task['input_prompt'])
                    error = False
                except Exception as e:
                    response = str(e)
                    error = True
                
                duration = time.time() - start_time
                
                # --- Evaluation Phase ---
                hard_score = self.evaluate_hard_gates(response, task.get('eval_rules', {}))
                quality_score = self.grade_quality(task['input_prompt'], response, task.get('eval_rules', {}).get('soft_score_rubric', 'Is it helpful?'))
                
                self.results.append({
                    "task_id": task.get('task_id', 'unknown'),
                    "task_name": task['name'],
                    "agent": agent_name,
                    "passed": hard_score["passed"],
                    "quality_score": quality_score,
                    "duration_seconds": round(duration, 2),
                    "fail_reasons": "; ".join(hard_score["failed_reasons"]),
                    "error": error
                })
                print(f"      ✅ Score: {quality_score}/5 | Passed: {hard_score['passed']}")

    def save_results(self):
        """
        Exports the benchmark results to a CSV file and a detailed JSON trace.
        - CSV: `../outputs/leaderboard.csv` (High-level metrics)
        - JSON: `../outputs/run_trace_<timestamp>.json` (Deep debugging)
        """
        # 1. Save CSV (Summary)
        df = pd.DataFrame(self.results)
        output_dir = os.path.join(os.path.dirname(__file__), "../outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, "leaderboard.csv")
        df.to_csv(csv_path, index=False)
        
        # 2. Save JSON Trace (Deep Dive)
        # This saves the full prompt, full response, and detailed fail reasons
        trace_path = os.path.join(output_dir, f"run_trace_{int(time.time())}.json")
        with open(trace_path, "w") as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\n📊 Summary saved to: {csv_path}")
        print(f"🕵️  Detailed Trace saved to: {trace_path}")
        return df

if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_benchmark()
    runner.save_results()