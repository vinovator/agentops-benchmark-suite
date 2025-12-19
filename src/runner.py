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
    def __init__(self):
        print("🚀 Initializing Benchmark Suite...")
        
        # 1. Load Config
        config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError("❌ config.yaml not found!")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # 2. Initialize Judge (The Teacher)
        print("   ⚖️  Initializing LLM Judge...")
        # You can hardcode a smart model here or add a 'judge' section to config
        self.judge_llm = get_llm(provider="google", model_name="gemini-1.5-flash")

        # 3. Initialize Agents (The Students)
        print("   🤖 Initializing Agents...")
        self.agents = {
            "Agent A": BaseAgent(self.config['agents']['agent_a']['model'], self.config['agents']['agent_a']['provider']),
            "Agent B": ReactAgent(self.config['agents']['agent_b']['model'], self.config['agents']['agent_b']['provider']),
            "Agent C": PlannerAgent(self.config['agents']['agent_c']['model'], self.config['agents']['agent_c']['provider'])
        }
        
        # 4. Load All Task Files
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
                    if isinstance(loaded, list):
                        self.tasks.extend(loaded)
                    elif isinstance(loaded, dict) and "tasks" in loaded:
                        self.tasks.extend(loaded["tasks"])
                    # ---------------------------------------------------------------
            else:
                print(f"   ⚠️ Warning: Task file not found: {file}")
        
        self.results = []

    def grade_quality(self, task_input, agent_response, rubric):
        """Soft Judge: Asks an LLM to rate the response 1-5."""
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
            return int(''.join(filter(str.isdigit, score)))
        except:
            return 3 

    def evaluate_hard_gates(self, response: str, rules: dict) -> dict:
        """Universal Validator: JSON Schema, Regex, and String Matching."""
        gates = rules.get("hard_gates", [])
        score = {"passed": True, "failed_reasons": []}
        
        # Extract JSON
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
            elif gate_type in ["crm_contact_match", "crm_deal_match", "crm_field_check", "crm_numeric_reference_check", "regex_all", "regex_any", "field_equals"]:
                targets = []
                # Collect all possible target strings from the params
                for key in ["expected_account_id", "expected_contact_id", "expected_email", "expected_deal_id", "expected_value", "expected", "patterns"]:
                    val = params.get(key)
                    if isinstance(val, list): targets.extend([str(v) for v in val])
                    elif val: targets.append(str(val))
                
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
                
                # Grading
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
        df = pd.DataFrame(self.results)
        output_path = os.path.join(os.path.dirname(__file__), "../outputs/leaderboard.csv")
        df.to_csv(output_path, index=False)
        print(f"\n📊 Results saved to: {output_path}")
        return df

if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_benchmark()
    runner.save_results()