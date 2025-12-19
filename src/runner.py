# src/runner.py
import yaml
import pandas as pd
import os
import time
from src.agents.base_agent import BaseAgent
from src.agents.react_agent import ReactAgent
from src.agents.planner_agent import PlannerAgent
from src.llm_factory import get_llm # Use factory to get the Judge

class BenchmarkRunner:
    def __init__(self):
        print("🚀 Initializing Benchmark Suite...")
        
        # Load Config
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "../config.yaml")
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize Judge (The Teacher) - Always use the smartest model available
        self.judge_llm = get_llm(provider="google", model_name="gemini-1.5-flash")

        # Initialize Agents (The Students)
        self.agents = {
            "Agent A": BaseAgent(self.config['agents']['agent_a']['model'], self.config['agents']['agent_a']['provider']),
            "Agent B": ReactAgent(self.config['agents']['agent_b']['model'], self.config['agents']['agent_b']['provider']),
            "Agent C": PlannerAgent(self.config['agents']['agent_c']['model'], self.config['agents']['agent_c']['provider'])
        }
        
        # Load All Task Files
        self.tasks = []
        task_files = ["sales_tasks.yaml", "rfp_tasks.yaml", "meeting_tasks.yaml"]
        
        for file in task_files:
            path = os.path.join(current_dir, f"../tasks/{file}")
            if os.path.exists(path):
                with open(path, "r") as f:
                    self.tasks.extend(yaml.safe_load(f))
        
        self.results = []

    def grade_quality(self, task_input, agent_response, rubric):
        """
        The Soft Judge: Asks an LLM to rate the response 1-5.
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
            # extract number just in case
            return int(''.join(filter(str.isdigit, score)))
        except:
            return 3 # Default to average if grading fails

    def evaluate_hard_gates(self, response: str, rules: dict) -> dict:
        # (Keep your existing hard gate logic here!)
        gates = rules.get("hard_gates", {})
        must_contain = gates.get("must_contain", [])
        forbidden_terms = gates.get("forbidden_terms", [])
        
        score = {"passed": True, "failed_reasons": []}
        
        for term in must_contain:
            if term.lower() not in response.lower():
                score["passed"] = False
                score["failed_reasons"].append(f"Missing: '{term}'")
        
        for term in forbidden_terms:
            if term.lower() in response.lower():
                score["passed"] = False
                score["failed_reasons"].append(f"Forbidden: '{term}'")
                
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
                
                # 1. Hard Grading (Python)
                hard_score = self.evaluate_hard_gates(response, task['eval_rules'])
                
                # 2. Soft Grading (LLM Judge)
                quality_score = self.grade_quality(task['input_prompt'], response, task['eval_rules'].get('soft_rubric', 'Is it helpful?'))
                
                # Log Data
                self.results.append({
                    "task_name": task['name'],
                    "agent": agent_name,
                    "passed": hard_score["passed"],
                    "quality_score_1_to_5": quality_score, # New Metric!
                    "duration_seconds": round(duration, 2),
                    "fail_reasons": "; ".join(hard_score["failed_reasons"]),
                    "error": error
                })
                print(f"      ✅ Score: {quality_score}/5 | Passed: {hard_score['passed']}")

    def save_results(self):
        df = pd.DataFrame(self.results)
        
        # Output directory fix
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, "../outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "leaderboard.csv")
        df.to_csv(output_path, index=False)
        return df

if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_benchmark()
    runner.save_results()