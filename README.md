# AgentOps Benchmarking Suite: Evaluating Agency in Enterprise Contexts

Benchmark for evaluating the performance of autonomous AI agents in simulated enterprise environments. It moves beyond simple QA to test complex, multi-step reasoning, tool use, and adherence to strict business constraints.

## Methodology

### 1. The World
The benchmark operates within a simulated enterprise environment containing:
*   **CRM Database**: A structured SQL-like backend (simulated via CSVs) containing Accounts, Contacts, Deals.
*   **Knowledge Base**: unstructured proprietary documents (Security Policies, Meeting Transcripts).
*   **Tools**: Agents are equipped with Python execution (Pandas), File Readers, and logical planning capabilities.

### 2. Tasks
We define a set of diverse "Traps" designed to expose specific agent weaknesses:
*   **Sales Retrieval**: Requires multi-hop joins (e.g., "Find the status of the deal for the company Bob works for"). *Tests: Logic & Search.*
*   **Compliance (RFP)**: Requires checking vendor answers against a strict `security_policy.md`. *Tests: Context Window & Constraint Adherence.*
*   **Meeting Extraction**: Requires resolving vague entity references (e.g., "Alice") to concrete database records (e.g., "alice@cyberdyne.com"). *Tests: Entity Resolution & hallucination.*

### 3. Evals
We implement a dual-layer supervision system for grading:
*   **Hard Gates (Programmatic)**: deterministic checks for critical failures (e.g., using forbidden terms, missing required data points).
*   **Soft Judge (LLM-based)**: A "Teacher" model (Gemini 1.5 Flash) grades the qualitative aspect of the response on a 1-5 Likert scale, assessing helpfulness and reasoning.

### 4. Traces
The benchmark captures full execution traces, including:
*   **Reasoning Steps**: The Chain-of-Thought (CoT) produced by the agent.
*   **Tool Invocations**: Inputs and outputs of every tool call (e.g., SQL queries, file reads).
*   **Debug Logs**: Detailed internal state changes (e.g., LangGraph state transitions).

### 5. Metrics
*   **Pass Rate**: Percentage of tasks passing all Hard Gates.
*   **Quality Score**: Mean score (1-5) assigned by the Soft Judge.
*   **Latency**: End-to-end execution time per task.

## Agent Strategy Variants

We benchmark three distinct agentic architectures:

*   **Baseline A (Zero-Shot)**: A direct LLM call with no tools or memory. Serves as the control for hallucination rates.
*   **Variant B (ReAct)**: A "Reason + Act" architecture equipped with Pandas tools. Optimized for data retrieval but often struggles with unstructured context.
*   **Variant C (Plan-and-Solve)**: A hierarchical "Strategist" (LangGraph) that first generates a multi-step plan before execution. Designed for maximum robustness in complex, mixed-modal tasks.

## Results & Insights

### Reproducing the Benchmark
To run the full evaluation suite:

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Generate World Data** (CRITICAL):
    Measurements require a populated mock CRM and Knowledge Base.
    Run the notebook `notebooks/01_generate_sandbox_data.ipynb` to create:
    *   `data/crm_mock/` (Accounts, Contacts, Deals CSVs)
    *   `data/knowledge_base/` (Markdown policies, PDF pricing guides)
    *   `data/transcripts/` (Meeting logs)

3.  **Configure Environment**:
    Create a `.env` file with your `GOOGLE_API_KEY`.

4.  **Run Benchmark**:
    ```bash
    python -m src.runner
    ```

### Analysis
Results are automatically compiled into `outputs/leaderboard.csv`. This data can be used to visualize the trade-offs between latency, cost, and accuracy across different agent strategies.

## License
MIT License. See [LICENSE](LICENSE) for details.
