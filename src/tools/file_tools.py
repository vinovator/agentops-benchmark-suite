# src/tools/file_tools.py
import os
from langchain_core.tools import tool

@tool
def list_files() -> str:
    """
    Lists all available files in the 'knowledge_base' and 'transcripts' directories.
    Use this tool BEFORE reading a file to ensure you have the correct filename.
    """
    # Dynamic path resolution to ensure we find the data directory regardless of where this is run from
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    
    # Define the two key directories we allow agents to access
    paths = {
        "KB": os.path.join(project_root, "data/knowledge_base"),
        "TRANSCRIPTS": os.path.join(project_root, "data/transcripts")
    }
    
    found_files = []
    for category, path in paths.items():
        if os.path.exists(path):
            # Exclude hidden files
            files = [f for f in os.listdir(path) if not f.startswith('.')]
            for f in files:
                found_files.append(f"[{category}] {f}")
    
    return "\n".join(found_files) if found_files else "No files found."

@tool
def read_document(file_name: str) -> str:
    """
    Reads the full content of a file. 
    Args: file_name (str) - The exact name of the file (e.g., 'product_whitepaper.md').
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    
    # Check both folders for the requested file
    search_paths = [
        os.path.join(project_root, "data/knowledge_base", file_name),
        os.path.join(project_root, "data/transcripts", file_name)
    ]
    
    for p in search_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {str(e)}"
                
    # Helpful error message guiding the agent to use the discovery tool
    return f"Error: File '{file_name}' not found. Did you use list_files to check the name?"
