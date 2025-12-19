import os
from langchain_core.tools import tool

@tool
def read_document(file_name: str) -> str:
    """
    Reads a document from the knowledge_base or transcripts folder.
    Args:
        file_name: The name of the file (e.g., 'security_policy.md', 'meeting_001.txt')
    """
    # Robust path finding
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    
    # Check both folders
    paths_to_check = [
        os.path.join(project_root, "data/knowledge_base", file_name),
        os.path.join(project_root, "data/transcripts", file_name)
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {str(e)}"
    
    return "Error: File not found. Check the file name."
