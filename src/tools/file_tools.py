import os
import re
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 1. Define the Input Schema (The Contract)
class ReadDocumentInput(BaseModel):
    file_name: str = Field(
        ..., 
        description="The exact name of the file to read. Do NOT include path, just the name (e.g., 'policy.md')."
    )

class ListFilesInput(BaseModel):
    pass # No input needed

# 2. Define the Tools using the Schema
@tool(args_schema=ListFilesInput)
def list_files() -> str:
    """
    Lists all available files in the 'knowledge_base' and 'transcripts' directories.
    Always run this BEFORE reading a document.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    
    paths = {
        "KB": os.path.join(project_root, "data/knowledge_base"),
        "TRANSCRIPTS": os.path.join(project_root, "data/transcripts")
    }
    
    found_files = []
    for category, path in paths.items():
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if not f.startswith('.')]
            for f in files:
                found_files.append(f"[{category}] {f}")
    
    return "\n".join(found_files) if found_files else "No files found."

@tool(args_schema=ReadDocumentInput)
def read_document(file_name: str) -> str:
    """
    Reads the full content of a file.
    """
    # 1. Sanitize: Strip quotes
    clean_name = re.sub(r"^['\"]|['\"]$", "", file_name.strip())
    
    # 2. Robustness: Strip directory prefixes (The "Agent B Loop Fix")
    # Agents often copy-paste "KB/file.md" from the list_files output. We fix that here.
    clean_name = os.path.basename(clean_name) 

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    
    # Check both folders
    search_paths = [
        os.path.join(project_root, "data/knowledge_base", clean_name),
        os.path.join(project_root, "data/transcripts", clean_name)
    ]
    
    for p in search_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {str(e)}"
                
    return f"Error: File '{clean_name}' not found. Did you use list_files to check the name?"
