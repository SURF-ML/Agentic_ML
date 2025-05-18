import os
import subprocess
import requests
from bs4 import BeautifulSoup # For web Browse
import json # For reading potential JSON config files or data
import shutil # For deleting directories
import glob # For pattern matching file names (e.g. finding specific extensions)
from datetime import datetime

from smolagents import tool

# --- File System Tools ---

@tool
def create_directory(path: str) -> str:
    """
    Creates a new directory at the specified path.
    If the directory already exists, it does nothing.
    Intermediate parent directories will also be created if they don't exist.

    Args:
        path: The path where the directory should be created.

    Returns:
        A message indicating success or failure.
    """
    try:
        os.makedirs(path, exist_ok=True)
        return f"Successfully created directory (or it already existed): {path}"
    except Exception as e:
        return f"Error creating directory {path}: {str(e)}"

@tool
def write_file(filepath: str, content: str, overwrite: bool = False) -> str:
    """
    Writes content to a file. Can create the file if it doesn't exist.
    Parent directories will be created if they don't exist.

    Args:
        filepath: The path to the file.
        content: The string content to write to the file.
        overwrite: If True, overwrites the file if it exists.
                   If False and file exists, an error is returned. Defaults to False.

    Returns:
        A message indicating success or failure.
    """
    try:
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        if not overwrite and os.path.exists(filepath):
            return f"Error: File {filepath} already exists and overwrite is False."
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to file: {filepath}"
    except Exception as e:
        return f"Error writing to file {filepath}: {str(e)}"

@tool
def append_to_file(filepath: str, content: str) -> str:
    """
    Appends content to an existing file. Creates the file if it doesn't exist.
    Parent directories will be created if they don't exist.

    Args:
        filepath: The path to the file.
        content: The string content to append to the file.

    Returns:
        A message indicating success or failure.
    """
    try:
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully appended to file: {filepath}"
    except Exception as e:
        return f"Error appending to file {filepath}: {str(e)}"

@tool
def read_file_content(filepath: str) -> str:
    """
    Reads the content of a specified file.

    Args:
        filepath: The path to the file.

    Returns:
        The content of the file as a string, or an error message if reading fails.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"Error: File not found at {filepath}."
    except Exception as e:
        return f"Error reading file {filepath}: {str(e)}"

@tool
def list_directory_contents(path: str = ".", show_hidden: bool = False, recursive: bool = False, max_depth: int = 2) -> str:
    """
    Lists the contents (files and subdirectories) of a specified directory.

    Args:
        path: The path to the directory. Defaults to the current directory (".").
        show_hidden: If True, includes hidden files/directories (starting with '.'). Defaults to False.
        recursive: If True, lists contents of subdirectories recursively up to max_depth. Defaults to False.
        max_depth: The maximum depth for recursive listing. Only used if recursive is True. Defaults to 2.

    Returns:
        A string listing the directory contents, or an error message.
    """
    try:
        if not os.path.isdir(path):
            return f"Error: {path} is not a valid directory."

        all_contents = []
        
        def _list_recursive(current_path, current_depth):
            if current_depth > max_depth:
                return

            entries = os.listdir(current_path)
            for entry in entries:
                if not show_hidden and entry.startswith('.'):
                    continue
                
                full_entry_path = os.path.join(current_path, entry)
                relative_entry_path = os.path.relpath(full_entry_path, start=path)
                
                if os.path.isdir(full_entry_path):
                    all_contents.append(f"DIR:  {relative_entry_path}/")
                    if recursive:
                        _list_recursive(full_entry_path, current_depth + 1)
                else:
                    all_contents.append(f"FILE: {relative_entry_path}")

        _list_recursive(path, 0)
        
        if not all_contents:
            return f"Directory {path} is empty or contains only hidden files (if show_hidden=False)."
        return f"Contents of {path}:\n" + "\n".join(all_contents)
    except Exception as e:
        return f"Error listing directory {path}: {str(e)}"


@tool
def delete_file_or_directory(path: str) -> str:
    """
    Deletes a file or a directory (recursively).
    USE WITH CAUTION. This operation is irreversible.

    Args:
        path: The path to the file or directory to delete.

    Returns:
        A message indicating success or failure.
    """
    try:
        if not os.path.exists(path):
            return f"Error: Path {path} does not exist."
        if os.path.isfile(path):
            os.remove(path)
            return f"Successfully deleted file: {path}"
        elif os.path.isdir(path):
            shutil.rmtree(path)
            return f"Successfully deleted directory (recursively): {path}"
        else:
            return f"Error: {path} is not a file or directory."
    except Exception as e:
        return f"Error deleting {path}: {str(e)}"

@tool
def replace_text_in_file(filepath: str, old_text: str, new_text: str, count: int = 0) -> str:
    """
    Replaces occurrences of old_text with new_text in a file.

    Args:
        filepath: The path to the file.
        old_text: The text to be replaced.
        new_text: The text to replace with.
        count: Maximum number of occurrences to replace. If 0, replaces all. Defaults to 0.

    Returns:
        A message indicating success (including number of replacements) or failure.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        if count == 0:
            modified_content, num_replacements = content.replace(old_text, new_text), content.count(old_text)
        else:
            modified_content, num_replacements = content.replace(old_text, new_text, count), content.count(old_text) # Approximate for this case
            # For exact count with limit, a more complex approach would be needed if only `count` replacements are made.
            # Python's replace behavior for `count` is what we rely on here.

        if num_replacements == 0 and old_text not in content: # Check if old_text was actually there
             return f"Text '{old_text}' not found in {filepath}. No changes made."

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(modified_content)
        
        return f"Successfully replaced {num_replacements} occurrence(s) of '{old_text}' with '{new_text}' in {filepath}."
    except FileNotFoundError:
        return f"Error: File not found at {filepath}."
    except Exception as e:
        return f"Error replacing text in file {filepath}: {str(e)}"

# --- Execution & Environment Tools ---

@tool
def execute_python_script(script_path: str, args: list[str] | None = None) -> str:
    """
    Executes a Python script with specified arguments.
    Ensure the script is safe to run.

    Args:
        script_path: The path to the Python script.
        args: A list of string arguments to pass to the script. Defaults to None.

    Returns:
        The standard output and standard error from the script execution, or an error message.
    """
    if not os.path.exists(script_path):
        return f"Error: Python script not found at {script_path}."
    if not script_path.endswith(".py"):
        return f"Error: {script_path} is not a Python script (must end with .py)."

    command = ["python", script_path]
    if args:
        command.extend(args)

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60) # Added timeout
        output = f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
        if process.returncode != 0:
            output += f"\nScript exited with error code: {process.returncode}"
        return output
    except subprocess.TimeoutExpired:
        return f"Error executing Python script {script_path}: Timeout after 60 seconds."
    except Exception as e:
        return f"Error executing Python script {script_path}: {str(e)}"

@tool
def execute_shell_command(command: str) -> str:
    """
    Executes a shell command.
    WARNING: This tool is powerful and can be dangerous if misused.
    Ensure commands are safe and come from a trusted source or are validated.
    Only simple commands are recommended. For complex operations, use dedicated tools or Python functions.

    Args:
        command: The shell command to execute.

    Returns:
        The standard output and standard error from the command, or an error message.
    """
    # Basic safety: disallow some obviously dangerous patterns if needed, though a comprehensive blacklist is hard.
    # For a real system, a much more robust sandboxing or allowlist approach is necessary.
    forbidden_patterns = ["rm -rf /", "mkfs", "> /dev/sda"] # Very simplistic examples
    for pattern in forbidden_patterns:
        if pattern in command:
            return f"Error: Command '{command}' contains a potentially dangerous pattern and was blocked."
    try:
        # Using shell=True is a security hazard. If possible, split command into a list.
        # For LLM-generated commands, shell=True might be hard to avoid initially.
        # Consider a timeout.
        process = subprocess.run(command, shell=True, capture_output=True, text=True, check=False, timeout=60)
        output = f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
        if process.returncode != 0:
            output += f"\nCommand exited with error code: {process.returncode}"
        return output
    except subprocess.TimeoutExpired:
        return f"Error executing command '{command}': Timeout after 60 seconds."
    except Exception as e:
        return f"Error executing command '{command}': {str(e)}"

@tool
def install_python_package(package_name: str) -> str:
    """
    Installs a Python package using pip.
    WARNING: Ensure the package name is valid and comes from a trusted source.
    This modifies the Python environment.

    Args:
        package_name: The name of the package to install (e.g., "numpy", "requests==2.25.1").

    Returns:
        A message indicating success or failure of the installation.
    """
    try:
        command = ["python", "-m", "pip", "install", package_name]
        process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=300) # 5 min timeout
        return f"Successfully installed package {package_name}.\nOutput:\n{process.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Error installing package {package_name}. Return code: {e.returncode}\nOutput:\n{e.stdout}\nError:\n{e.stderr}"
    except subprocess.TimeoutExpired:
        return f"Error installing package {package_name}: Timeout after 300 seconds."
    except Exception as e:
        return f"Error installing package {package_name}: {str(e)}"

# --- Web Interaction Tools ---

@tool
def search_wikipedia(query: str) -> str:
    """
    Fetches a summary of a Wikipedia page for a given query.
    Args:
        query: The search term to look up on Wikipedia.
    Returns:
        str: A summary of the Wikipedia page if successful, or an error message if the request fails.
    Raises:
        requests.exceptions.RequestException: If there is an issue with the HTTP request.
    """
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        title = data["title"]
        extract = data["extract"]

        return f"Summary for {title}: {extract}"

    except requests.exceptions.RequestException as e:
        return f"Error fetching Wikipedia data: {str(e)}"

@tool
def browse_webpage(url: str) -> str:
    """
    Fetches and returns the textual content of a webpage.
    It tries to extract the main content and cleans it up.

    Args:
        url: The URL of the webpage to browse.

    Returns:
        The cleaned textual content of the webpage, or an error message.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status() # Raise an exception for HTTP errors

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text
        text = soup.get_text()

        # Break into lines and remove leading/trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        if not text:
            return f"Could not extract meaningful text from {url}. The page might be JavaScript-heavy or empty."

        return text[:10000] # Limit content length to avoid overwhelming the LLM
    except requests.exceptions.RequestException as e:
        return f"Error fetching webpage {url}: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred while Browse {url}: {str(e)}"

@tool
def search_arxiv(query: str, max_results: int = 3) -> str:
    """
    Searches arXiv for papers matching the query.
    Requires the 'arxiv' package to be installed (`pip install arxiv`).

    Args:
        query: The search query (e.g., "machine learning interpretability").
        max_results: The maximum number of results to return. Defaults to 3.

    Returns:
        A formatted string with search results (title, authors, summary, link), or an error message.
    """
    try:
        import arxiv # type: ignore
    except ImportError:
        return "Error: The 'arxiv' library is not installed. Please install it using 'pip install arxiv'."

    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = []
        for r in search.results():
            authors = ", ".join(author.name for author in r.authors)
            results.append(
                f"Title: {r.title}\n"
                f"Authors: {authors}\n"
                f"Published: {r.published.strftime('%Y-%m-%d')}\n"
                f"Summary: {r.summary[:500]}...\n" # Truncate summary
                f"Link: {r.entry_id}\n"
                f"PDF: {r.pdf_url}"
            )
        if not results:
            return f"No results found on arXiv for query: {query}"
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Error searching arXiv for '{query}': {str(e)}"

@tool
def search_github_repositories(query: str, max_results: int = 3, language: str | None = None, sort: str = "best-match") -> str:
    """
    Searches GitHub for repositories matching the query.
    Uses the GitHub search API (publicly accessible for basic queries).

    Args:
        query: The search query (e.g., "pytorch cnn example").
        max_results: The maximum number of results to return. Defaults to 3.
        language: Filter by programming language (e.g., "python"). Defaults to None.
        sort: Sort criteria. Options: 'stars', 'forks', 'help-wanted-issues', 'updated', 'best-match'. Defaults to 'best-match'.


    Returns:
        A formatted string with search results (name, description, URL, stars), or an error message.
    """
    try:
        base_url = "https://api.github.com/search/repositories"
        params = {
            "q": query,
            "sort": sort,
            "order": "desc",
            "per_page": max_results
        }
        if language:
            params["q"] += f" language:{language}"

        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SmolAgent-GitHubSearchTool/1.0"
        }
        response = requests.get(base_url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        if not data["items"]:
            return f"No repositories found on GitHub for query: {query}"

        results = []
        for item in data["items"]:
            results.append(
                f"Name: {item['full_name']}\n"
                f"Description: {item.get('description', 'N/A')[:200]}...\n"
                f"URL: {item['html_url']}\n"
                f"Stars: {item['stargazers_count']}\n"
                f"Language: {item.get('language', 'N/A')}\n"
                f"Last Updated: {item.get('updated_at')}"
            )
        return "\n\n---\n\n".join(results)
    except requests.exceptions.RequestException as e:
        return f"Error searching GitHub: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred while searching GitHub: {str(e)}"

# --- Agent Specific/Memory Tools ---

# Define the primary scratchpad path
SCRATCHPAD_PATH = "./agent_environment/agent/files/scratchpad.txt"

@tool
def read_scratchpad(scratchpad_file: str = SCRATCHPAD_PATH) -> str:
    """
    Reads the content of the agent's scratchpad file.

    Args:
        scratchpad_file: The path to the scratchpad file. Defaults to "agent_scratchpad.txt".

    Returns:
        The content of the scratchpad file, or a message if it's empty or not found.
    """
    try:
        with open(scratchpad_file, "r", encoding="utf-8") as f:
            content = f.read()
        if not content:
            return "Scratchpad is empty."
        return content
    except FileNotFoundError:
        return f"Scratchpad file '{scratchpad_file}' not found. You can create it using update_scratchpad."
    except Exception as e:
        return f"Error reading scratchpad file {scratchpad_file}: {str(e)}"

@tool
def update_scratchpad(content: str, scratchpad_file: str = SCRATCHPAD_PATH, mode: str = "a") -> str:
    """
    Updates the agent's scratchpad file by appending or overwriting content.
    Creates the scratchpad file if it doesn't exist.

    Args:
        content: The content to write to the scratchpad.
        scratchpad_file: The path to the scratchpad file. Defaults to "agent_scratchpad.txt".
        mode: "a" to append, "w" to overwrite. Defaults to "a" (append).

    Returns:
        A message indicating success or failure.
    """
    if mode not in ["a", "w"]:
        return "Error: Invalid mode. Use 'a' for append or 'w' for overwrite."
    try:
        # Ensure directory exists if scratchpad_file includes a path
        parent_dir = os.path.dirname(scratchpad_file)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(scratchpad_file, mode, encoding="utf-8") as f:
            f.write(content + ("\n" if mode == "a" and content and not content.endswith("\n") else "")) # Add newline if appending
        action = "appended to" if mode == "a" else "overwritten"
        return f"Successfully {action} scratchpad file: {scratchpad_file}"
    except Exception as e:
        return f"Error updating scratchpad file {scratchpad_file}: {str(e)}"

@tool
def inspect_file_type_and_structure(filepath: str, head_lines: int = 10) -> str:
    """
    Inspects a file to determine its likely type based on extension and optionally shows its first few lines.
    For JSON or text-based configuration files (like .yaml, .ini, .cfg, .toml), it tries to parse and show structure.

    Args:
        filepath: The path to the file.
        head_lines: Number of initial lines to show for text files. Defaults to 10.

    Returns:
        A string describing the file type, its structure (for known types), and/or its first few lines.
    """
    if not os.path.exists(filepath):
        return f"Error: File not found at {filepath}."
    if not os.path.isfile(filepath):
        return f"Error: {filepath} is not a file."

    _, extension = os.path.splitext(filepath)
    extension = extension.lower()
    
    file_info = [f"File: {filepath}", f"Size: {os.path.getsize(filepath)} bytes", f"Extension: {extension}"]

    try:
        if extension == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    structure = json.dumps(data, indent=2, ensure_ascii=False)
                    file_info.append("Type: JSON file")
                    file_info.append(f"Structure (or content if small):\n{structure[:2000]}{'...' if len(structure) > 2000 else ''}") # Limit output
                except json.JSONDecodeError as je:
                    file_info.append("Type: Likely JSON file (based on extension), but failed to parse.")
                    file_info.append(f"Parsing Error: {str(je)}")
                    with open(filepath, "r", encoding="utf-8") as f_text: # Try reading as text
                        content = f_text.read()
                    file_info.append(f"First {head_lines} lines:\n" + "\n".join(content.splitlines()[:head_lines]))

        elif extension in [".yaml", ".yml", ".ini", ".cfg", ".toml", ".txt", ".md", ".py", ".sh", ".r", ".sql", ".csv", ".log", ".html", ".css", ".js", ".xml"]:
            file_info.append(f"Type: Text-based file (extension: {extension})")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            file_info.append(f"Total lines: {len(lines)}")
            file_info.append(f"First {min(head_lines, len(lines))} lines:\n" + "".join(lines[:head_lines]))
            if len(lines) > head_lines:
                file_info.append("...")
        else:
            file_info.append(f"Type: Binary or unknown file type (extension: {extension}). Preview not available.")

        return "\n".join(file_info)

    except Exception as e:
        return f"Error inspecting file {filepath}: {str(e)}"

# --- Agent Logging Tools ---
LOG_FILES_DIR = "./agent_environment/agent/agent_log_files/"

@tool
def log_agent_message(filename: str, message: str, timestamp: bool = True) -> str:
    """
    Logs a custom message from the agent to a specified file in the agent's log directory.
    The agent can use this to record its thoughts, intermediate decisions, observations,
    or any data it wants to persist for later review or use.

    The log directory is fixed at './agent_environment/agent/agent_log_files/'.
    The agent should provide a descriptive filename. Common extensions like .txt or .log are recommended.

    Args:
        filename: The name for the log file (e.g., "my_thoughts_on_step1.txt", "api_response.log").
                  This will be created under './agent_environment/agent/agent_log_files/'.
                  The agent should choose a name that helps it retrieve the log later if needed.
        message: The string message or data to log.
        timestamp: If True (default), a timestamp will be prepended to the message.

    Returns:
        A confirmation message indicating success or failure, including the full path to the log file.
    """
    try:
        # Ensure the base log directory exists
        os.makedirs(LOG_FILES_DIR, exist_ok=True)

        # Sanitize filename to prevent directory traversal or invalid characters (basic)
        safe_filename = "".join(c for c in filename if c.isalnum() or c in ('.', '_', '-')).strip()
        if not safe_filename:
            return "Error: Provided filename is invalid or results in an empty name after sanitization."

        filepath = os.path.join(LOG_FILES_DIR, safe_filename)

        log_entry = message
        if timestamp:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{now}] {message}"

        # Use append mode to avoid accidental overwrites if the agent uses the same filename,
        # but add a newline to separate entries.
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

        return f"Successfully logged message to: {filepath}"
    except Exception as e:
        return f"Error logging agent message to {filename} in {LOG_FILES_DIR}: {str(e)}"

@tool
def list_agent_log_files(path_to_list: str | None = None) -> str:
    """
    Lists files present in the agent's log directory or a specified subdirectory within it.
    The default log directory is './agent_environment/agent/agent_log_files/'.

    Args:
        path_to_list: Optional. A specific path relative to the script's execution directory
                      to list files from. If None, defaults to the agent's primary log directory.
                      The agent can use this to list files in subfolders it might have created
                      within its log space, or list the main log directory itself.

    Returns:
        A string listing the files and directories found, or an error message.
    """
    target_path = path_to_list if path_to_list is not None else LOG_FILES_DIR

    try:
        if not os.path.exists(target_path):
            return f"Error: The path '{target_path}' does not exist."
        if not os.path.isdir(target_path):
            return f"Error: The path '{target_path}' is not a directory."

        entries = os.listdir(target_path)
        if not entries:
            return f"The directory '{target_path}' is empty."

        # Differentiate files and directories for clarity
        files = [f for f in entries if os.path.isfile(os.path.join(target_path, f))]
        directories = [d for d in entries if os.path.isdir(os.path.join(target_path, d))]

        output_lines = [f"Contents of '{target_path}':"]
        if directories:
            output_lines.append("Directories:")
            for d in sorted(directories):
                output_lines.append(f"  {d}/")
        if files:
            output_lines.append("Files:")
            for f in sorted(files):
                output_lines.append(f"  {f}")

        return "\n".join(output_lines)
    except Exception as e:
        return f"Error listing contents of '{target_path}': {str(e)}"
    

