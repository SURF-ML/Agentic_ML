import os
import subprocess
import requests
from bs4 import BeautifulSoup # For web Browse
import json # For reading potential JSON config files or data
import shutil # For deleting directories
import glob # For pattern matching file names (e.g. finding specific extensions)
from datetime import datetime

import zipfile
import re
import fnmatch
from importlib import metadata as importlib_metadata # For package versions


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
    

@tool
def download_file_from_url(url: str, save_directory: str, filename: str | None = None) -> str:
    """
    Downloads a file from a given URL and saves it to a specified directory.

    Args:
        url: The URL of the file to download.
        save_directory: The local directory path where the file should be saved.
        filename: Optional. The desired filename. If None, it tries to infer from the URL
                  or Content-Disposition header.

    Returns:
        A message indicating success (including the full save path) or failure.
    """
    try:
        os.makedirs(save_directory, exist_ok=True)
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status() # Raise an exception for HTTP errors

        if not filename:
            if "content-disposition" in response.headers:
                cd = response.headers['content-disposition']
                fname_match = re.search(r'filename="?([^"]+)"?', cd)
                if fname_match:
                    filename = fname_match.group(1)
            if not filename: # Fallback to URL part
                filename = url.split('/')[-1]
                if not filename or "?" in filename : # if URL ends with / or has query params
                     filename = "downloaded_file" # Default generic name

        save_path = os.path.join(save_directory, filename)
        
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return f"Successfully downloaded file from {url} to {save_path}"
    except requests.exceptions.RequestException as e:
        return f"Error downloading file from {url}: {str(e)}"
    except IOError as e:
        return f"Error saving file to {save_directory}: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
    

@tool
def find_files_by_pattern(directory: str, pattern: str, recursive: bool = True) -> str:
    """
    Finds files within a specified directory (and its subdirectories) that match a given glob pattern.

    Args:
        directory: The directory to start the search from.
        pattern: The glob pattern to match filenames (e.g., "*.txt", "data_*.csv").
        recursive: If True, searches subdirectories recursively. Defaults to True.

    Returns:
        A string listing the found files, or a message if no files are found or an error occurs.
    """
    if not os.path.isdir(directory):
        return f"Error: Directory '{directory}' not found."
    
    try:
        search_path = os.path.join(directory, pattern)
        if recursive:
            # For recursive search with glob, the pattern might need to include '**/'
            # e.g. if pattern is '*.txt', use os.path.join(directory, '**', pattern)
            # However, glob itself handles the recursive flag well if pattern doesn't specify dir components.
            # Let's ensure the pattern is relative for recursion within the directory.
            if os.path.isabs(pattern):
                return "Error: For recursive search, pattern should be relative (e.g., '*.txt' or 'subfolder/*.log')."

            # If pattern is simple like '*.txt', create recursive path
            if not any(c in pattern for c in ['/', '\\', '*']): # Simple pattern
                search_path_recursive = os.path.join(directory, '**', pattern)
            else: # Pattern already has path components or complex wildcards
                search_path_recursive = os.path.join(directory, pattern)

            found_files = glob.glob(search_path_recursive, recursive=True)
        else:
            found_files = glob.glob(search_path)
            
        # Filter out directories if the pattern accidentally matches them (e.g. "data*")
        actual_files = [f for f in found_files if os.path.isfile(f)]

        if not actual_files:
            return f"No files found matching pattern '{pattern}' in directory '{directory}' (recursive={recursive})."
        
        return f"Files found matching '{pattern}' in '{directory}':\n" + "\n".join(actual_files)
    except Exception as e:
        return f"Error finding files in {directory} with pattern {pattern}: {str(e)}"

@tool
def unzip_file(zip_filepath: str, extract_to_dir: str) -> str:
    """
    Extracts the contents of a ZIP archive to a specified directory.

    Args:
        zip_filepath: The path to the .zip file.
        extract_to_dir: The directory where the contents should be extracted.
                        It will be created if it doesn't exist.

    Returns:
        A message indicating success or failure.
    """
    if not os.path.exists(zip_filepath):
        return f"Error: ZIP file not found at {zip_filepath}."
    if not zipfile.is_zipfile(zip_filepath):
        return f"Error: {zip_filepath} is not a valid ZIP file."

    try:
        os.makedirs(extract_to_dir, exist_ok=True)
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to_dir)
        return f"Successfully extracted '{zip_filepath}' to '{extract_to_dir}'."
    except zipfile.BadZipFile:
        return f"Error: Bad ZIP file or unsupported compression method in {zip_filepath}."
    except Exception as e:
        return f"Error extracting ZIP file {zip_filepath}: {str(e)}"

@tool
def zip_files(items_to_zip: list[str], output_zip_path: str, base_dir_to_arc_from: str | None = None) -> str:
    """
    Compresses specified files or directories into a single ZIP archive.

    Args:
        items_to_zip: A list of paths to files or directories to be added to the ZIP file.
        output_zip_path: The full path for the output .zip file (e.g., '/path/to/archive.zip').
        base_dir_to_arc_from: Optional. A common base directory. If provided, paths in the
                              ZIP file will be relative to this directory. Otherwise, paths
                              will be relative to the item's own directory or be absolute.
                              For example, to zip 'project/src/file.txt' as 'src/file.txt' in the zip,
                              set base_dir_to_arc_from to 'project'.

    Returns:
        A message indicating success or failure.
    """
    try:
        # Ensure output directory exists
        output_zip_dir = os.path.dirname(output_zip_path)
        if output_zip_dir:
            os.makedirs(output_zip_dir, exist_ok=True)

        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for item_path in items_to_zip:
                if not os.path.exists(item_path):
                    return f"Error: Item '{item_path}' not found. Aborting zip operation."
                
                if os.path.isfile(item_path):
                    arcname = None
                    if base_dir_to_arc_from:
                        arcname = os.path.relpath(item_path, start=base_dir_to_arc_from)
                    else: # Use filename if no base_dir, or make relative to its own dir if full path
                        arcname = os.path.basename(item_path)
                    zipf.write(item_path, arcname)
                elif os.path.isdir(item_path):
                    for root, _, files in os.walk(item_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = None
                            if base_dir_to_arc_from:
                                arcname = os.path.relpath(file_path, start=base_dir_to_arc_from)
                            else: # Make path relative to the directory being zipped
                                arcname = os.path.relpath(file_path, start=os.path.dirname(item_path) if item_path.endswith(os.sep) else item_path)
                                if arcname == ".": # if item_path is a file and base_dir is its dir
                                    arcname = os.path.basename(file_path)
                            zipf.write(file_path, arcname)
        return f"Successfully created ZIP file: {output_zip_path} containing {len(items_to_zip)} item(s)/root(s)."
    except Exception as e:
        return f"Error creating ZIP file {output_zip_path}: {str(e)}"

@tool
def grep_directory(directory: str, search_pattern: str, file_pattern: str = "*", recursive: bool = True, case_sensitive: bool = False) -> str:
    """
    Searches for a text pattern (regex) in files within a directory.

    Args:
        directory: The directory to search in.
        search_pattern: The regular expression to search for.
        file_pattern: Glob pattern for filenames to search within (e.g., "*.txt", "*.py"). Defaults to "*".
        recursive: If True, searches subdirectories. Defaults to True.
        case_sensitive: If False, the search is case-insensitive. Defaults to False.

    Returns:
        A string listing matching files and lines, or a message if no matches are found.
    """
    if not os.path.isdir(directory):
        return f"Error: Directory '{directory}' not found."

    results = []
    regex_flags = 0 if case_sensitive else re.IGNORECASE
    try:
        compiled_regex = re.compile(search_pattern, regex_flags)
    except re.error as e:
        return f"Error: Invalid regex pattern '{search_pattern}': {e}"

    for root, _, files in os.walk(directory):
        for filename in files:
            if fnmatch.fnmatch(filename, file_pattern):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if compiled_regex.search(line):
                                results.append(f"{os.path.relpath(filepath, directory)}:{line_num}: {line.strip()}")
                except Exception as e:
                    results.append(f"Error reading file {filepath}: {str(e)}")
        if not recursive:
            break # Only process the top directory if not recursive

    if not results:
        return f"No matches found for pattern '{search_pattern}' in files matching '{file_pattern}' within '{directory}'."
    return f"Search results for '{search_pattern}' in '{directory}':\n" + "\n".join(results)

@tool
def read_pdf_content(filepath: str, max_pages: int = 0) -> str:
    """
    Extracts text content from a PDF file.
    Requires the 'PyPDF2' package: `pip install PyPDF2`.

    Args:
        filepath: The path to the PDF file.
        max_pages: Maximum number of pages to read. If 0, reads all pages. Defaults to 0.

    Returns:
        The extracted text content as a string, or an error message.
    """
    try:
        import PyPDF2 # type: ignore
    except ImportError:
        return "Error: The 'PyPDF2' library is not installed. Please install it using 'pip install PyPDF2'."

    if not os.path.exists(filepath):
        return f"Error: PDF file not found at {filepath}."
    if not filepath.lower().endswith(".pdf"):
        return f"Error: File {filepath} does not appear to be a PDF file."

    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            if reader.is_encrypted:
                # Try to decrypt with an empty password, common for some "locked" PDFs
                try:
                    if reader.decrypt('') == PyPDF2.PasswordType.OWNER_PASSWORD: # or USER_PASSWORD
                         pass # Successfully decrypted
                    else: # This branch might indicate it needs a real password or failed for other reason
                        return f"Error: PDF file {filepath} is encrypted and could not be decrypted with an empty password."
                except Exception: # PyPDF2 can raise various things on decrypt failure
                    return f"Error: PDF file {filepath} is encrypted and decryption failed."


            text_content = []
            num_pages_to_read = len(reader.pages)
            if max_pages > 0:
                num_pages_to_read = min(num_pages_to_read, max_pages)

            for i in range(num_pages_to_read):
                page = reader.pages[i]
                text_content.append(page.extract_text() or "") # Ensure empty string if None

        full_text = "\n".join(text_content).strip()
        if not full_text:
            return f"No text could be extracted from {filepath}. The PDF might contain images of text or use complex encoding."
        return f"Extracted text from {filepath} (first {num_pages_to_read} pages):\n{full_text[:5000]}{'...' if len(full_text) > 5000 else ''}" # Limit output
    except Exception as e:
        return f"Error reading PDF file {filepath}: {str(e)}"

# --- Execution & Environment Tools ---

@tool
def check_python_package_version(package_name: str) -> str:
    """
    Checks if a specific Python package is installed and returns its version.

    Args:
        package_name: The name of the Python package.

    Returns:
        A string indicating the package version or if it's not found.
    """
    try:
        version = importlib_metadata.version(package_name)
        return f"Package '{package_name}' is installed with version: {version}"
    except importlib_metadata.PackageNotFoundError:
        return f"Package '{package_name}' is not installed."
    except Exception as e:
        return f"Error checking package version for '{package_name}': {str(e)}"

@tool
def list_installed_python_packages() -> str:
    """
    Lists all installed Python packages and their versions.

    Returns:
        A string listing installed packages and their versions, or an error message.
    """
    try:
        distributions = importlib_metadata.distributions()
        packages = []
        for dist in distributions:
            packages.append(f"{dist.name} ({dist.version})")
        
        if not packages:
            return "No Python packages found."
        return "Installed Python packages:\n" + "\n".join(sorted(packages))
    except Exception as e:
        return f"Error listing installed Python packages: {str(e)}"

# --- Web Interaction Tools ---

@tool
def search_google_scholar(query: str, max_results: int = 5) -> str:
    """
    Searches Google Scholar for academic papers.
    Requires the 'scholarly' package: `pip install scholarly`.

    Args:
        query: The search query.
        max_results: The maximum number of results to return. Defaults to 5.

    Returns:
        A formatted string with search results, or an error message.
    """
    try:
        from scholarly import scholarly # type: ignore
        from scholarly import MaxRetriesExceededException, ProxyError # type: ignore
    except ImportError:
        return "Error: The 'scholarly' library is not installed. Please install it using 'pip install scholarly'."

    results_output = []
    try:
        # It's good practice to set a proxy if you're making many requests,
        # but for a simple tool, direct access might work for a while.
        # scholarly.use_proxy(http="your_proxy", https="your_proxy")
        
        search_query = scholarly.search_pubs(query)
        count = 0
        for i in range(max_results): # scholarly's generator can be slow or hit limits
            try:
                pub = next(search_query)
                if pub:
                    title = pub.get('bib', {}).get('title', 'N/A')
                    authors = ", ".join(pub.get('bib', {}).get('author', ['N/A']))
                    venue = pub.get('bib', {}).get('venue', 'N/A')
                    pub_year = pub.get('bib', {}).get('pub_year', 'N/A')
                    abstract = pub.get('bib', {}).get('abstract', 'N/A')
                    url = pub.get('pub_url', pub.get('eprint_url', 'N/A')) # pub_url is often paywalled, eprint_url might be a PDF

                    results_output.append(
                        f"Title: {title}\n"
                        f"Authors: {authors}\n"
                        f"Venue: {venue} ({pub_year})\n"
                        f"Abstract: {abstract[:300]}...\n"
                        f"URL: {url}\n"
                        f"Citations: {pub.get('num_citations', 'N/A')}"
                    )
                    count +=1
                if count >= max_results:
                    break
            except StopIteration:
                break # No more results
            except MaxRetriesExceededException:
                results_output.append("Note: scholarly library hit max retries. Results may be incomplete.")
                break
            except ProxyError:
                results_output.append("Note: scholarly library encountered a proxy error. Are you rate-limited or is network/proxy misconfigured?")
                break
            except Exception as e: # Catch other potential errors from scholarly per item
                results_output.append(f"Note: Error fetching a specific Scholar result: {e}")
                continue


        if not results_output:
            return f"No results found on Google Scholar for query: {query}"
        return f"Google Scholar results for '{query}':\n\n" + "\n\n---\n\n".join(results_output)
    except MaxRetriesExceededException:
        return "Error searching Google Scholar: Max retries exceeded. You might be rate-limited. Try using a proxy with scholarly."
    except ProxyError:
        return "Error searching Google Scholar: Proxy error. Check your network or proxy configuration for scholarly."
    except Exception as e:
        return f"Error searching Google Scholar for '{query}': {str(e)}"


# --- Agent Specific/Memory Tools ---

@tool
def manage_agent_tasks(action: str, task_description: str | None = None, task_id: int | None = None, tasks_file: str | None = None) -> str:
    """
    Manages an agent's to-do list stored in a JSON file.
    Actions: "add", "remove", "list", "complete", "uncomplete", "clear".

    Args:
        action: The operation to perform: "add", "remove", "list", "complete", "uncomplete", "clear".
        task_description: The description of the task (required for "add").
        task_id: The ID of the task (required for "remove", "complete", "uncomplete").
        tasks_file: Path to the JSON file storing tasks.

    Returns:
        A message indicating the result of the action.
    """
    action = action.lower()
    valid_actions = ["add", "remove", "list", "complete", "uncomplete", "clear"]
    if action not in valid_actions:
        return f"Error: Invalid action '{action}'. Valid actions are: {', '.join(valid_actions)}."

    # Ensure tasks directory exists
    try:
        os.makedirs(os.path.dirname(tasks_file), exist_ok=True)
    except Exception as e:
        return f"Error creating directory for tasks file: {str(e)}"
        
    # Load tasks
    tasks = []
    if os.path.exists(tasks_file):
        try:
            with open(tasks_file, "r", encoding="utf-8") as f:
                tasks = json.load(f)
        except json.JSONDecodeError:
            return f"Error: Tasks file '{tasks_file}' is corrupted. Could not decode JSON."
        except Exception as e:
            return f"Error reading tasks file '{tasks_file}': {str(e)}"

    # Perform action
    if action == "add":
        if not task_description:
            return "Error: Task description is required for 'add' action."
        new_id = max(task.get("id", 0) for task in tasks) + 1 if tasks else 1
        tasks.append({"id": new_id, "description": task_description, "status": "pending", "created_at": datetime.now().isoformat()})
        message = f"Task '{task_description}' added with ID {new_id}."
    
    elif action == "list":
        if not tasks:
            return "No tasks in the list."
        output = ["Current tasks:"]
        for task in tasks:
            output.append(f"- ID {task['id']}: {task['description']} (Status: {task['status']})")
        return "\n".join(output)

    elif action == "clear":
        tasks = []
        message = "All tasks have been cleared."

    else: # Actions requiring task_id: remove, complete, uncomplete
        if task_id is None:
            return f"Error: Task ID is required for '{action}' action."
        
        task_found = False
        for task in tasks:
            if task.get("id") == task_id:
                task_found = True
                if action == "remove":
                    tasks.remove(task)
                    message = f"Task ID {task_id} removed."
                elif action == "complete":
                    task["status"] = "completed"
                    task["completed_at"] = datetime.now().isoformat()
                    message = f"Task ID {task_id} marked as completed."
                elif action == "uncomplete":
                    task["status"] = "pending"
                    if "completed_at" in task:
                        del task["completed_at"]
                    message = f"Task ID {task_id} marked as pending."
                break
        if not task_found:
            return f"Error: Task ID {task_id} not found."

    # Save tasks
    try:
        with open(tasks_file, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2)
        return message
    except Exception as e:
        return f"Error writing tasks file '{tasks_file}': {str(e)}"

@tool
def retrieve_agent_log_segment(
    filename: str,
    lines_from_end: int | None = None,
    lines_from_start: int | None = None,
    keyword: str | None = None,
    case_sensitive_keyword: bool = False,
    log_dir: str | None = None
) -> str:
    """
    Retrieves segments or filtered content from a specified agent log file.

    Args:
        filename: The name of the log file within the agent's log directory.
        lines_from_end: Optional. Number of lines to retrieve from the end of the file (tail).
        lines_from_start: Optional. Number of lines to retrieve from the start of the file (head).
                           If both start and end are given, end takes precedence for simple tail/head.
                           If keyword is also used, this can define the initial set of lines to filter.
        keyword: Optional. A keyword to filter lines by. Only lines containing this keyword will be returned.
        case_sensitive_keyword: Optional. If True, keyword search is case-sensitive. Defaults to False.
        log_dir: The base directory for agent logs.

    Returns:
        The requested log segment as a string, or an error message.
    """
    filepath = os.path.join(log_dir, filename)
    if not os.path.exists(filepath):
        return f"Error: Log file not found at {filepath}."
    if not os.path.isfile(filepath):
        return f"Error: {filepath} is not a file."

    try:
        with open(filepath, "r", encoding="utf-8", errors='ignore') as f:
            all_lines = f.readlines()

        if not all_lines:
            return f"Log file '{filename}' is empty."

        # Apply keyword filter first if present
        if keyword:
            if not case_sensitive_keyword:
                keyword_to_search = keyword.lower()
                filtered_lines = [line for line in all_lines if keyword_to_search in line.lower()]
            else:
                keyword_to_search = keyword
                filtered_lines = [line for line in all_lines if keyword_to_search in line]
            
            if not filtered_lines:
                 return f"No lines containing keyword '{keyword}' found in '{filename}'."
            lines_to_process = filtered_lines
        else:
            lines_to_process = all_lines

        # Apply line count limits
        if lines_from_end is not None and lines_from_end > 0:
            output_lines = lines_to_process[-lines_from_end:]
        elif lines_from_start is not None and lines_from_start > 0:
            output_lines = lines_to_process[:lines_from_start]
        else: # No specific line count, return all (potentially keyword-filtered) lines
            output_lines = lines_to_process
        
        if not output_lines: # Could happen if keyword filtered everything or counts were off
            return f"No matching log entries found in '{filename}' with the specified criteria."
            
        return f"Log segment from '{filename}':\n" + "".join(output_lines)

    except Exception as e:
        return f"Error retrieving log segment from {filepath}: {str(e)}"
    
@tool
def ask_user_for_input(prompt_message: str) -> str:
    """
    Prompts the human user for input and returns their response.
    This tool should be used when the agent needs clarification, a decision,
    or information it cannot find on its own.

    Args:
        prompt_message: The question or message to display to the user.

    Returns:
        A string containing the user's typed response.
        The response will be prefixed with "User input: ".
    """
    print(f"\nAGENT NEEDS YOUR INPUT:")
    print(prompt_message)
    
    try:
        user_response = input("Your response: ")
        return f"User input: {user_response}"
    except EOFError:
        return "User input: [No input provided - EOFError]. The user may have terminated the input stream."
    except KeyboardInterrupt:
        return "User input: [Input interrupted by user - KeyboardInterrupt]. The agent should probably stop or try a different approach."
    except Exception as e:
        return f"User input: [Error during input: {str(e)}]. Something went wrong while trying to get user input."
