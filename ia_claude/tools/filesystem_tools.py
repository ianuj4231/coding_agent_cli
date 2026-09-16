import os
from langchain.tools import tool

from ia_claude.observability.logger import get_logger


_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
logger = get_logger(__name__)


@tool
def read_file(file_path: str) -> str:
   """
   Read and return the contents of a UTF-8 text file.

   Args:
       file_path: Required absolute or relative file path. The tool-call
           argument must be named exactly ``file_path``, not ``path``.
   """
   if not file_path or not file_path.strip():
       return "Error: file path cannot be empty"
   if not os.path.exists(file_path):
       return f"Error: file not found: {file_path}"
   if not os.path.isfile(file_path):
       return f"Error: path is not a file: {file_path}"
   size = os.path.getsize(file_path)
   if size > _MAX_FILE_SIZE_BYTES:
       return f"Error: file too large ({size} bytes). Max allowed is {_MAX_FILE_SIZE_BYTES} bytes"
   try:
       with open(file_path, "r", encoding="utf-8") as f:
           return f.read()
   except UnicodeDecodeError:
       return f"Error: file is not valid UTF-8 text: {file_path}"
   except PermissionError:
       return f"Error: permission denied: {file_path}"
   except Exception as e:
       return f"Error: {e}"


@tool
def write_file(file_path: str, content: str) -> str:
   """
   Write UTF-8 text to a file; the watcher updates the index asynchronously.

   Args:
       file_path: Required absolute or relative file path. The tool-call
           argument must be named exactly ``file_path``, not ``path``.
       content: Required complete text to write. The tool-call argument must
           be named exactly ``content``.
   """
   if not file_path or not file_path.strip():
       return "Error: file path cannot be empty"
   try:
       if os.path.dirname(file_path):
           os.makedirs(os.path.dirname(file_path), exist_ok=True)
       with open(file_path, "w", encoding="utf-8") as f:
           f.write(content)
       logger.info("Filesystem write succeeded: path=%s bytes=%d", file_path, len(content.encode("utf-8")))
       return f"Written {file_path}; indexing scheduled by filesystem watcher"
   except PermissionError:
       return f"Error: permission denied: {file_path}"
   except Exception as e:
       return f"Error: {e}"




@tool
def append_file(file_path: str, content: str) -> str:
   """
   Append UTF-8 text to a file; the watcher updates the index asynchronously.

   Args:
       file_path: Required absolute or relative file path. The tool-call
           argument must be named exactly ``file_path``, not ``path``.
       content: Required text to append. The tool-call argument must be named
           exactly ``content``.
   """
   if not file_path or not file_path.strip():
       return "Error: file path cannot be empty"
   if not os.path.exists(file_path):
       return f"Error: file not found: {file_path}"
   if not os.path.isfile(file_path):
       return f"Error: path is not a file: {file_path}"
   try:
       with open(file_path, "a", encoding="utf-8") as f:
           f.write(content)
       logger.info("Filesystem append succeeded: path=%s bytes=%d", file_path, len(content.encode("utf-8")))
       return f"Appended to {file_path}; indexing scheduled by filesystem watcher"
   except PermissionError:
       return f"Error: permission denied: {file_path}"
   except Exception as e:
       return f"Error: {e}"


@tool
def delete_file(file_path: str) -> str:
   """
   Delete a file; the watcher removes its index entries asynchronously.

   Args:
       file_path: Required absolute or relative file path. The tool-call
           argument must be named exactly ``file_path``, not ``path``.
   """
   if not file_path or not file_path.strip():
       return "Error: file path cannot be empty"
   if not os.path.exists(file_path):
       return f"Error: file not found: {file_path}"
   if not os.path.isfile(file_path):
       return f"Error: path is not a file (use a directory tool for directories): {file_path}"
   try:
       os.remove(file_path)
       logger.info("Filesystem delete succeeded: path=%s", file_path)
       return f"Deleted {file_path}; index cleanup scheduled by filesystem watcher"
   except PermissionError:
       return f"Error: permission denied: {file_path}"
   except Exception as e:
       return f"Error: {e}"


@tool
def list_directory(directory: str) -> str:
   """
   List files and subdirectories inside a directory.

   Args:
       directory: Required absolute or relative directory path. The tool-call
           argument must be named exactly ``directory``, not ``path``.
   """
   if not directory or not directory.strip():
       return "Error: directory cannot be empty"
   if not os.path.exists(directory):
       return f"Error: directory not found: {directory}"
   if not os.path.isdir(directory):
       return f"Error: path is not a directory: {directory}"
   try:
       entries = os.listdir(directory)
       return "\n".join(sorted(entries)) if entries else "(empty directory)"
   except PermissionError:
       return f"Error: permission denied: {directory}"
   except Exception as e:
       return f"Error: {e}"


@tool
def file_exists(file_path: str) -> str:
   """
   Check whether a file or directory exists.

   Args:
       file_path: Required absolute or relative path. The tool-call argument
           must be named exactly ``file_path``, not ``path``.
   """
   if not file_path or not file_path.strip():
       return "Error: file path cannot be empty"
   return str(os.path.exists(file_path))


