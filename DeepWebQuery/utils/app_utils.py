import os
import shutil
from typing import List, Dict, Tuple


class Apputils:
    @staticmethod
    def create_directory(directory_path):
        """
        Create a directory if it does not exist.

        Parameters:
            directory_path (str): The path of the directory to be created.

        Example:
        ```python
        create_directory("/path/to/new/directory")
        ```

        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    @staticmethod
    def remove_directory(directory_path):
        """
        Remove a directory if it exists.

        Parameters:
            directory_path (str): The path of the directory to be removed.

        Example:
        ```python
        DirectoryManager.remove_directory("/path/to/existing/directory")
        ```
        """
        if os.path.exists(directory_path):
            # Use shutil.rmtree to remove the directory even if it contains files
            shutil.rmtree(directory_path)
