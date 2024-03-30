import os
import shutil
import tiktoken


class Apputils:
    @staticmethod
    def create_directory(directory_path: str) -> None:
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
    def remove_directory(directory_path: str) -> None:
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

    @staticmethod
    def find_latest_chroma_folder(folder_path: str) -> str:
        """
        Find the latest Chroma folder within the specified directory.

        Args:
            folder_path (str): The path to the directory containing Chroma folders.

        Returns:
            str: The path of the folder with the latest timestamp.

        Note:
            This method identifies Chroma folders based on their subdirectory structure,
            extracts Unix timestamps from their names, and returns the path of the folder
            with the latest timestamp.

        Example:
            ```python
            latest_folder = YourClass.find_latest_chroma_folder("/path/to/chroma_folders")
            print(latest_folder)
            ```
        """
        # Get a list of subdirectories in the specified folder
        subfolders = [f for f in os.listdir(
            folder_path) if os.path.isdir(os.path.join(folder_path, f))]

        if not subfolders:
            print("No chroma folders found.")
            return None

        # Extract Unix timestamps from folder names
        timestamps = [int(subfolder.split('_')[1]) for subfolder in subfolders]

        # Find the index of the folder with the latest timestamp
        latest_index = timestamps.index(max(timestamps))

        # Get the path of the folder with the latest timestamp
        latest_folder = os.path.join(folder_path, subfolders[latest_index])

        return latest_folder

    @staticmethod
    def count_num_tokens(text: str, model: str) -> int:
        """
        Returns the number of tokens in the given text.
        Args:
            text (str): The text to count tokens in.
            model (str, optional): The name of the GPT model to use. Defaults to the model specified in the app config.

        Returns:
            int: The number of tokens in the text.
        """
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
