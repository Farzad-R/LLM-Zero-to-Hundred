import os
import pandas as pd
from typing import List


class Memory:
    @staticmethod
    def write_chat_history_to_file(chat_history_lst: List, file_path: str) -> None:
        """
        Writes the chat history list to a CSV file.

        Parameters:
            chat_history_lst (List[Tuple[str, str]]): The chat history list to be written to the file.
            file_path (str): The path to the CSV file where the chat history will be written.

        Returns:
            None
        """
        # Create a DataFrame from the chat history list
        df = pd.DataFrame(chat_history_lst, columns=[
                          "User query", "Response"])

        # Check if the file exists and is not empty to avoid writing headers again
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            # If the file exists, append without headers
            df.to_csv(file_path, mode='a', header=False,
                      index=False, encoding='utf-8')
        else:
            # If the file does not exist, write with headers
            df.to_csv(file_path, mode='w', header=True,
                      index=False, encoding='utf-8')

    @staticmethod
    def read_recent_chat_history(file_path: str, num_entries: int = 2) -> List:
        """
        Reads the most recent entries from the chat history CSV file.

        Parameters:
            file_path (str): The path to the CSV file from which to read the chat history.
            num_entries (int): The number of recent entries to read from the chat history.

        Returns:
            List[str]: A list of the most recent chat history entries as strings, or an empty list if an error occurs.
        """
        try:
            recent_history = []
            last_rows = pd.read_csv(file_path).tail(num_entries)
            for _, row in last_rows.iterrows():
                row_dict = row.to_dict()
                recent_history.append(str(row_dict))

            return recent_history
        except Exception as e:
            print(f"Chat history could not be loaded. {e}")
            return []
