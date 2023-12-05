"""
Helper for controling ports. Example is for port 8000.

On Windows:

Open Command Prompt as an administrator.

Run the following command to find out which process is using port 8000:
netstat -ano | findstr :8000


Look for the line that has 0.0.0.0:8000 or [::]:8000 and note the PID (Process Identifier) at the end of that line.

To find out which application the PID corresponds to, run:
tasklist /fi "pid eq <PID>"

Replace <PID> with the actual PID number.

Once you know which application is using the port, you can decide if you want to close it. If you do, you can either close the application normally or use the following command to forcefully terminate the process:
taskkill /PID <PID> /F

Again, replace <PID> with the actual PID number.


On macOS and Linux:


Open Terminal.

Run the following command to find out which process is using port 8000:
sudo lsof -i :8000


Look for the PID in the output, which is usually in the second column.

To stop the process, you can use the kill command:
sudo kill -9 <PID>

Replace <PID> with the actual PID number.
"""


import http.server
import socketserver
import yaml
import os

with open("configs/app_config.yml") as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)

PORT = app_config["serve"]["port"]
DIRECTORY1 = app_config["directories"]["data_directory"]
DIRECTORY2 = app_config["directories"]["data_directory_2"]


class SingleDirectoryHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    Custom HTTP request handler that serves files from a single directory.

    This class extends the SimpleHTTPRequestHandler and sets the serving directory to DIRECTORY1.

    Example:
    ```python
    handler = SingleDirectoryHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()
    ```

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the SingleDirectoryHTTPRequestHandler.

        Parameters:
            - args: Additional positional arguments for the base class.
            - kwargs: Additional keyword arguments for the base class.
        """
        super().__init__(*args, directory=DIRECTORY1, **kwargs)


class MultiDirectoryHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    Custom HTTP request handler that serves files from multiple directories.

    This class extends the SimpleHTTPRequestHandler and allows serving files from DIRECTORY1 and DIRECTORY2
    based on the first directory component in the requested path.

    Example:
    ```python
    handler = MultiDirectoryHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()
    ```

    """

    def translate_path(self, path):
        """
        Translate the requested path to the actual file path.

        Parameters:
            path (str): The requested path.

        Returns:
            str: The translated file path.

        """
        # Split the path to get the first directory component
        parts = path.split('/', 2)
        if len(parts) > 1:
            first_directory = parts[1]
            # Check if the first directory matches any of your target directories
            if first_directory == os.path.basename(DIRECTORY1):
                path = os.path.join(DIRECTORY1, *parts[2:])

            elif first_directory == os.path.basename(DIRECTORY2):
                path = os.path.join(DIRECTORY2, *parts[2:])
            else:
                # If the first part of the path is not a directory, check both directories for the file
                file_path1 = os.path.join(DIRECTORY1, first_directory)
                file_path2 = os.path.join(DIRECTORY2, first_directory)
                if os.path.isfile(file_path1):
                    return file_path1
                elif os.path.isfile(file_path2):
                    return file_path2
        # If there's no match, use the default directory
        return super().translate_path(path)


with socketserver.TCPServer(("", PORT), MultiDirectoryHTTPRequestHandler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()
