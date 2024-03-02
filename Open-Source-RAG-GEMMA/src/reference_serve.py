import http.server
import socketserver
import yaml
import os
from pyprojroot import here

with open(here("configs/app_config.yml")) as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)

PORT = app_config["serve"]["port"]
DIRECTORY1 = app_config["directories"]["data_directory"]
DIRECTORY2 = app_config["directories"]["data_directory_2"]


class SingleDirectoryHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    Custom HTTP request handler that serves files from a single directory.

    This class extends the SimpleHTTPRequestHandler and sets the serving directory to DIRECTORY1.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the SingleDirectoryHTTPRequestHandler.

        Parameters:
            args: Additional positional arguments for the base class.
            kwargs: Additional keyword arguments for the base class.
        """
        super().__init__(*args, directory=DIRECTORY1, **kwargs)


class MultiDirectoryHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    Custom HTTP request handler that serves files from multiple directories.

    This class extends the SimpleHTTPRequestHandler and allows serving files from DIRECTORY1 and DIRECTORY2
    based on the first directory component in the requested path.
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


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), MultiDirectoryHTTPRequestHandler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()
