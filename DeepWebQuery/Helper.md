# Helper for controling ports. Example is for port 8000.

### On Windows:

Open Command Prompt as an administrator.

Run the following command to find out which process is using port 8000:
```
netstat -ano | findstr :8000
```

Look for the line that has 0.0.0.0:8000 or [::]:8000 and note the PID (Process Identifier) at the end of that line.

To find out which application the PID corresponds to, run:
```
tasklist /fi "pid eq <PID>"
```
Replace <PID> with the actual PID number.

Once you know which application is using the port, you can decide if you want to close it. If you do, you can either close the application normally or use the following command to forcefully terminate the process:
```
taskkill /PID <PID> /F
```
Again, replace <PID> with the actual PID number.


### On macOS and Linux:


Open Terminal.

Run the following command to find out which process is using port 8000:
```
sudo lsof -i :8000
```

Look for the PID in the output, which is usually in the second column.

To stop the process, you can use the kill command:
```
sudo kill -9 <PID>
```
Replace <PID> with the actual PID number.