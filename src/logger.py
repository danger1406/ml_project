import logging
import os
from datetime import datetime


LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
#So here we also need to give the log path which is the combination of path.join Means we are combining multiple paths. The first one gives the current working directory, and we are adding logs to it and the file log file. 
# logs and that naming convention every file will start with logs and the string that we have defined bassically

os.makedirs(logs_path,exist_ok=True)
#This means that even though there is a folder append the files to that folder 

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,

# This argument sets the threshold for the logger. Only log messages with a severity level equal to or higher than this will be recorded.
)

if __name__=="__main__":
    logging.info("Start aindhi raoio")