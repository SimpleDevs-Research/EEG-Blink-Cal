import os
import shutil

# Create directory, delete first if already exists
# - Params:
#   - _DIR: the query directory
#   - delete_existing: Should we delete, or keep the directory if it exists?
# - Returns:
#   - the query directory
def mkdirs(_DIR:str, delete_existing:bool=True):
    
    # If the folder already exists, delete it
    if delete_existing and os.path.exists(_DIR): shutil.rmtree(_DIR)

    # Create a new empty directory
    os.makedirs(_DIR, exist_ok=True)

    # Return the directory to indicate completion
    return _DIR

