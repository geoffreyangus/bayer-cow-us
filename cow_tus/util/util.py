"""
"""
import os
import os.path as path

def require_dir(dir_str):
    """
    """
    if not(os.path.exists(dir_str)):
        require_dir(os.path.dirname(dir_str))
        os.mkdir(dir_str)