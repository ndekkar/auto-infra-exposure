"""
This file is part of EIRA 
Copyright (C) 2024 GFDRR, World Bank
Developers: 
- Jose Antonio Leon Torres  
- Luc Marius Jacques Bonnafous
- Natalia Romero

Your should have received a copy of the GNU Geneal Public License along
with EIRA.
----------

file handler

This book defines classes to handle files 
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional
from urllib.parse import urlencode, urlparse, parse_qs


# https://www.youtube.com/watch?v=eUF8NZPuM_4
#link to generate a downloaded link for a file saved in one drive


def download_onedrive_file(onedrive_url, finalfilename: str, target_folder: str):
    """
    Downloads a file from OneDrive to a specified target folder.

    :param onedrive_url: The OneDrive shareable link (must be converted to a direct download link).
    :param target_folder: The folder where the file should be saved.
    :return: The full path of the downloaded file.
    """
    # Check if the file already exist
    # Construct the full path to the file
    file_path = os.path.join(target_folder, finalfilename)
    
    # Check if the file exists
    if os.path.isfile(file_path):
        print (f"The file '{finalfilename}' already exist in the local database")
        return  # Exit the function immediately
    

    # Ensure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        
    
    # Create the full path to save the downloaded file
    file_path = os.path.join(target_folder, finalfilename)

    # Download the file
    try:
        response = requests.get(onedrive_url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        # Save the file to the target path
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        return file_path

    except Exception as e:
        print (f"An unexpected error ocurred : {e}")


def download_onedrive_file_2(onedrive_url: str, finalfilename: str, target_folder: str) -> str:
    """
    Downloads a file from OneDrive to a specified target folder, with a progress bar.

    :param onedrive_url: The OneDrive shareable link (must be converted to a direct download link).
    :param finalfilename: The desired name for the downloaded file.
    :param target_folder: The folder where the file should be saved.
    :return: The full path of the downloaded file.
    """
    # Construct the full path to the file
    file_path = os.path.join(target_folder, finalfilename)

    # Check if the file already exists
    if os.path.isfile(file_path):
        print(f"The file '{finalfilename}' already exists in the local database.")
        return file_path  # Return the existing file path

    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    try:
        # Start downloading the file
        #print(f"Downloading the file from OneDrive URL: {onedrive_url}") #For Debbuging
        print(f"Downloading the file from Remote EIRA Database")
        response = requests.get(onedrive_url, stream=True, verify=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Get total file size from the headers
        total_size = int(response.headers.get('content-length', 0))

        # Initialize tqdm progress bar
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=finalfilename) as progress_bar:
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        file.write(chunk)
                        progress_bar.update(len(chunk))  # Update progress bar

        print(f"File '{finalfilename}' has been downloaded successfully to '{file_path}'.")
        return file_path

    except requests.exceptions.RequestException as req_err:
        print(f"HTTP error occurred: {req_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return ""



def download_onedrive_file_imp(onedrive_url: str, finalfilename: str, target_folder: str) -> Optional[str]:
    """
    Downloads a file from OneDrive to a specified target folder.

    :param onedrive_url: The OneDrive shareable link (must be converted to a direct download link).
    :param finalfilename: The name to save the file as.
    :param target_folder: The folder where the file should be saved.
    :return: The full path of the downloaded file or None if an error occurs.
    """
    # Ensure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Construct the full path to the file
    file_path = os.path.join(target_folder, finalfilename)

    # Check if the file already exists
    if os.path.isfile(file_path):
        print(f"The file '{finalfilename}' already exists in the local database.")
        return file_path  # Return the existing file path

    # HTTP headers to mimic a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "*/*",
        "Connection": "keep-alive",
    }

    # Download the file
    try:
        with requests.get(onedrive_url, headers=headers, stream=True, allow_redirects=True) as response:
            response.raise_for_status()  # Raise an HTTPError for bad responses
            
            # Save the file to the target path
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        file.write(chunk)

        print(f"File downloaded successfully to '{file_path}'.")
        return file_path

    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except OSError as os_err:
        print(f"File system error occurred: {os_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return None  # Return None if any error occurs






def download_file(url, target_folder):
    """
    Downloads a file from a URL to a specified target folder and returns the full path of the downloaded file.

    :param url: The URL of the file to download.
    :param target_folder: The folder where the file should be saved.
    :return: The full path of the downloaded file.
    """
    # Ensure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Get the file name from the URL
    file_name = os.path.basename(url)
    
    # Create the full path for the downloaded file
    file_path = os.path.join(target_folder, file_name)
    
    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check if the request was successful
    
    # Save the file to the target path
    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    
    return file_path




def download_file_2(url, download_dir=None, overwrite=True):
    """Download file from url to given target folder and provide full path of the downloaded file.

    Parameters
    ----------
    url : str
        url containing data to download
    download_dir : Path or str, optional
        the parent directory of the eventually downloaded file
        default: local_data.save_dir as defined in climada.conf
    overwrite : bool, optional
        whether or not an already existing file at the target location should be overwritten,
        by default True

    Returns
    -------
    str
        the full path to the eventually downloaded file
    """
    file_name = url.split('/')[-1]
    if file_name.strip() == '':
        raise ValueError(f"cannot download {url} as a file")
    download_path = Path(download_dir)
    file_path = download_path.absolute().joinpath(file_name)
    if file_path.exists():
        if not file_path.is_file() or not overwrite:
            raise FileExistsError(f"cannot download to {file_path}")

    try:
        req_file = requests.get(url, stream=True)
    except IOError as ioe:
        raise type(ioe)('Check URL and internet connection: ' + str(ioe)) from ioe
    if req_file.status_code < 200 or req_file.status_code > 299:
        raise ValueError(f'Error loading page {url}\n'
                         f' Status: {req_file.status_code}\n'
                         f' Content: {req_file.content}')

    total_size = int(req_file.headers.get('content-length', 0))
    block_size = 1024

    #LOGGER.info('Downloading %s to file %s', url, file_path)
    with file_path.open('wb') as file:
        for data in req_file.iter_content(block_size):

            file.write(data)

    return str(file_path)


def convert_to_downloadfile_onedrive(url):
    """
    This function convert the embeded onedrive link into a download link 
    To do it, this function find a specified word and remote it and all characters following it in a string.
    Example: original link: https://1drv.ms/i/s!Am3mBdUhMjIjhYkc_XOVPrCJfMjwLQ?embed=1&width=2928&height=1556
    Procedure: Copy the original link to the intenet browser and click enter. You will see the file. Then copy the new path genereted by the browser. Yo will see someting like this:
    https://onedrive.live.com/embed?resid=23323221D505E66D%2183100&authkey=%21AP1zlT6wiXzI8C0&width=2928&height=1556 
    Then replace the work embed by download.
    Modified link: https://onedrive.live.com/download?resid=23323221D505E66D%2183100&authkey=%21AP1zlT6wiXzI8C0& 
    
    Parameters
    ----------
    url : str
        The original string.
    
    Returns
    -------
    return: The modified string (url) which has the structure to be a download file of onedrive.
    """
        

    return url


