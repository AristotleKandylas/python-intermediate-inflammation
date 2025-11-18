"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np
from inflammation import models, views
import glob
import os
import json

class JSONDataSource:
    """Class to represent a data source for inflammation data from JSON files."""
    def __init__(self, dir_path):
        self.dir_path = dir_path
        
    def load_json(filename):
        """Load a numpy array from a JSON document.
        
        Expected format:
        [
        {
            "observations": [0, 1]
        },
        {
            "observations": [0, 2]
        }    
        ]
        :param filename: Filename of CSV to load
        """
        with open(filename, 'r', encoding='utf-8') as file:
            data_as_json = json.load(file)
            return [np.array(entry['observations']) for entry in data_as_json]
        
    def load_inflammation_data(self):
        data_file_paths = glob.glob(os.path.join(self.dir_path, 'inflammation*.json'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation JSON files found in path {self.dir_path}")
        data = map(models.load_json, data_file_paths)
        return list(data)

def load_csv(filename:str) -> np.ndarray:  
    """Load a Numpy array from csv

    Parameters
    ----------
    filename : str
        path to the csv file

    Returns
    -------
    np.ndarray
        2D array of inflammation data
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.

    :param data: A 2D numpy array of inflammation data
    :return: A 1D numpy array of daily mean inflammation values
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.
    
    :param data: A 2D numpy array of inflammation data
    :return: A 1D numpy array of daily max inflammation values
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.
    
    :param data: A 2D numpy array of inflammation data
    :return: A 1D numpy array of daily min inflammation values
    """
    return np.min(data, axis=0)


def s_dev(data):
    """Computes and returns standard deviation for data.

    :param data: Input data for standard deviation calculation
    """
    # Use the NumPy standard deviation function
    std_dev = np.std(data, axis=0)
    
    # Return the result in the original dictionary format
    return std_dev
    
