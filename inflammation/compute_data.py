"""Module containing mechanism for calculating standard deviation between datasets.
"""

import glob
import os
import numpy as np

from inflammation import models, views
from models import JSONDataSource

class CSVDataSource:
    """Class to represent a data source for inflammation data from CSV files."""
    def __init__(self, dir_path):
        self.dir_path = dir_path
        
    def load_inflammation_data(self):
        """Loads inflammation data from a CSV file.

<<<<<<< HEAD
        Args:
            file_path (str): The path to the CSV file containing inflammation data.
        Returns:
            np.ndarray: A 2D array where each row represents a patient and each column represents a day.
        """
        data_file_paths = glob.glob(os.path.join(self.dir_path, 'inflammation*.csv'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation CSV files found in path {self.dir_path}")
        data = map(models.load_csv, data_file_paths)
        return list(data)
    
def compute_standard_deviation_by_day(data):
=======
def analyse_data(data_dir):
    """Calculates the standard deviation by day between datasets.

    Gets all the inflammation data from CSV files within a directory,
    works out the mean inflammation value for each day across all datasets,
    then plots the graphs of standard deviation of these means."""
    data_file_paths = glob.glob(os.path.join(data_dir, 'inflammation*.csv'))
    if len(data_file_paths) == 0:
        raise ValueError(f"No inflammation data CSV files found in path {data_dir}")
    data = map(models.load_csv, data_file_paths)


>>>>>>> 9946564 (Improved formatting and documentation)
    means_by_day = map(models.daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))

    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)
    return daily_standard_deviation


# def analyse_data(data_source):
#     """Calculates the standard deviation by day between datasets.

#     Gets all the inflammation data from CSV files within a directory,
#     works out the mean inflammation value for each day across all datasets,
#     then plots the graphs of standard deviation of these means."""
#     data = data_source.load_inflammation_data()

#     means_by_day = map(models.daily_mean, data)
#     means_by_day_matrix = np.stack(list(means_by_day))

#     daily_standard_deviation = np.std(means_by_day_matrix, axis=0)

#     graph_data = {
#         'standard deviation by day': daily_standard_deviation,
#     }
#     # views.visualize(graph_data)
#     return graph_data

def analyse_data(data_dir):
    """Calculates the standard deviation by day between datasets.
    Gets all the inflammation data from CSV files within a directory, works out the mean
    inflammation value for each day across all datasets, then visualises the
    standard deviation of these means on a graph."""
    data_file_paths = glob.glob(os.path.join(data_dir, 'inflammation*.csv'))
    if len(data_file_paths) == 0:
        raise ValueError(f"No inflammation csv's found in path {data_dir}")
    
    data = map(models.load_csv, data_file_paths)
    daily_standard_deviation = compute_standard_deviation_by_day(data)
    
    return daily_standard_deviation


if __name__ == '__main__':

    #Before polymorphism example
    # infiles = glob.glob(os.path.join('data', 'inflammation*.csv'))
    # data_source = CSVDataSource('data')
    # analyse_data(data_source)

    # Determine data source type based on file extension - polymorphism example
    infiles = glob.glob(os.path.join('data', 'inflammation*.*'))
    _, extension = os.path.splitext(infiles[0])
    if extension == '.json':
        data_source = JSONDataSource(os.path.dirname(infiles[0]))
    elif extension == '.csv':
        data_source = CSVDataSource(os.path.dirname(infiles[0]))
    else:
        raise ValueError(f'Unsupported data file format: {extension}')
    analyse_data(data_source)


