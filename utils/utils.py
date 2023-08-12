import os
import pandas as pd
import datetime
from tqdm import tqdm

def get_dataset(params:dict=None)->pd.DataFrame:
    '''
        Import data

        Parameters
        ----------
        params: dict
            dictionary with parameters i.e paths, ...

        Returns
        -------
        DataFrame containing the files with the corresponding class
    '''


    # Get directory names ie classes
    directories_list = [f for f in os.listdir(params['data_path']) if os.path.isdir(os.path.join(params['data_path'], f))]

    # Parse all directory and get the contained files
    files, labels = [], []
    for directory in directories_list:
        # Get files
        path = os.path.join(params['data_path'], directory)
        files_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))] 

        files += [os.path.join(path, file) for file in files_list]
        labels += [ directory ] * len(files_list)

    # Store information in DataFrame
    df = pd.DataFrame({})
    df['Files'] = files
    df['Labels'] = labels

    return df


def create_instances(df:pd.DataFrame=None, number_of_iterations:int=None)->list:
    '''
        Training instances (positive, negative) for training and evaluating
        Siamese network

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing the filenames along with their corresponsing class
        number_of_iterations: int
            Number of iteration for creating positive & negative instances


        Returns
        -------
        list with positive and negative examples
    '''
    # Create dataset for Siamese network
    # ---------------------------------------------------------
    # Each instance of the dataset has the form: [anchor image, positive/negative image, class]

    # Sanity check
    
    data = []
    for iterations in tqdm(range(number_of_iterations)):
        for label in df['Labels'].unique():
            sample_size = min(df[df['Labels'] == label].shape[0], df[df['Labels'] != label].shape[0])
            
            anchors = list( df[df['Labels'] == label].sample(sample_size)['Files'] )

            positives = list( df[df['Labels'] == label].sample(sample_size)['Files'] )
            negatives = list( df[df['Labels'] != label].sample(sample_size)['Files'] )

            for anchor, positive, negative in zip(anchors, positives, negatives):
                data.append( [anchor, positive, 0.0] )
                data.append( [anchor, negative, 1.0] )

    return data




def format_time(elapsed: float):
    '''
        Function for presenting time in 
        
        Parameters
        ----------
        elapsed: time in seconds
        Returns
        -------
        elapsed as string
    '''
    return str(datetime.timedelta(seconds=int(round((elapsed)))))