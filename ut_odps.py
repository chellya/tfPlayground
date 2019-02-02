import tensorflow as tf
import numpy as np
from IPython.display import clear_output, Image, display, HTML
import webbrowser, os
import importlib
import pandas as pd
from pandas import DataFrame
from odps import ODPS
from odps.df import DataFrame as odpsDataFrame
from odps.df import Scalar
from odps import options
# configure lifecycle for all output tables (option lifecycle)
options.lifecycle = 30
# handle string type as bytes when downloading with Tunnel (option tunnel.string_as_binary)
options.tunnel.string_as_binary = True
# get more records when sorting the DataFrame with MaxCompute
options.df.odps.sort.limit = 100000000
# print execution infomation and logview url
options.verbose = True
o = ODPS('HBx7pwRyDFAzPzsp', '0q43MHenOKSTV49ekfzVFn12SX2Dhx',
         project='trip_search', endpoint='http://service.odps.aliyun-inc.com/api')

def get_data_by_partition(df, start_pt, end_pt=None, name='ds'):
    '''Get data by partition.
    
    Both start and end are inclusive.
    
    Args:
        df: Input dataframe.
        start_pt: Starting partition (Inclusive).
        end_pt: (Optional.) Ending partition (Inclusive). If None (the default),
            end_pt is the last partition.
        name: (Optional.) Field name of partition. Defaults to `ds`.
        
    Returns:
        A dataframe contains specified partitions.
    '''
    if end_pt is None:
        return df[df[name]>=start_pt]
    else:
        return df[df[name]>=start_pt][df[name]<=end_pt]