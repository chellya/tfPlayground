import tensorflow as tf
import numpy as np
from IPython.display import clear_output, Image, display, HTML
import webbrowser, os
import importlib
import pandas as pd
from pandas import DataFrame






''' 
Define the base input function from odps table.

Returns:
  A dict object. 
  Each key refers to one feature.
  Each value is the corresponding data in the form of [Batch_sz, Current_feature_size]

Examples:

Demo schema_config
schema_config =  \
    {
        'history_ids':
        {
            'feat_name': 'history_ids',
            'column_name': 'history_ids',
            'length': 1,
            'dtype': tf.string,
            'string_to_patched_seq': {
                'seq_len': 5,
                'patch_value': '0',
                'delimiter': ';',
                'dtype': tf.int32
            }
        },
        'label_id':
        {
            'feat_name': 'label_id',
            'column_name': 'label_id',
            'length': 1,
            'dtype': tf.int32,
        },
        'is_pos':
        {
            'feat_name': 'is_pos',
            'column_name': 'is_pos',
            'length': 1,
            'dtype': tf.float32,
        },
    }
'''
def input_table_batch_fn(table_name, batch_size, schema_config, allow_smaller_final_batch=True,num_epoches=None,slice_count=None, slice_id=None):
    selected_col = ','.join([e['column_name'] for e in schema_config])
    file_queue = tf.train.string_input_producer([table_name], num_epochs=num_epoches)
    print(selected_col)
    reader = tf.TableRecordReader(slice_count=slice_count, slice_id=slice_id,csv_delimiter=',',
                                                                selected_cols=selected_col,
                                                                num_threads=32, capacity=batch_size*20)
    key, value = reader.read_up_to(file_queue, batch_size)
    batch_res = tf.train.shuffle_batch([value], batch_size=batch_size, capacity=batch_size*20, enqueue_many=True,
                                                                         num_threads=16, min_after_dequeue=batch_size, allow_smaller_final_batch=allow_smaller_final_batch)
    record_defaults = [[''] for _ in range(np.sum([e['length'] for e in schema_config]))]
    feature = tf.decode_csv(batch_res, record_defaults=record_defaults,field_delim=',')

    res = {}
    length = len(schema_config)
    start = 0
    for e in schema_config:
        datatype=e['dtype']
        length = e['length']
        name = e['feat_name']
        if datatype == tf.string:
            b = feature[start:start + length]
        else:
            b = tf.string_to_number(feature[start:start + length], datatype)
        # From [Batch_sz,Feat_sz] to [Feat_sz,Batch_sz]
        val = tf.transpose(b)
        if 'string_to_patched_seq' in e:
            param = e['string_to_patched_seq']
            seq_len = param.get('seq_len',5)
            patch_value = param.get('patch_value','</s>')
            delimiter = param.get('delimiter',';')
            dtype = param.get('dtype',tf.string)
            print(val)
            val = tf.map_fn(lambda x:tf.string_split_and_pad(x,max_length=seq_len,delimiter=delimiter,default_value=patch_value)
                     ,val,parallel_iterations=512,back_prop=False)
            if dtype==tf.string:
                pass 
            else:
                val = tf.string_to_number(val,out_type=dtype)

        res[name]=val
        
        start = start + length

    return res