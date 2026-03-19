"""
Write dump and load functions to
    implement json serialization/deserialization for dict with tuple as key
Serialization idea:
    1.turn to kStr_2_kv={str(k):{
        'k':k,
        'v':v
    }for k,v in ...}
    2.save to json

"""

# from misc_util import myJSONEncoder
from ..misc_util import myJSONEncoder

import json
class DictWithTupleAsKey:
    @staticmethod
    def dump(tuple_key_dict, file_path):
        def tuple_key_dict_to_str_key_dict(tuple_key_dict):
            str_key_dict = {str(k): {'k': k, 'v': v} for k, v in tuple_key_dict.items()}
            return str_key_dict

        str_key_dict = tuple_key_dict_to_str_key_dict(tuple_key_dict)
        with open(file_path, 'w') as f:
            json.dump(str_key_dict, f,cls=myJSONEncoder)

    @staticmethod
    def load(file_path):
        def str_key_dict_to_tuple_key_dict(str_key_dict):
            tuple_key_dict = {tuple(v['k']): v['v'] for k, v in str_key_dict.items()}
            return tuple_key_dict

        with open(file_path, 'r') as f:
            str_key_dict = json.load(f)
        tuple_key_dict = str_key_dict_to_tuple_key_dict(str_key_dict)
        return tuple_key_dict
if __name__ == '__main__':
    dic = {
        ('a', 1): [12, 35],
        ('a', 2): 13,
    }

    DictWithTupleAsKey.dump(dic, 'ttt435.json')
    dic2 = DictWithTupleAsKey.load('ttt435.json')
    print(dic2)