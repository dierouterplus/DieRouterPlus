import ast
import argparse
import dill
from pathlib import Path

import os
import json

def match_identifier(param_dict, identifier):
    """
    Check if param_dict contains all key-value pairs from the identifier dictionary.
    """
    for key, value in identifier.items():
        """
        If value is of dictionary type, then param_dict[key] must also be of dictionary type.
        Recursively check the nested dictionaries within the dictionary.
        """
        if key not in param_dict:
            return False

        if type(value) is dict:
            if type(param_dict[key]) is not dict or match_identifier(param_dict[key], value) is False:
                return False
        elif param_dict[key] != value:
            return False
    return True


def get_previous_step_output_path(testcase_dir, identifier):
    """
    testcase_dir: directory of testcase. For instance, '../testcase6/'
    identifier is a nested dictionary where the keys are integers and values are feature dictionaries.
    identifier[0] is used to locate the PathFinder result directory under testcase_dir.
    identifier[1] is used to locate the Continuous Optimization result under the PathFinder result.
    """
    def locate_subdirectory(parent_dir, feature_dict):
        for subdir_name in os.listdir(parent_dir):
            subdir = os.path.join(parent_dir, subdir_name)
            if os.path.isdir(subdir):
                for filename in os.listdir(subdir):
                    if filename.endswith('.json'):
                        json_path = os.path.join(subdir, filename)
                        try:
                            with open(json_path, 'r') as file:
                                param_dict = json.load(file)
                                if match_identifier(param_dict, feature_dict):
                                    return subdir
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON from file {json_path}: {e}")
                            continue
        return None


    subdir = None
    for key in range(0, len(identifier)):
        subdir = locate_subdirectory(testcase_dir, identifier[str(key)])
        if subdir is None:
            return None
        else:
            testcase_dir = subdir

    return subdir


def parse_params(result_dir):
    params_path = os.path.join(result_dir, "params.json")

    with open(params_path, "r") as f:
        params = json.load(f)

    return params

def load_routing_res(res_dir):
    res_dir = Path(res_dir)
    assert res_dir.is_dir(), f'{res_dir} is not a valid directory.'

    pkl_files = list(res_dir.glob('*.pkl'))
    assert len(pkl_files) == 1, f'No or multiple .pkl files found in the directory.'
    pkl_file_path = pkl_files[0]

    with open(pkl_file_path, "rb") as f:
        res = dill.load(f)

    return res


def str_to_dict(s):
    """
    Parse the string into a dictionary.
    The input string is expected to be in a format similar to {'key1': 'value1', 'key2': 'value2'}.
    """
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {e}")


def check_result_exists(res_dir, params_dict):
    """
    :param res_dir: The directory to store result directories
    :param params_dict: The hyperparameters dictionary to search.
    :return:
    """
    def convert_value(value):
        if callable(value):
            return value.__name__
        return value

    params_dict = {k: convert_value(v) for k, v in params_dict.items()}

    for subdir_name in os.listdir(res_dir):
        subdir = os.path.join(res_dir, subdir_name)
        if os.path.isdir(subdir):
            for filename in os.listdir(subdir):
                if filename.endswith('.json'):
                    json_path = os.path.join(subdir, filename)
                    with open(json_path, 'r') as file:
                        param_dict_to_comp = json.load(file)
                        if param_dict_to_comp == params_dict:
                            return True, subdir
    return False, None
