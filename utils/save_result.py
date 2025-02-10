import os
import json
import dill
import datetime

def create_timestamped_folder(parent_folder):
    timestamp = datetime.datetime.now().strftime("run_%Y%m%d%H%M%S")
    timestamped_folder = os.path.join(parent_folder, timestamp)
    os.makedirs(timestamped_folder, exist_ok=False)
    return timestamped_folder


def save_experiment_params(params, folder):
    def convert_value(value):
        if callable(value):
            return value.__name__
        return value

    converted_params = {k: convert_value(v) for k, v in params.items()}
    params_path = os.path.join(folder, "params.json")
    with open(params_path, "w") as f:
        json.dump(converted_params, f, indent=4)


def save_experiment_results(results, params, parent_folder, res_name):
    """
    :param results:
    :param params:
    :param parent_folder:
    :param res_name:
    :return:
    """
    folder = create_timestamped_folder(parent_folder)
    save_experiment_params(params, folder)
    results_path = os.path.join(folder, res_name)
    with open(results_path, "wb") as f:
        dill.dump(results, f)

    print(f"Save experiment results to {folder}")
