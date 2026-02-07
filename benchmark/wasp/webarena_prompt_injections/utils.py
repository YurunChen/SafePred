# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import os
import copy
from typing import List


def load_prompt_injection_config(config_file_path: str) -> List[dict]:
    """
    Reads and parses the prompt injection config JSON file.
    Args:
        config_file_path (str): The path to the prompt injection config JSON file.
    Returns:
        List[dict]: A list of dictionaries representing the prompt injections.
    """
    with open(config_file_path, "r") as config_file:
        config_data = json.load(config_file)

    return config_data


def write_json(dict_object_to_write, path_to_write_to: str):
    with open(path_to_write_to, "w") as json_file:
        json.dump(dict_object_to_write, json_file, indent=4)


def write_json_with_task_ids_as_individual_files(
    list_of_dict_objects_to_write, path_to_write_to: str
):
    if type(list_of_dict_objects_to_write) != list:
        raise ValueError("This function is meant to write a list as individual files.")

    for dict_object in list_of_dict_objects_to_write:
        index = dict_object["task_id"]
        full_path_to_write_to = os.path.join(path_to_write_to, f"{index}.json")
        with open(full_path_to_write_to, "w") as json_file:
            json.dump(dict_object, json_file, indent=4)


def get_absolute_path_to_sibling_directory_with_name(sibling_dir_name: str):
    cwd = os.getcwd()
    sibling_directory = os.path.join(cwd, "..", sibling_dir_name)
    return os.path.abspath(sibling_directory)


def write_bash_script(path_to_script: str, content_of_script: str):
    with open(path_to_script, "w") as file:
        file.write(content_of_script)

    os.chmod(path_to_script, 0o755)  # rwxr-xr-x permissions


def mkdir_in_output_folder_and_return_absolute_path(output_dir: str, new_sub_dir: str):
    absolute_path_to_new_subdir = os.path.abspath(os.path.join(output_dir, new_sub_dir))
    os.makedirs(absolute_path_to_new_subdir, exist_ok=True)
    return absolute_path_to_new_subdir


def instantiate_dict_str_with_params(webarena_task_field: dict, params_dict: dict):

    def dict_dfs(D, params):
        if isinstance(D, (list, dict)):
            items = enumerate(D) if isinstance(D, list) else D.items()
            for k, v in items:
                if isinstance(v, str):
                    D[k] = v.format(**params)
                elif isinstance(v, (dict, list)):
                    dict_dfs(v, params)

    instantiated_webarena_task_field = copy.deepcopy(webarena_task_field)
    dict_dfs(instantiated_webarena_task_field, params_dict)

    return instantiated_webarena_task_field