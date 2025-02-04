"""
AutoMark is a lightweight tool for testing programming assignments
 
Copyright (C) 2020 Ivan Sosnovik

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

----------------------------------------------------------------------

This script runs the client script for AutoMark
There are 2 main functions the end-user should use:
* get_progress(username) --- just to get the current progress to the stdout
* test_student_function(username, function, arg_keys) --- to test the provided function 
    and to print the result / error to the stdout

This scripts automatically downloads local tests into the `local_tests` folder
Compatible with Python 2/3
"""

from __future__ import print_function

import inspect
import math
import os
import sys
import json
from enum import Enum, auto

import duckdb
import numpy
import pandas
import requests
import shutil
import hashlib
import pyspark

from pyspark.sql import SparkSession

try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = OSError


class ServerError(BaseException):
    pass


class Config:
    server_url = 'http://big-data-grading-2025.westeurope.cloudapp.azure.com:8080'
    cwd = os.path.dirname(os.path.realpath(__file__))
    test_folder = os.path.join(cwd, 'local_tests')
    test_path = os.path.join(test_folder, 'tests.pickle')
    username = ''


class TaskType(Enum):
    STANDARD = auto()
    DUCKDB_SQL = auto()
    MAP_REDUCE = auto()
    PYSPARK_RDD = auto()
    PYSPARK_DF = auto()

    def __str__(self):
        return self.name


def serialize_test_data(data):
    """
    Barrie: This function converts complex objects to be compatible with the automark framework for "remote" data.
    :param data: that we want to be processed by automark.
    :return:
    """
    if isinstance(data, numpy.ndarray):
        return {"data": data.tolist(), "type": "numpy.ndarray"}
    elif isinstance(data, pandas.DataFrame):
        return {"data": data.to_dict(), "type": "pandas.DataFrame"}
    elif isinstance(data, pyspark.RDD):
        num_partitions = data.getNumPartitions()
        return {'data': data.collect(), 'type': "pyspark.RDD", 'num_partitions': num_partitions}
    elif isinstance(data, pyspark.sql.DataFrame):
        num_partitions = data.rdd.getNumPartitions()
        return {'data': data.toJSON().collect(), 'type': "pyspark.DataFrame", 'num_partitions': num_partitions}
    else:
        return {"data": data, "type": type(data).__name__}


def deserialize_test_data(data):
    """
    Barrie: This function converts complex objects to be compatible with the automark framework for "remote" data.
    :param data:
    :return:
    """
    arg_value = data['data']
    if data['type'] == "numpy.ndarray":
        new_arg_value = numpy.array(arg_value)
    elif data['type'] == "pandas.DataFrame":
        new_arg_value = pandas.DataFrame.from_dict(arg_value)
    elif data['type'] == "pyspark.RDD":
        spark_session = SparkSession.builder \
            .master("local") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .getOrCreate()
        new_arg_value = spark_session.sparkContext.parallelize(arg_value, data['num_partitions'])
    elif data['type'] == "pyspark.DataFrame":
        spark_session = SparkSession.builder \
            .master("local") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .getOrCreate()
        new_arg_value = spark_session.read.json(spark_session.sparkContext.parallelize(arg_value, data['num_partitions']))
    else:
        new_arg_value = arg_value
    return new_arg_value


class TaskTypeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'TaskType':
            return TaskType
        return super().find_class(module, name)


def execute_duckdb_sql(query, **dfs):
    con = duckdb.connect(database=":memory:", read_only=False)
    for df_name, df_data in dfs.items():
        con.register(df_name, df_data)
    result = con.execute(query).fetchdf()
    return result


def mapreduce(input_partitions, f_map, f_reduce, num_reducers=2, print_debug_text=False):
    if print_debug_text: print("---Starting MAP-phase.---")
    # These arrays will hold the outputs of the map-phase for each input partition
    map_output_partitions = []
    # We are running the map-phase on each input partition now. In a real mapreduce system, this would run
    # in parallel on different machines.
    for counter, input_partition in enumerate(input_partitions):
        if print_debug_text: print(f"  Applying f_map to input partition {counter}")
        map_output = []
        # We apply f_map to each (key, value) pair in the input partition, and store the corresponding outputs
        for key, value in input_partition:
            if print_debug_text: print(f"    f_map({key}, {value}) -> ")
            for mapped_key, mapped_value in f_map(key, value):
                if print_debug_text: print(f"      ({mapped_key}, {mapped_value})")
                map_output.append((mapped_key, mapped_value))
        # Store output partition of this map operation
        map_output_partitions.append(map_output)

        # Next, we start the shuffle phase. We need to create several reducer partitions from the map outputs,
    # assign each key to a partition and collect all the values for this key.
    if print_debug_text: print("\n---Starting SHUFFLE-phase.---")
    reduce_input_partitions = []
    # We create as many reducer partitions as specified by num_reducers
    for _ in range(0, num_reducers):
        reduce_input_partitions.append(dict())

    # We shuffle each map output partition now. In a real mapreduce system, this would run
    # in parallel on different machines.
    for counter, map_output_partition in enumerate(map_output_partitions):
        if print_debug_text: print(f"  Shuffling map output input partition {counter}")
        # We process each (key, value) pair from the map-output here
        for key, value in map_output_partition:
            # We determine the target partition for this key via hash-partitioning
            target_partition_index = abs(hash(key)) % num_reducers
            if print_debug_text: print(
                f"    Assigning key [{key}] to reducer input {target_partition_index} via hash-partitioning")
            # We add the value to the group of the key in the target partition
            target_partition = reduce_input_partitions[target_partition_index]
            if not key in target_partition:
                target_partition[key] = []
            target_partition[key].append(value)

    # Next, we run the reduce-phase
    if print_debug_text: print("\n---Starting REDUCE-phase.---")
    reduce_output_partitions = []

    # We reduce each reduce partition now. In a real mapreduce system, this would run
    # in parallel on different machines.
    for counter, reduce_input_partition in enumerate(reduce_input_partitions):
        reduce_output = []
        if print_debug_text: print(f"  Applying f_reduce to reduce_input partition {counter}")
        # We apply f_reduce to each key and its corresponding values now and collect the results
        for key, values in reduce_input_partition.items():
            if print_debug_text: print(f"    f_reduce({key}, {values}) -> ")
            for reduced_key, reduced_value in f_reduce(key, values):
                if print_debug_text: print(f"      ({reduced_key}, {reduced_value})")
                reduce_output.append((reduced_key, reduced_value))
        reduce_output_partitions.append(reduce_output)

    return reduce_output_partitions


# MAIN FUNCTIONS
def configure(server_url='', username=''):
    if server_url:
        Config.server_url = server_url
    if username:
        Config.username = username


def get_progress():
    """Print the current progress to stdout"""
    endpoint = f'{Config.server_url}/get_progress/{Config.username}'
    response = requests.get(endpoint)
    data = response.json()

    if 'error' in data:
        raise ServerError(data['error'])

    print('-' * 70)
    print('| {:67}|'.format(data['name']))
    print('-' * 70)
    for k, v in data['progress'].items():
        print('| {:50}| {:15}|'.format(k, v))
    print('-' * 70)


def test_student_function(function):
    """Test the provided function and print the result / error to stdout
    # Args:
        function - a function as an object. callable
        arg_keys - a list of the function's arguments as srings. 
            Example: `['arg1', 'arg2']`
    """
    if not _local_tests_are_valid():
        _remove_local_tests()
        print('Downloading local tests...')
        _load_local_tests(Config.username)

    print('Running local tests...')
    if not _passed_local_tests(function):
        print('{} failed some local tests'.format(function.__name__))
        return

    print('{} successfully passed local tests'.format(function.__name__))

    print('Running remote test...')
    sys.stdout.flush()
    if not _passed_remote_test(Config.username, function):
        print("Test failed. Please review your code.")
        return

    print("Test was successful. Congratulations!")


# UTILITY FUNCTIONS
# Local tests
def _remove_local_tests():
    try:
        os.remove(Config.test_path)
        print('The current version of local tests is outdated. The local tests are removed.')
        sys.stdout.flush()
    except FileNotFoundError:
        pass


def _load_local_tests(username):
    if not os.path.exists(Config.test_folder):
        os.makedirs(Config.test_folder)

    try:
        endpoint = f'{Config.server_url}/load_tests/{username}'
        stream = requests.get(endpoint, stream=True)
        if stream.status_code == 200:
            with open(Config.test_path, 'wb') as f:
                stream.raw.decode_content = True
                shutil.copyfileobj(stream.raw, f)
            print('Local tests are downloaded.')
            sys.stdout.flush()
        else:
            error = stream.json()['error']
            raise ServerError(error)
    except ServerError as e:
        raise e
    except:
        raise ServerError('Error downloading local tests.')


def _local_tests_are_valid():
    try:
        hash_md5 = hashlib.md5()
        with open(Config.test_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        local_md5 = hash_md5.hexdigest()
        endpoint = f'{Config.server_url}/check_sum/{local_md5}'
        response = requests.get(endpoint).json()
        return response['success']
    except:
        return False


def _passed_local_tests(function):
    arg_keys = inspect.getfullargspec(function).args
    with open(Config.test_path, 'rb') as f:
        test_data = TaskTypeUnpickler(f).load()

    data = test_data[function.__name__]
    task_type = data['task_type']
    inputs = data['inputs']
    if task_type in {TaskType.PYSPARK_RDD, TaskType.PYSPARK_DF}:
        for some_list_inputs in inputs:
            for key, value in some_list_inputs.items():
                some_list_inputs[key] = deserialize_test_data(value)
    outputs = data['outputs']

    for in_, out_ in zip(inputs, outputs):
        kwargs_ = {k: in_[k] for k in arg_keys}
        if task_type is TaskType.DUCKDB_SQL:
            answer = execute_duckdb_sql(function(), **in_)
        elif task_type is TaskType.MAP_REDUCE:
            input_partitions = in_['input_partitions']
            answer = mapreduce(input_partitions, *function())
        else:
            answer = function(**kwargs_)

        if isinstance(out_, float):
            answer_is_correct = numpy.allclose(answer, out_, rtol=1e-5, atol=1e-5)
        elif isinstance(out_, pandas.DataFrame):
            answer_is_correct = (
                len(out_) == len(answer) and 
                len(pandas.concat([out_, answer]).drop_duplicates(keep=False)) == 0
            )
        elif isinstance(out_, numpy.ndarray):
            answer_is_correct = numpy.allclose(out_, answer, rtol=1e-5, atol=1e-5)
        elif task_type is TaskType.MAP_REDUCE:  # Mapreduce order may be switched
            answer_is_correct = mapreduce_results_equal(out_, answer)
        elif task_type is TaskType.PYSPARK_RDD:
            answer_is_correct = pyspark_rdd_results_equal(out_, serialize_test_data(answer))
        elif task_type is TaskType.PYSPARK_DF:
            answer_is_correct = pyspark_df_results_equal(out_, serialize_test_data(answer))
        else:
            raise Exception("Unknown data type:", type(out_))

        if not answer_is_correct:
            return False
    return True


# Remote tests
def _passed_remote_test(username, function):
    endpoint = f'{Config.server_url}/get_test_input/{username}/{function.__name__}'
    response = requests.get(endpoint)
    data = response.json()

    if 'error' in data:
        print("\x1b[31mERROR:", data['error'], "\x1b[0m")
        return False

    arg_keys = inspect.getfullargspec(function).args
    args = []
    if len(arg_keys) != 0:
        # Argument order is important here
        for arg_key in arg_keys:
            arg_value = deserialize_test_data(data['input'][arg_key])
            args.append(arg_value)
    else:
        # We have a task like the duckdb tasks, where we only have indirect kwargs we pass to some execute function
        for input_name, input_value in data['input'].items():
            data['input'][input_name] = deserialize_test_data(input_value)

    if TaskType[data['task_type']] is TaskType.DUCKDB_SQL:
        test_result = execute_duckdb_sql(function(), **data['input'])
    elif TaskType[data['task_type']] is TaskType.MAP_REDUCE:
        input_partitions = data['input']['input_partitions']
        test_result = mapreduce(input_partitions, *function())
    else:
        test_result = function(*args)

    answer = json.dumps(serialize_test_data(test_result))

    endpoint = f'{Config.server_url}/check_answer/{username}/{function.__name__}/{data["ipd"]}/{answer}'
    response = requests.get(endpoint)
    if not response.status_code == 200:
        assert not response.ok
        raise ServerError('Internal Error Occurred')
    assert response.ok
    answer_response = response.json()

    return answer_response['success']


# Utils
def mapreduce_results_equal(expected_mapreduce_result, mapreduce_result):
    results_by_key = {}
    for partition in mapreduce_result:
        for key, value in partition:
            results_by_key[key] = value

    expected_keys_and_values = {}
    for partition in expected_mapreduce_result:
        for key, value in partition:
            expected_keys_and_values[key] = value

    if len(results_by_key) != len(expected_keys_and_values):
        return False

    for key_of_interest, expected_value in expected_keys_and_values.items():
        if key_of_interest not in results_by_key:
            return False
        else:
            observed_value = results_by_key[key_of_interest]
            if isinstance(expected_value, (float, int)):
                if not math.isclose(expected_value, observed_value, rel_tol=1e-5, abs_tol=1e-5):
                    return False
            else:
                if expected_value != observed_value:
                    return False
    return True


def pyspark_rdd_results_equal(expected_result, actual_result):
    if expected_result['num_partitions'] is not actual_result['num_partitions']:
        return False

    expected_result_data = expected_result['data']
    expected_result_data.sort(key=lambda x: x[0])
    actual_result_data = actual_result['data']
    actual_result_data.sort(key=lambda x: x[0])
    if len(expected_result_data) != len(actual_result_data):
        # print(f"  ERROR: Expected {len(expected_result_data)} keys in the output, but found {len(actual_result_data)}!")
        return False
    for left, right in zip(expected_result_data, actual_result_data):
        left_key, left_value = left
        right_key, right_value = right
        if left_key != right_key:
            # print(f"  ERROR: Unable to find key [{left_key}] in result!")
            # print(f"  ERROR: Found [{right_key}] in result instead!")
            return False
        if isinstance(left_value, (float, int)):
            if not math.isclose(left_value, right_value, rel_tol=1e-5, abs_tol=1e-5):
                # print(f"  ERROR: Key [{right_key}] has unexpected value [{right_value}]!")
                return False
        else:
            if left_value != right_value:
                # print(f"  ERROR: Key [{right_key}] has unexpected value [{right_value}]!")
                return False
    return True


def pyspark_df_results_equal(expected_result, actual_result):
    if expected_result['num_partitions'] is not actual_result['num_partitions']:
        return False
    expected_result_data = expected_result['data']
    expected_result_data.sort(key=lambda x: x[0])

    actual_result_data = actual_result['data']
    actual_result_data.sort(key=lambda x: x[0])

    if len(expected_result_data) != len(actual_result_data):
        return False
    for left, right in zip(expected_result_data, actual_result_data):
        if len(left) != len(right):
            return False
        for left_col_value, right_col_value in zip(left, right):
            if isinstance(left_col_value, (float, int)):
                if not math.isclose(left_col_value, right_col_value, rel_tol=1e-5, abs_tol=1e-5):
                    return False
            else:
                if left_col_value != right_col_value:
                    return False
    return True
