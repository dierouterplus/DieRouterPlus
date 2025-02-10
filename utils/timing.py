import time

def timeit(func_name):
    """
    A decorator to measure the execution time of a function.
    :param func_name: The name of the function being measured.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Function {func_name} executed in {elapsed_time:.4f} seconds")
            return result
        return wrapper
    return decorator
