import re

def read_log_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        log_data = file.read()  # 读取整个文件的内容
    return log_data

# 调用函数，读取 log 文件
log_data = read_log_file("../testcase_1_10_M_Pin_Baseline")
# log_data = read_log_file("../testcase_1_10_S_Pin_Max_2")
# log_data = read_log_file("../testcase_1_10_M_Pin_Max_2")
# 定义正则表达式用于提取信息
patterns = {
    'testcase_name': r"Processing (testcase\d+)",
    'hybrid_initial_routing_time': r"Hybrid Initial Routing Finished in:(\d+(?:\.\d*(?:e[+-]?\d+)?)?)s",
    'system_delay_after_hybrid': r"System delay after Hybrid Initial Routing is (\d+(?:\.\d*(?:e[+-]?\d+)?)?)",
    'violation_rip_up_time': r"Violation Rip Up and Reroute took (\d+(?:\.\d*(?:e[+-]?\d+)?)?) seconds",
    'system_delay_after_violations': r"System delay after eliminating the violations is (\d+(?:\.\d*(?:e[+-]?\d+)?)?)",
    'performance_rip_up_time': r"Performance-Driven Rip Up and Reroute took (\d+(?:\.\d*(?:e[+-]?\d+)?)?) seconds",
    'opt_system_delay': r"Opt system delay is (\d+(?:\.\d*(?:e[+-]?\d+)?)?)",
    'function_solve_time': r"Function Solve executed in (\d+(?:\.\d*(?:e[+-]?\d+)?)?) seconds",
    'system_delay_after_continuous_optimization': r"System delay after Continuous Optimization is (\d+(?:\.\d*(?:e[+-]?\d+)?)?)",
    'legalization_delay': r"After legalization, System delay is (\d+(?:\.\d*(?:e[+-]?\d+)?)?)",
    'dp_time': r"Function perform_dp_legal executed in (\d+(?:\.\d*(?:e[+-]?\d+)?)?) seconds"
}

# 存储所有提取的数据
testcase_data = []

# 使用正则表达式提取信息
for testcase_match in re.finditer(r"Processing (testcase\d+)(.*?Finish \1)", log_data, re.DOTALL):
    testcase_info = {}
    testcase_name = testcase_match.group(1)
    testcase_info['testcase_name'] = testcase_name

    # 提取每一项信息
    for key, pattern in patterns.items():
        if key == 'testcase_name':
            continue
        match = re.search(pattern, testcase_match.group(2))
        if match:
            testcase_info[key] = float(match.group(1))
        else:
            testcase_info[key] = None

    testcase_data.append(testcase_info)

# 输出结果
# 第一部分: 输出关于系统延迟的信息
print("Testcase, Hybrid Initial Routing, Violation Driven R&R, Perf Driven R&R, Conti Sol, Legalized")
for data in testcase_data:
    print(f"{data['testcase_name']}, "
          f"{data['system_delay_after_hybrid']:.2f}, "
          f"{data['system_delay_after_violations']:.2f}, "
          f"{data['opt_system_delay']:.2f}, "
          f"{data['system_delay_after_continuous_optimization']:.2f}, "
          f"{data['legalization_delay']:.2f}")

# 第二部分: 输出执行时间的信息

print("\nTestcase, Hybrid Initial Routing, Violation Driven R&R, Perf Driven R&R, Conti Sol, Legalized")
for data in testcase_data:
    total_execution_time = (data['hybrid_initial_routing_time'] or 0) + (data['violation_rip_up_time'] or 0) + \
                            (data['performance_rip_up_time'] or 0) + (data['function_solve_time'] or 0) + \
                            (data['dp_time'] or 0)
    print(f"{data['testcase_name']}, {data['hybrid_initial_routing_time']:.3f}, {data['violation_rip_up_time']:.3f}, {data['performance_rip_up_time']:.3f}, {data['function_solve_time']:.3f}, {data['dp_time']:.3f}")
    print(f"Total execution time: {total_execution_time:.2f} seconds\n")
