from benchmark.ne.mujoco.mujoco_benchmarks import Benchmark
import numpy as np
import time
# from baseline.mmes.mmes import MMES 
from baseline.lcc_cmaes.lcc_cmaes import LCC_CMAES
# from baseline.wcc_mmes.wcc_mmes import WCC_MMES
from utils_ import fun_record, log_to_tensorboard, result_record, run_parallel_task, plot_evaluation_curve_best_so_far, running_data_record
import jax
# 启用 JAX 编译缓存
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_log_compiles", True)
'''
此为 NDAs 优化实验示例，使用 MMES 优化器在 CEC2013 LSGO 上进行测试
'''

# parallel_index 一定要放在最后，否则会报错，因为 run_parallel_task 传参是按位置传递的
def nda_optimization_task(fun_id, MaxFEs, parallel_index):

    # region：为什么在这里初始化 fun 而不是从外部传入？ ==================================
    # =================================================================================
    # 1. 避免 Pickling (序列化) 错误：
    #    许多底层由 C/C++ 扩展编写的基准测试函数对象 (如 CEC2013/2017) 包含了指针或
    #    复杂的内部状态，这些对象通常不支持 Python 的 pickle 协议，会导致
    #    "TypeError: can't pickle SwigPyObject objects" 或类似崩溃。
    #
    # 2. 降低多进程通信开销 (IPC Overhead)：
    #    即使对象支持序列化，这些函数对象往往包含巨大的数据结构 (如 1000维的旋转矩阵
    #    和偏移向量，大小可达数 MB)。如果在主进程初始化并传递给子进程，每次任务启动
    #    都需要通过 IPC 管道传输这些大数据，会造成极大的性能损耗和启动延迟。
    #
    # 3. 内存安全性：
    #    通过只传递轻量级的 fun_id (int)，让每个子进程在本地内存空间重新加载或生成
    #    函数对象，既安全又高效 (利用操作系统 Copy-on-Write 机制或快速的本地初始化)。
    # =================================================================================
    # endregion =======================================================================

    time_start = time.time()

    bench = Benchmark()
    fun = bench.get_function(fun_id)
    info = bench.get_info(fun_id)

    fun_ = fun_record(fun)

    problem_ = {'fitness_function': fun_,  # fitness function
    'ndim_problem': info['dimension'],  # dimension
    'lower_boundary': info['lower'] * np.ones((info['dimension'],)),  # lower search boundary
    'upper_boundary': info["upper"]* np.ones((info['dimension'],))}

    options_ = {'max_function_evaluations': MaxFEs,  # to set optimizer options
        'mean': np.zeros(info['dimension']),
        'sigma': 0.005,
        'is_restart': False,
        'verbose': True,  # 启用详细输出 / Enable verbose output
        'seed_rng': 42+ parallel_index,
        'fitness_threshold': -np.inf,  # 明确设置避免意外终止 / Explicitly set to avoid accidental termination
        'early_stopping_evaluations': np.inf,# 禁用早停 / Disable early stopping
        }
    optimizer =LCC_CMAES(problem_, options_)
    results_ = optimizer.optimize()

    # # 打印终止信号 / Print termination signal
    # print(f"[进程/Process {parallel_index}] 终止信号/Termination: {optimizer.termination_signal}")
    # print(f"[进程/Process {parallel_index}] 评估次数/Evaluations: {optimizer.n_function_evaluations}/{MaxFEs}")

    time_end = time.time()

    return results_["fitness_record"], (time_end - time_start)


if __name__ == '__main__':
    for fun_id in [36,40,47,49]:
        MaxFEs = 4E3
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        logs_dir = f'repository/logs/baseline/lcc_cmaes/mujoco/maxfes_{MaxFEs:.0E}'.replace('+0', '') + f'/f{fun_id}/{timestamp}'
        output_path = f'repository/data/baseline/lcc_cmaes/mujoco/maxfes_{MaxFEs:.0E}'.replace('+0', '') + f'/f{fun_id}/{timestamp}'

        fitness_record, average_time = run_parallel_task(nda_optimization_task, parallel_num=1, fun_id=fun_id, MaxFEs=MaxFEs)#把parallel_num从5变为1，防止死锁现象

        output_data = {'baseline':[[]], 'baseline_time':[]}
        output_data['baseline'] = fitness_record
        output_data['baseline_time'].append(average_time)

        # log_to_tensorboard(fitness_record, logs_dir) # 这里默认采样间隔为 1E3
        result_record(output_data, output_path, record_FEs_list=[1E3, 2E3, 3E3, 4E3])

        # 可以直接用 utils 里的绘图函数画图
        # plot_evaluation_curve_best_so_far(output_data, output_path.replace('data', 'plot'), maxfes=MaxFEs)

        # running_data_record(output_data, output_path)

        print(f'Function {fun_id} completed. Average Time: {average_time:.2f} seconds.')