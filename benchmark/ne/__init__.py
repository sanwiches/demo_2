"""NeuroEvolution Benchmark Suite / 神经进化基准测试套件

Contains benchmarks using different physics engines:
- Brax: GPU-accelerated physics engine
- MuJoCo: Industry-standard physics simulator
"""

__all__ = ['BraxBenchmark', 'MuJoCoBenchmark']


def __getattr__(name: str):
    if name == "BraxBenchmark":
        from .brax.brax_benchmarks import Benchmark as _Benchmark
        return _Benchmark
    if name == "MuJoCoBenchmark":
        from .mujoco.mujoco_benchmarks import Benchmark as _Benchmark
        return _Benchmark
    raise AttributeError(f"module 'benchmark.ne' has no attribute {name!r}")
