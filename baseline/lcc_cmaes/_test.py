#!/usr/bin/env python3
"""
LCC-CMAES Minimal Test Script
LCC-CMAES 最小测试脚本

This script tests the basic functionality of the standalone LCC-CMAES package.
此脚本测试独立 LCC-CMAES 包的基本功能。

Usage / 使用方法:
    python -m baseline.lcc_cmaes._test

Or run directly:
    或者直接运行:
    python baseline/lcc_cmaes/_test.py
"""

import os
import sys
import traceback

# =============================================================================
# Add package to path / 添加包到路径
# =============================================================================
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


# =============================================================================
# Test Functions / 测试函数
# =============================================================================

def test_imports():
    """Test that all modules can be imported. / 测试所有模块能否正常导入。"""
    print("[TEST 1] Testing imports...")
    try:
        from baseline.lcc_cmaes import LCC_CMAES, optimize_with_lcc_cmaes
        print("  ✓ Successfully imported LCC_CMAES")
        print("  ✓ Successfully imported optimize_with_lcc_cmaes")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_model_loading():
    """Test that the trained model can be loaded. / 测试训练好的模型能否加载。"""
    print("\n[TEST 2] Testing model loading...")
    try:
        from baseline.lcc_cmaes.env.agent.inference import InferenceAgent

        # Get model path / 获取模型路径
        model_path = os.path.join(_current_dir, 'model', 'epoch-9.pt')

        if not os.path.exists(model_path):
            print(f"  ✗ Model file not found: {model_path}")
            return False

        # Create a minimal options object / 创建最小配置对象
        class MinimalOptions:
            def __init__(self):
                self.device = 'cpu'  # Use CPU for testing / 使用 CPU 测试

        agent_opts = MinimalOptions()
        agent = InferenceAgent(agent_opts)

        # Try to load the model / 尝试加载模型
        agent.load(model_path)
        agent.eval()

        print(f"  ✓ Model loaded successfully from: {model_path}")
        print(f"  ✓ Model size: {os.path.getsize(model_path) / 1024:.1f} KB")
        print(f"  ✓ Actor network is on device: {agent.device}")
        return True
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test that the actor network can perform forward pass. / 测试 Actor 网络能否执行前向传播。"""
    print("\n[TEST 3] Testing forward pass...")
    try:
        import torch
        from baseline.lcc_cmaes.env.agent.inference import InferenceAgent

        model_path = os.path.join(_current_dir, 'model', 'epoch-9.pt')

        class MinimalOptions:
            def __init__(self):
                self.device = 'cpu'

        # Create agent and load model / 创建智能体并加载模型
        agent = InferenceAgent(MinimalOptions())
        agent.load(model_path)
        agent.eval()

        # Create a dummy state / 创建虚拟状态
        state_dim = 58
        state = torch.randn(1, state_dim)

        # Forward pass / 前向传播
        with torch.no_grad():
            action, logprob, value = agent.actor(state)

        print(f"  ✓ Forward pass successful")
        print(f"  ✓ Action shape: {action.shape}")
        print(f"  ✓ Action logits range: [{action.min():.2f}, {action.max():.2f}]")
        print(f"  ✓ Log prob: {logprob.item():.4f}")
        return True
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        traceback.print_exc()
        return False


def test_action_selection():
    """Test that action selection works correctly. / 测试动作选择是否正常工作。"""
    print("\n[TEST 4] Testing action selection...")
    try:
        import torch
        from baseline.lcc_cmaes.env.agent.inference import InferenceAgent

        model_path = os.path.join(_current_dir, 'model', 'epoch-9.pt')

        class MinimalOptions:
            def __init__(self):
                self.device = 'cpu'

        agent = InferenceAgent(MinimalOptions())
        agent.load(model_path)
        agent.eval()

        # Test with multiple random states / 测试多个随机状态
        state_dim = 58
        num_actions = 3  # MiVD, MaVD, RaVD

        action_counts = [0, 0, 0]
        num_trials = 10

        with torch.no_grad():
            for _ in range(num_trials):
                state = torch.randn(1, state_dim)
                action, _, _ = agent.actor(state)
                action_id = action.argmax(dim=-1).item()
                action_counts[action_id] += 1

        print(f"  ✓ Tested {num_trials} random states")
        print(f"  ✓ Action distribution: {action_counts}")
        print(f"  ✓ Actions selected: {[i for i, count in enumerate(action_counts) if count > 0]}")
        return True
    except Exception as e:
        print(f"  ✗ Action selection failed: {e}")
        traceback.print_exc()
        return False


def test_package_structure():
    """Test that the package structure is correct. / 测试包结构是否正确。"""
    print("\n[TEST 5] Testing package structure...")
    try:
        # Check key files and directories exist / 检查关键文件和目录是否存在
        required_items = [
            ('model/epoch-9.pt', 'Model file'),
            ('env/agent/inference.py', 'Inference agent'),
            ('env/agent/network/actor_network.py', 'Actor network'),
            ('env/optimizer/opt.py', 'CMAES optimizer'),
            ('utils/options.py', 'Configuration'),
            ('lcc_cmaes.py', 'Main module'),
        ]

        all_exist = True
        for path, description in required_items:
            full_path = os.path.join(_current_dir, path)
            if os.path.exists(full_path):
                print(f"  ✓ {description}: {path}")
            else:
                print(f"  ✗ Missing {description}: {path}")
                all_exist = False

        return all_exist
    except Exception as e:
        print(f"  ✗ Structure check failed: {e}")
        traceback.print_exc()
        return False


# =============================================================================
# Main Test Runner / 主测试运行器
# =============================================================================

def run_all_tests():
    """Run all tests and print summary. / 运行所有测试并打印摘要。"""
    print("=" * 70)
    print("LCC-CMAES Package Test Suite")
    print("LCC-CMAES 包测试套件")
    print("=" * 70)

    tests = [
        ("Import Test", test_imports),
        ("Model Loading Test", test_model_loading),
        ("Forward Pass Test", test_forward_pass),
        ("Action Selection Test", test_action_selection),
        ("Package Structure Test", test_package_structure),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Test crashed: {e}")
            traceback.print_exc()
            results.append((name, False))

    # Print summary / 打印摘要
    print("\n" + "=" * 70)
    print("Test Summary / 测试摘要")
    print("=" * 70)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}   {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print("-" * 70)
    print(f"  Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n  ✓ All tests passed! The package is ready to use.")
        print("  ✓ 所有测试通过！包已准备就绪。")
        return 0
    else:
        print(f"\n  ⚠ {total - passed} test(s) failed. Please check the errors above.")
        print(f"  ⚠ {total - passed} 个测试失败。请检查上述错误。")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
