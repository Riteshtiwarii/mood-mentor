"""
Tests for Stress & Performance Benchmarks (Task 7)
"""

import pytest
from benchmarks.stress_test import run_performance_and_stress_benchmark

def test_stress_concurrency_and_latency():
    # Run smaller benchmark for automated CI test suite
    results = run_performance_and_stress_benchmark(num_requests=30, max_workers=6)
    
    assert results["errors"] == 0, f"Benchmark encountered {results['errors']} errors"
    assert results["throughput"] > 15.0, f"Throughput {results['throughput']} too low"
    assert results["p95"] < 150.0, f"P95 latency {results['p95']} ms exceeded 150ms limit"
    assert results["accuracy"] >= 0.80, f"Classification accuracy {results['accuracy']} below 80%"
