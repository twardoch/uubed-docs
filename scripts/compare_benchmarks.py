#!/usr/bin/env python3
"""
Cross-Language Benchmark Comparison Script for uubed Project

This script compares performance benchmarks between Rust and Python implementations
of the uubed encoding libraries, generating comprehensive reports and visualizations.
"""

import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys

def parse_rust_benchmarks(file_path: str) -> Dict:
    """Parse Rust criterion benchmark results."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        benchmarks = {}
        for test_name, test_data in data.items():
            if 'mean' in test_data:
                benchmarks[test_name] = {
                    'mean': test_data['mean']['estimate'],
                    'std_dev': test_data['std_dev']['estimate'],
                    'throughput': test_data.get('throughput', {}),
                    'unit': 'ns'  # Criterion uses nanoseconds
                }
        
        return benchmarks
    except Exception as e:
        print(f"Error parsing Rust benchmarks: {e}")
        return {}

def parse_python_benchmarks(file_path: str) -> Dict:
    """Parse Python pytest-benchmark results."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        benchmarks = {}
        for test_data in data.get('benchmarks', []):
            name = test_data['name']
            stats = test_data['stats']
            
            benchmarks[name] = {
                'mean': stats['mean'],
                'std_dev': stats['stddev'],
                'min': stats['min'],
                'max': stats['max'],
                'unit': 's'  # pytest-benchmark uses seconds
            }
        
        return benchmarks
    except Exception as e:
        print(f"Error parsing Python benchmarks: {e}")
        return {}

def normalize_benchmark_names(rust_benchmarks: Dict, python_benchmarks: Dict) -> Tuple[Dict, Dict]:
    """Normalize benchmark names for comparison."""
    # Mapping between Rust and Python benchmark names
    name_mapping = {
        'encode_q64': 'test_q64_encode',
        'decode_q64': 'test_q64_decode',
        'encode_mq64': 'test_mq64_encode',
        'decode_mq64': 'test_mq64_decode',
        'encode_eq64': 'test_eq64_encode',
        'decode_eq64': 'test_eq64_decode',
        'encode_shq64': 'test_shq64_encode',
        'decode_shq64': 'test_shq64_decode',
        'encode_t8q64': 'test_t8q64_encode',
        'decode_t8q64': 'test_t8q64_decode',
        'encode_zoq64': 'test_zoq64_encode',
        'decode_zoq64': 'test_zoq64_decode',
    }
    
    # Normalize Rust benchmark names
    normalized_rust = {}
    for rust_name, rust_data in rust_benchmarks.items():
        for canonical_name, python_name in name_mapping.items():
            if canonical_name in rust_name.lower():
                normalized_rust[canonical_name] = rust_data
                break
    
    # Normalize Python benchmark names
    normalized_python = {}
    for python_name, python_data in python_benchmarks.items():
        for canonical_name, mapped_python_name in name_mapping.items():
            if mapped_python_name in python_name.lower():
                normalized_python[canonical_name] = python_data
                break
    
    return normalized_rust, normalized_python

def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between time units."""
    conversion_factors = {
        'ns': 1e-9,  # nanoseconds to seconds
        's': 1,      # seconds to seconds
        'ms': 1e-3,  # milliseconds to seconds
        'us': 1e-6,  # microseconds to seconds
    }
    
    if from_unit not in conversion_factors or to_unit not in conversion_factors:
        return value
    
    # Convert to seconds first, then to target unit
    seconds = value * conversion_factors[from_unit]
    return seconds / conversion_factors[to_unit]

def calculate_performance_ratios(rust_benchmarks: Dict, python_benchmarks: Dict) -> Dict:
    """Calculate performance ratios between Rust and Python implementations."""
    ratios = {}
    
    for benchmark_name in set(rust_benchmarks.keys()) & set(python_benchmarks.keys()):
        rust_data = rust_benchmarks[benchmark_name]
        python_data = python_benchmarks[benchmark_name]
        
        # Convert both to seconds for comparison
        rust_mean_seconds = convert_units(rust_data['mean'], rust_data['unit'], 's')
        python_mean_seconds = convert_units(python_data['mean'], python_data['unit'], 's')
        
        # Calculate ratio (Python time / Rust time)
        # Higher ratio means Rust is faster
        ratio = python_mean_seconds / rust_mean_seconds if rust_mean_seconds > 0 else float('inf')
        
        ratios[benchmark_name] = {
            'ratio': ratio,
            'rust_mean': rust_mean_seconds,
            'python_mean': python_mean_seconds,
            'speedup': f"{ratio:.2f}x faster" if ratio > 1 else f"{1/ratio:.2f}x slower"
        }
    
    return ratios

def generate_comparison_chart(ratios: Dict, output_path: str):
    """Generate performance comparison chart."""
    if not ratios:
        print("No benchmark data to visualize")
        return
    
    # Prepare data for plotting
    benchmark_names = list(ratios.keys())
    ratio_values = [ratios[name]['ratio'] for name in benchmark_names]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create bar chart
    bars = plt.bar(range(len(benchmark_names)), ratio_values, 
                   color=['green' if r > 1 else 'red' for r in ratio_values])
    
    # Customize the chart
    plt.xlabel('Benchmark')
    plt.ylabel('Performance Ratio (Python Time / Rust Time)')
    plt.title('Cross-Language Performance Comparison\n(Higher = Rust Faster)')
    plt.xticks(range(len(benchmark_names)), benchmark_names, rotation=45, ha='right')
    
    # Add horizontal line at y=1 for reference
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for i, (bar, ratio) in enumerate(zip(bars, ratio_values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.2f}x',
                ha='center', va='bottom' if height > 0 else 'top')
    
    # Add legend
    plt.legend(['Reference (Equal Performance)', 'Rust Faster', 'Python Faster'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_detailed_report(rust_benchmarks: Dict, python_benchmarks: Dict, 
                           ratios: Dict, output_path: str):
    """Generate detailed markdown report."""
    report = []
    
    # Header
    report.append("# Cross-Language Performance Comparison Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    report.append("")
    
    # Summary
    report.append("## Summary")
    report.append("")
    
    if ratios:
        avg_ratio = sum(ratios[name]['ratio'] for name in ratios) / len(ratios)
        rust_faster_count = sum(1 for name in ratios if ratios[name]['ratio'] > 1)
        python_faster_count = len(ratios) - rust_faster_count
        
        report.append(f"- **Total Benchmarks Compared**: {len(ratios)}")
        report.append(f"- **Average Performance Ratio**: {avg_ratio:.2f}x")
        report.append(f"- **Rust Faster**: {rust_faster_count} benchmarks")
        report.append(f"- **Python Faster**: {python_faster_count} benchmarks")
        report.append("")
        
        if avg_ratio > 1:
            report.append(f"**Overall**: Rust implementation is {avg_ratio:.2f}x faster on average")
        else:
            report.append(f"**Overall**: Python implementation is {1/avg_ratio:.2f}x faster on average")
    else:
        report.append("- **No comparable benchmarks found**")
    
    report.append("")
    
    # Detailed Results
    report.append("## Detailed Results")
    report.append("")
    
    if ratios:
        report.append("| Benchmark | Rust (seconds) | Python (seconds) | Ratio | Performance |")
        report.append("|-----------|----------------|------------------|-------|-------------|")
        
        for name in sorted(ratios.keys()):
            data = ratios[name]
            rust_time = f"{data['rust_mean']:.6f}"
            python_time = f"{data['python_mean']:.6f}"
            ratio = f"{data['ratio']:.2f}"
            performance = data['speedup']
            
            report.append(f"| {name} | {rust_time} | {python_time} | {ratio} | {performance} |")
    else:
        report.append("No comparable benchmarks found between Rust and Python implementations.")
    
    report.append("")
    
    # Rust-only benchmarks
    rust_only = set(rust_benchmarks.keys()) - set(python_benchmarks.keys())
    if rust_only:
        report.append("## Rust-Only Benchmarks")
        report.append("")
        report.append("| Benchmark | Mean Time | Unit |")
        report.append("|-----------|-----------|------|")
        
        for name in sorted(rust_only):
            data = rust_benchmarks[name]
            mean_time = f"{data['mean']:.6f}"
            unit = data['unit']
            report.append(f"| {name} | {mean_time} | {unit} |")
        
        report.append("")
    
    # Python-only benchmarks
    python_only = set(python_benchmarks.keys()) - set(rust_benchmarks.keys())
    if python_only:
        report.append("## Python-Only Benchmarks")
        report.append("")
        report.append("| Benchmark | Mean Time | Unit |")
        report.append("|-----------|-----------|------|")
        
        for name in sorted(python_only):
            data = python_benchmarks[name]
            mean_time = f"{data['mean']:.6f}"
            unit = data['unit']
            report.append(f"| {name} | {mean_time} | {unit} |")
        
        report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    if ratios:
        slow_benchmarks = [name for name in ratios if ratios[name]['ratio'] < 0.5]
        if slow_benchmarks:
            report.append("### Performance Concerns")
            report.append("")
            report.append("The following benchmarks show concerning performance characteristics:")
            report.append("")
            for name in slow_benchmarks:
                ratio = ratios[name]['ratio']
                report.append(f"- **{name}**: Python is {1/ratio:.2f}x faster than Rust")
            report.append("")
            report.append("Consider investigating these implementations for potential optimizations.")
            report.append("")
        
        fast_benchmarks = [name for name in ratios if ratios[name]['ratio'] > 10]
        if fast_benchmarks:
            report.append("### Outstanding Performance")
            report.append("")
            report.append("The following benchmarks show excellent Rust performance:")
            report.append("")
            for name in fast_benchmarks:
                ratio = ratios[name]['ratio']
                report.append(f"- **{name}**: Rust is {ratio:.2f}x faster than Python")
            report.append("")
    
    # Footer
    report.append("## Methodology")
    report.append("")
    report.append("- **Rust Benchmarks**: Generated using `criterion` crate")
    report.append("- **Python Benchmarks**: Generated using `pytest-benchmark`")
    report.append("- **Comparison Method**: Direct timing comparison with unit normalization")
    report.append("- **Performance Ratio**: Python execution time divided by Rust execution time")
    report.append("- **Higher ratio indicates better Rust performance**")
    report.append("")
    
    # Write report to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

def main():
    """Main function to orchestrate benchmark comparison."""
    parser = argparse.ArgumentParser(description='Compare Rust and Python benchmark results')
    parser.add_argument('--rust', required=True, help='Path to Rust benchmark results JSON')
    parser.add_argument('--python', required=True, help='Path to Python benchmark results JSON')
    parser.add_argument('--output', required=True, help='Output path for comparison report')
    parser.add_argument('--chart', help='Output path for performance chart')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Parse benchmark results
    if args.verbose:
        print("Parsing Rust benchmark results...")
    rust_benchmarks = parse_rust_benchmarks(args.rust)
    
    if args.verbose:
        print("Parsing Python benchmark results...")
    python_benchmarks = parse_python_benchmarks(args.python)
    
    if not rust_benchmarks and not python_benchmarks:
        print("Error: No benchmark data found in either file")
        sys.exit(1)
    
    # Normalize benchmark names
    if args.verbose:
        print("Normalizing benchmark names...")
    normalized_rust, normalized_python = normalize_benchmark_names(rust_benchmarks, python_benchmarks)
    
    # Calculate performance ratios
    if args.verbose:
        print("Calculating performance ratios...")
    ratios = calculate_performance_ratios(normalized_rust, normalized_python)
    
    # Generate comparison chart
    if args.chart:
        if args.verbose:
            print(f"Generating performance chart: {args.chart}")
        generate_comparison_chart(ratios, args.chart)
    
    # Generate detailed report
    if args.verbose:
        print(f"Generating detailed report: {args.output}")
    generate_detailed_report(normalized_rust, normalized_python, ratios, args.output)
    
    # Print summary to console
    print("\n=== Benchmark Comparison Summary ===")
    print(f"Rust benchmarks found: {len(rust_benchmarks)}")
    print(f"Python benchmarks found: {len(python_benchmarks)}")
    print(f"Comparable benchmarks: {len(ratios)}")
    
    if ratios:
        avg_ratio = sum(ratios[name]['ratio'] for name in ratios) / len(ratios)
        print(f"Average performance ratio: {avg_ratio:.2f}x")
        
        if avg_ratio > 1:
            print(f"Rust is {avg_ratio:.2f}x faster on average")
        else:
            print(f"Python is {1/avg_ratio:.2f}x faster on average")
    
    print(f"\nDetailed report saved to: {args.output}")
    if args.chart:
        print(f"Performance chart saved to: {args.chart}")

if __name__ == "__main__":
    main()