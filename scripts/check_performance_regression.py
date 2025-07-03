#!/usr/bin/env python3
"""
Performance Regression Detection Script for uubed Project

This script analyzes performance benchmark trends to detect regressions
across multiple days and generates alerts when performance degrades
beyond acceptable thresholds.
"""

import json
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import statistics

class PerformanceRegression:
    """Class to handle performance regression detection."""
    
    def __init__(self, threshold_percent: float = 5.0, lookback_days: int = 7):
        """
        Initialize regression detector.
        
        Args:
            threshold_percent: Percentage increase that constitutes a regression
            lookback_days: Number of days to look back for trend analysis
        """
        self.threshold_percent = threshold_percent
        self.lookback_days = lookback_days
        self.regressions = []
        
    def fetch_benchmark_history(self, repo_name: str, benchmark_name: str) -> List[Dict]:
        """
        Fetch benchmark history from GitHub Pages benchmark data.
        
        Args:
            repo_name: Repository name (e.g., 'uubed-rs')
            benchmark_name: Name of the benchmark
            
        Returns:
            List of benchmark data points with timestamps
        """
        try:
            # URL pattern for benchmark-action GitHub Pages data
            url = f"https://twardoch.github.io/{repo_name}/dev/bench/data.js"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse JavaScript data (usually starts with 'window.BENCHMARK_DATA = ')
            data_text = response.text
            if data_text.startswith('window.BENCHMARK_DATA = '):
                json_data = data_text[len('window.BENCHMARK_DATA = '):]
                if json_data.endswith(';'):
                    json_data = json_data[:-1]
                
                benchmark_data = json.loads(json_data)
                
                # Extract history for specific benchmark
                history = []
                for entry in benchmark_data.get('entries', {}).get(benchmark_name, []):
                    history.append({
                        'timestamp': entry.get('commit', {}).get('timestamp'),
                        'value': entry.get('value', 0),
                        'unit': entry.get('unit', 'ns'),
                        'commit': entry.get('commit', {}).get('id', ''),
                        'message': entry.get('commit', {}).get('message', '')
                    })
                
                return history
                
        except Exception as e:
            print(f"Error fetching benchmark history for {repo_name}/{benchmark_name}: {e}")
            return []
    
    def analyze_trend(self, history: List[Dict]) -> Dict:
        """
        Analyze performance trend over the lookback period.
        
        Args:
            history: List of benchmark data points
            
        Returns:
            Trend analysis results
        """
        if len(history) < 2:
            return {'status': 'insufficient_data', 'message': 'Not enough data points'}
        
        # Sort by timestamp
        history.sort(key=lambda x: x['timestamp'])
        
        # Filter to lookback period
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        recent_history = [
            point for point in history 
            if datetime.fromisoformat(point['timestamp'].replace('Z', '+00:00')) >= cutoff_date
        ]
        
        if len(recent_history) < 2:
            return {'status': 'insufficient_recent_data', 'message': 'Not enough recent data'}
        
        # Calculate trend
        values = [point['value'] for point in recent_history]
        baseline = statistics.mean(values[:len(values)//2]) if len(values) >= 4 else values[0]
        recent = statistics.mean(values[len(values)//2:]) if len(values) >= 4 else values[-1]
        
        # Calculate percentage change
        if baseline > 0:
            percent_change = ((recent - baseline) / baseline) * 100
        else:
            percent_change = 0
        
        # Determine regression status
        is_regression = percent_change > self.threshold_percent
        
        return {
            'status': 'regression' if is_regression else 'normal',
            'baseline': baseline,
            'recent': recent,
            'percent_change': percent_change,
            'threshold': self.threshold_percent,
            'data_points': len(recent_history),
            'timespan_days': self.lookback_days,
            'is_regression': is_regression
        }
    
    def check_all_benchmarks(self, repositories: List[str], benchmarks: List[str]) -> Dict:
        """
        Check all benchmarks across all repositories for regressions.
        
        Args:
            repositories: List of repository names
            benchmarks: List of benchmark names to check
            
        Returns:
            Dictionary with regression analysis results
        """
        results = {}
        
        for repo in repositories:
            results[repo] = {}
            
            for benchmark in benchmarks:
                print(f"Checking {repo}/{benchmark}...")
                
                history = self.fetch_benchmark_history(repo, benchmark)
                analysis = self.analyze_trend(history)
                
                results[repo][benchmark] = analysis
                
                if analysis.get('is_regression', False):
                    self.regressions.append({
                        'repository': repo,
                        'benchmark': benchmark,
                        'analysis': analysis
                    })
        
        return results
    
    def generate_regression_report(self, results: Dict, output_path: str):
        """
        Generate a detailed regression report.
        
        Args:
            results: Results from check_all_benchmarks
            output_path: Path to save the report
        """
        report = []
        
        # Header
        report.append("# Performance Regression Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append(f"Threshold: {self.threshold_percent}% performance degradation")
        report.append(f"Lookback Period: {self.lookback_days} days")
        report.append("")
        
        # Summary
        total_benchmarks = sum(len(repo_results) for repo_results in results.values())
        regression_count = len(self.regressions)
        
        report.append("## Summary")
        report.append("")
        report.append(f"- **Total Benchmarks Analyzed**: {total_benchmarks}")
        report.append(f"- **Regressions Detected**: {regression_count}")
        report.append(f"- **Regression Rate**: {(regression_count/total_benchmarks)*100:.1f}%")
        report.append("")
        
        if regression_count > 0:
            report.append("## 🚨 Performance Regressions Detected")
            report.append("")
            
            for regression in self.regressions:
                repo = regression['repository']
                benchmark = regression['benchmark']
                analysis = regression['analysis']
                
                report.append(f"### {repo} - {benchmark}")
                report.append("")
                report.append(f"- **Performance Degradation**: {analysis['percent_change']:.2f}%")
                report.append(f"- **Baseline Performance**: {analysis['baseline']:.6f}")
                report.append(f"- **Recent Performance**: {analysis['recent']:.6f}")
                report.append(f"- **Data Points**: {analysis['data_points']}")
                report.append("")
                report.append("**Action Required**: Investigate performance degradation in this benchmark.")
                report.append("")
        else:
            report.append("## ✅ No Performance Regressions")
            report.append("")
            report.append("All benchmarks are performing within acceptable parameters.")
            report.append("")
        
        # Detailed Results
        report.append("## Detailed Analysis")
        report.append("")
        
        for repo, repo_results in results.items():
            report.append(f"### {repo}")
            report.append("")
            
            if repo_results:
                report.append("| Benchmark | Status | Change | Baseline | Recent | Data Points |")
                report.append("|-----------|--------|--------|----------|--------|-------------|")
                
                for benchmark, analysis in repo_results.items():
                    status = "🚨 REGRESSION" if analysis.get('is_regression', False) else "✅ Normal"
                    change = f"{analysis.get('percent_change', 0):.2f}%"
                    baseline = f"{analysis.get('baseline', 0):.6f}"
                    recent = f"{analysis.get('recent', 0):.6f}"
                    points = analysis.get('data_points', 0)
                    
                    report.append(f"| {benchmark} | {status} | {change} | {baseline} | {recent} | {points} |")
                
                report.append("")
            else:
                report.append("No benchmark data available for this repository.")
                report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        if regression_count > 0:
            report.append("### Immediate Actions")
            report.append("")
            report.append("1. **Investigate Root Cause**: Review recent commits that may have introduced performance regressions")
            report.append("2. **Profile Code**: Use profiling tools to identify performance bottlenecks")
            report.append("3. **Compare Implementations**: Check if optimizations in one language can be applied to others")
            report.append("4. **Update Benchmarks**: Ensure benchmarks are still relevant and accurate")
            report.append("")
            
            report.append("### Long-term Monitoring")
            report.append("")
            report.append("1. **Lower Threshold**: Consider lowering the regression threshold for critical benchmarks")
            report.append("2. **More Frequent Checks**: Increase the frequency of performance monitoring")
            report.append("3. **Automated Alerts**: Set up automated alerts for performance team")
            report.append("4. **Performance Budget**: Establish performance budgets for each component")
            report.append("")
        else:
            report.append("### Continuous Improvement")
            report.append("")
            report.append("1. **Maintain Current Practices**: Current performance monitoring is effective")
            report.append("2. **Optimize Further**: Look for opportunities to improve performance beyond current levels")
            report.append("3. **Expand Coverage**: Consider adding more benchmarks for comprehensive coverage")
            report.append("4. **Cross-Language Learning**: Apply optimizations learned in one language to others")
            report.append("")
        
        # Footer
        report.append("## Methodology")
        report.append("")
        report.append("- **Data Source**: GitHub Pages benchmark data from benchmark-action")
        report.append("- **Trend Analysis**: Comparing recent performance against baseline average")
        report.append("- **Regression Criteria**: Performance degradation exceeding threshold percentage")
        report.append("- **Statistical Method**: Mean comparison with configurable lookback period")
        report.append("")
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))

def main():
    """Main function to orchestrate regression detection."""
    parser = argparse.ArgumentParser(description='Check for performance regressions')
    parser.add_argument('--threshold', type=float, default=5.0, 
                       help='Regression threshold percentage (default: 5.0)')
    parser.add_argument('--days', type=int, default=7, 
                       help='Days to look back for trend analysis (default: 7)')
    parser.add_argument('--output', required=True, 
                       help='Output path for regression report')
    parser.add_argument('--repositories', nargs='+', 
                       default=['uubed-rs', 'uubed-py'],
                       help='Repositories to check (default: uubed-rs uubed-py)')
    parser.add_argument('--benchmarks', nargs='+',
                       default=['encode_q64', 'decode_q64', 'encode_mq64', 'decode_mq64'],
                       help='Benchmarks to check')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize regression detector
    detector = PerformanceRegression(
        threshold_percent=args.threshold,
        lookback_days=args.days
    )
    
    if args.verbose:
        print(f"Checking for regressions with {args.threshold}% threshold over {args.days} days")
        print(f"Repositories: {args.repositories}")
        print(f"Benchmarks: {args.benchmarks}")
    
    # Check all benchmarks
    results = detector.check_all_benchmarks(args.repositories, args.benchmarks)
    
    # Generate report
    if args.verbose:
        print(f"Generating regression report: {args.output}")
    
    detector.generate_regression_report(results, args.output)
    
    # Print summary
    total_benchmarks = sum(len(repo_results) for repo_results in results.values())
    regression_count = len(detector.regressions)
    
    print(f"\n=== Performance Regression Analysis ===")
    print(f"Total benchmarks analyzed: {total_benchmarks}")
    print(f"Regressions detected: {regression_count}")
    print(f"Regression rate: {(regression_count/total_benchmarks)*100:.1f}%")
    
    if regression_count > 0:
        print(f"\n🚨 PERFORMANCE REGRESSIONS DETECTED!")
        for regression in detector.regressions:
            repo = regression['repository']
            benchmark = regression['benchmark']
            change = regression['analysis']['percent_change']
            print(f"  - {repo}/{benchmark}: {change:.2f}% degradation")
        
        print(f"\nDetailed report saved to: {args.output}")
        
        # Exit with error code for CI/CD
        sys.exit(1)
    else:
        print(f"\n✅ No performance regressions detected")
        print(f"Report saved to: {args.output}")

if __name__ == "__main__":
    main()