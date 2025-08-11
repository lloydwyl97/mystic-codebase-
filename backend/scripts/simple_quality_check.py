#!/usr/bin/env python3
"""
Simple Code Quality Check for Mystic Trading Platform

Runs essential quality checks with proper error handling.
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class QualityResult:
    """Result of a quality check"""

    tool: str
    success: bool
    output: str
    error_count: int = 0
    warning_count: int = 0
    duration: float = 0.0


class SimpleQualityChecker:
    """Simple but robust code quality checker"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results: List[QualityResult] = []
        self.start_time = time.time()

        print("🔍 Mystic Trading Platform - Simple Quality Check")
        print("=" * 50)

    def run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run a command with proper encoding handling"""
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self.project_root,
                timeout=60,  # 1 minute timeout
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(cmd, -1, "", "Command timed out")
        except FileNotFoundError:
            return subprocess.CompletedProcess(cmd, -1, "", "Command not found")
        except Exception as e:
            return subprocess.CompletedProcess(cmd, -1, "", f"Error: {str(e)}")

    def check_black(self) -> QualityResult:
        """Check code formatting with black"""
        print("🎨 Checking code formatting (black)...")
        start_time = time.time()

        result = self.run_command(
            [
                sys.executable,
                "-m",
                "black",
                "--check",
                "--line-length=120",
                ".",
            ]
        )

        duration = time.time() - start_time
        success = result.returncode == 0

        return QualityResult(
            tool="black",
            success=success,
            output=result.stdout + result.stderr,
            duration=duration,
        )

    def check_isort(self) -> QualityResult:
        """Check import sorting with isort"""
        print("📦 Checking import sorting (isort)...")
        start_time = time.time()

        result = self.run_command(
            [
                sys.executable,
                "-m",
                "isort",
                "--check-only",
                "--profile=black",
                ".",
            ]
        )

        duration = time.time() - start_time
        success = result.returncode == 0

        return QualityResult(
            tool="isort",
            success=success,
            output=result.stdout + result.stderr,
            duration=duration,
        )

    def check_flake8(self) -> QualityResult:
        """Check code style with flake8"""
        print("🔍 Checking code style (flake8)...")
        start_time = time.time()

        result = self.run_command(
            [
                sys.executable,
                "-m",
                "flake8",
                "--max-line-length=120",
                "--extend-ignore=E203,W503",
                ".",
            ]
        )

        duration = time.time() - start_time
        success = result.returncode == 0

        # Count errors and warnings
        lines = result.stdout.split("\n") if result.stdout else []
        error_count = len([line for line in lines if line.strip() and "E" in line])
        warning_count = len([line for line in lines if line.strip() and "W" in line])

        return QualityResult(
            tool="flake8",
            success=success,
            output=result.stdout,
            error_count=error_count,
            warning_count=warning_count,
            duration=duration,
        )

    def check_bandit(self) -> QualityResult:
        """Check security with bandit"""
        print("🔒 Checking security (bandit)...")
        start_time = time.time()

        result = self.run_command(
            [
                sys.executable,
                "-m",
                "bandit",
                "-r",
                ".",
                "-f",
                "json",
                "-o",
                "bandit-report.json",
            ]
        )

        duration = time.time() - start_time

        # Parse bandit results
        try:
            with open("bandit-report.json", "r", encoding="utf-8") as f:
                bandit_data = json.load(f)
                error_count = len(
                    [
                        issue
                        for issue in bandit_data.get("results", [])
                        if issue.get("issue_severity") == "HIGH"
                    ]
                )
                warning_count = len(
                    [
                        issue
                        for issue in bandit_data.get("results", [])
                        if issue.get("issue_severity") in ["MEDIUM", "LOW"]
                    ]
                )
        except (FileNotFoundError, json.JSONDecodeError):
            error_count = 0
            warning_count = 0

        success = error_count == 0

        return QualityResult(
            tool="bandit",
            success=success,
            output=result.stdout,
            error_count=error_count,
            warning_count=warning_count,
            duration=duration,
        )

    def check_vulture(self) -> QualityResult:
        """Check for dead code with vulture"""
        print("💀 Checking for dead code (vulture)...")
        start_time = time.time()

        result = self.run_command([sys.executable, "-m", "vulture", ".", "--min-confidence", "80"])

        duration = time.time() - start_time

        # Count dead code instances
        lines = result.stdout.split("\n") if result.stdout else []
        warning_count = len([line for line in lines if line.strip() and ":" in line])

        return QualityResult(
            tool="vulture",
            success=warning_count == 0,
            output=result.stdout,
            warning_count=warning_count,
            duration=duration,
        )

    def check_optimization_tests(self) -> QualityResult:
        """Run optimization tests"""
        print("🧪 Running optimization tests...")
        start_time = time.time()

        result = self.run_command([sys.executable, "test_optimizations.py"])

        duration = time.time() - start_time
        success = result.returncode == 0

        return QualityResult(
            tool="optimization_tests",
            success=success,
            output=result.stdout,
            duration=duration,
        )

    def run_all_checks(self) -> List[QualityResult]:
        """Run all quality checks"""
        checks = [
            self.check_black,
            self.check_isort,
            self.check_flake8,
            self.check_bandit,
            self.check_vulture,
            self.check_optimization_tests,
        ]

        for check in checks:
            try:
                result = check()
                self.results.append(result)

                status = "✅ PASS" if result.success else "❌ FAIL"
                print(f"   {status} ({result.duration:.2f}s)")

                if not result.success and result.output:
                    # Show first few lines of output
                    lines = result.output.split("\n")[:3]
                    for line in lines:
                        if line.strip():
                            print(f"   {line[:100]}...")

            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                self.results.append(
                    QualityResult(tool=check.__name__, success=False, output=str(e))
                )

        return self.results

    def print_summary(self):
        """Print quality check summary"""
        total_time = time.time() - self.start_time

        # Calculate summary statistics
        passed = sum(1 for r in self.results if r.success)
        failed = len(self.results) - passed
        total_errors = sum(r.error_count for r in self.results)
        total_warnings = sum(r.warning_count for r in self.results)

        # Determine overall status
        if failed == 0:
            overall_status = "✅ EXCELLENT"
        elif failed <= 2:
            overall_status = "⚠️  GOOD"
        elif failed <= 3:
            overall_status = "⚠️  NEEDS IMPROVEMENT"
        else:
            overall_status = "❌ POOR"

        print("\n" + "=" * 50)
        print("📊 QUALITY CHECK SUMMARY")
        print("=" * 50)
        print(f"Overall Status: {overall_status}")
        print(f"Total Checks: {len(self.results)}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Total Errors: {total_errors}")
        print(f"Total Warnings: {total_warnings}")
        print(f"Total Time: {total_time:.2f}s")

        print("\n📋 Detailed Results:")
        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"  {result.tool:<20} {status}")
            if result.error_count > 0:
                print(f"    Errors: {result.error_count}")
            if result.warning_count > 0:
                print(f"    Warnings: {result.warning_count}")


def main():
    """Main function"""
    checker = SimpleQualityChecker()

    # Run all checks
    results = checker.run_all_checks()

    # Display summary
    checker.print_summary()

    # Exit with appropriate code
    failed_count = sum(1 for r in results if not r.success)
    sys.exit(failed_count)


if __name__ == "__main__":
    main()
