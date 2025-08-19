#!/usr/bin/env python3
"""
Automated Code Quality Checks for Mystic Trading Platform

Runs comprehensive code quality analysis including:
- Linting (flake8, pylint)
- Code formatting (black, isort)
- Type checking (mypy)
- Security scanning (bandit)
- Code complexity (radon)
- Dead code detection (vulture)
- Test coverage
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class QualityResult:
    """Result of a quality check"""

    tool: str
    success: bool
    output: str
    error_count: int = 0
    warning_count: int = 0
    duration: float = 0.0


class CodeQualityChecker:
    """Comprehensive code quality checker"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results: list[QualityResult] = []
        self.start_time = time.time()

        # Directories to check
        self.check_dirs = [
            ".",
            "modules",
            "utils",
            "services",
            "routes",
            "middleware",
        ]

        # Files to exclude
        self.exclude_patterns = [
            "*/venv/*",
            "*/venv311/*",
            "*/venv312/*",
            "*/.venv/*",
            "*/__pycache__/*",
            "*/build/*",
            "*/dist/*",
            "*/migrations/*",
            "*/tests/*",
            "*.egg-info/*",
        ]

        print("ðŸ” Mystic Trading Platform - Code Quality Checker")
        print("=" * 60)

    def run_command(
        self, cmd: list[str], capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a command and return the result"""
        try:
            return subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                cwd=self.project_root,
                timeout=300,  # 5 minute timeout
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(cmd, -1, "", f"Command timed out: {' '.join(cmd)}")
        except FileNotFoundError:
            return subprocess.CompletedProcess(cmd, -1, "", f"Command not found: {' '.join(cmd)}")

    def check_black(self) -> QualityResult:
        """Check code formatting with black"""
        print("ðŸŽ¨ Checking code formatting (black)...")
        start_time = time.time()

        result = self.run_command(
            [
                sys.executable,
                "-m",
                "black",
                "--check",
                "--diff",
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
        print("ðŸ“¦ Checking import sorting (isort)...")
        start_time = time.time()

        result = self.run_command(
            [
                sys.executable,
                "-m",
                "isort",
                "--check-only",
                "--diff",
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
        print("ðŸ” Checking code style (flake8)...")
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

    def check_mypy(self) -> QualityResult:
        """Check type annotations with mypy"""
        print("ðŸ” Checking type annotations (mypy)...")
        start_time = time.time()

        result = self.run_command([sys.executable, "-m", "mypy", "--ignore-missing-imports", "."])

        duration = time.time() - start_time
        success = result.returncode == 0

        # Count errors
        lines = result.stdout.split("\n") if result.stdout else []
        error_count = len([line for line in lines if "error:" in line])

        return QualityResult(
            tool="mypy",
            success=success,
            output=result.stdout,
            error_count=error_count,
            duration=duration,
        )

    def check_bandit(self) -> QualityResult:
        """Check security with bandit"""
        print("ðŸ”’ Checking security (bandit)...")
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
            with open("bandit-report.json") as f:
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
        print("ðŸ’€ Checking for dead code (vulture)...")
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

    def check_radon(self) -> QualityResult:
        """Check code complexity with radon"""
        print("ðŸ“Š Checking code complexity (radon)...")
        start_time = time.time()

        result = self.run_command([sys.executable, "-m", "radon", "cc", ".", "-a", "-s"])

        duration = time.time() - start_time

        # Count complex functions
        lines = result.stdout.split("\n") if result.stdout else []
        warning_count = len([line for line in lines if "F" in line and ":" in line])

        return QualityResult(
            tool="radon",
            success=warning_count < 10,  # Allow some complex functions
            output=result.stdout,
            warning_count=warning_count,
            duration=duration,
        )

    def check_pytest(self) -> QualityResult:
        """Run tests with pytest"""
        print("ðŸ§ª Running tests (pytest)...")
        start_time = time.time()

        result = self.run_command([sys.executable, "-m", "pytest", "--tb=short", "-v"])

        duration = time.time() - start_time
        success = result.returncode == 0

        return QualityResult(
            tool="pytest",
            success=success,
            output=result.stdout,
            duration=duration,
        )

    def check_coverage(self) -> QualityResult:
        """Check test coverage"""
        print("ðŸ“ˆ Checking test coverage...")
        start_time = time.time()

        # Run coverage
        self.run_command(
            [sys.executable, "-m", "coverage", "run", "-m", "pytest"]
        )

        # Generate report
        report_result = self.run_command(
            [sys.executable, "-m", "coverage", "report", "--show-missing"]
        )

        duration = time.time() - start_time

        # Parse coverage percentage
        coverage_percent = 0
        if report_result.stdout:
            for line in report_result.stdout.split("\n"):
                if "TOTAL" in line and "%" in line:
                    try:
                        coverage_percent = float(line.split("%")[0].split()[-1])
                        break
                    except (ValueError, IndexError):
                        pass

        success = coverage_percent >= 70  # Minimum 70% coverage

        return QualityResult(
            tool="coverage",
            success=success,
            output=report_result.stdout,
            warning_count=int(100 - coverage_percent),
            duration=duration,
        )

    def run_all_checks(self) -> list[QualityResult]:
        """Run all quality checks"""
        checks = [
            self.check_black,
            self.check_isort,
            self.check_flake8,
            self.check_mypy,
            self.check_bandit,
            self.check_vulture,
            self.check_radon,
            self.check_pytest,
            self.check_coverage,
        ]

        for check in checks:
            try:
                result = check()
                self.results.append(result)

                status = "âœ… PASS" if result.success else "âŒ FAIL"
                print(f"   {status} ({result.duration:.2f}s)")

                if not result.success and result.output:
                    print(f"   Details: {result.output[:200]}...")

            except Exception as e:
                print(f"   âŒ ERROR: {e}")
                self.results.append(
                    QualityResult(tool=check.__name__, success=False, output=str(e))
                )

        return self.results

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive quality report"""
        total_time = time.time() - self.start_time

        # Calculate summary statistics
        passed = sum(1 for r in self.results if r.success)
        failed = len(self.results) - passed
        total_errors = sum(r.error_count for r in self.results)
        total_warnings = sum(r.warning_count for r in self.results)

        # Determine overall status
        if failed == 0:
            overall_status = "âœ… EXCELLENT"
        elif failed <= 2:
            overall_status = "âš ï¸  GOOD"
        elif failed <= 4:
            overall_status = "âš ï¸  NEEDS IMPROVEMENT"
        else:
            overall_status = "âŒ POOR"

        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "summary": {
                "total_checks": len(self.results),
                "passed": passed,
                "failed": failed,
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "total_time": total_time,
            },
            "results": [
                {
                    "tool": r.tool,
                    "success": r.success,
                    "error_count": r.error_count,
                    "warning_count": r.warning_count,
                    "duration": r.duration,
                    "output": (r.output[:500] + "..." if len(r.output) > 500 else r.output),
                }
                for r in self.results
            ],
        }

        return report

    def print_summary(self, report: dict[str, Any]):
        """Print quality check summary"""
        print("\n" + "=" * 60)
        print("ðŸ“Š CODE QUALITY SUMMARY")
        print("=" * 60)

        summary = report["summary"]
        print(f"Overall Status: {report['overall_status']}")
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed']} âœ…")
        print(f"Failed: {summary['failed']} âŒ")
        print(f"Total Errors: {summary['total_errors']}")
        print(f"Total Warnings: {summary['total_warnings']}")
        print(f"Total Time: {summary['total_time']:.2f}s")

        print("\nðŸ“‹ Detailed Results:")
        for result in report["results"]:
            status = "âœ… PASS" if result["success"] else "âŒ FAIL"
            print(f"  {result['tool']:<15} {status}")
            if result["error_count"] > 0:
                print(f"    Errors: {result['error_count']}")
            if result["warning_count"] > 0:
                print(f"    Warnings: {result['warning_count']}")

    def save_report(self, report: dict[str, Any], filename: str = "quality-report.json"):
        """Save quality report to file"""
        try:
            with open(filename, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nðŸ“„ Report saved to: {filename}")
        except Exception as e:
            print(f"\nâŒ Failed to save report: {e}")


def main():
    """Main function"""
    checker = CodeQualityChecker()

    # Run all checks
    results = checker.run_all_checks()

    # Generate and display report
    report = checker.generate_report()
    checker.print_summary(report)

    # Save report
    checker.save_report(report)

    # Exit with appropriate code
    failed_count = sum(1 for r in results if not r.success)
    sys.exit(failed_count)


if __name__ == "__main__":
    main()


