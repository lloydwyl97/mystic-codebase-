#!/usr/bin/env python3
"""
Comprehensive Code Quality Check Script for Mystic Trading Platform
Runs all professional-grade code quality tools in sequence.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


class QualityChecker:
    """Professional code quality checker with comprehensive reporting"""

    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.start_time = time.time()
        self.backend_dir = Path(__file__).parent
        self.project_root = self.backend_dir.parent
        self.python_executable = sys.executable

    def run_command(self, command: List[str], name: str, cwd: Path = None) -> Dict[str, Any]:
        """Run a command and capture results"""
        if cwd is None:
            cwd = self.backend_dir

        print(f"\n{'='*60}")
        print(f"Running {name}...")
        print(f"{'='*60}")

        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            success = result.returncode == 0
            output = result.stdout + result.stderr

            print(f"✅ {name} completed successfully" if success else f"❌ {name} failed")
            if output.strip():
                print(output)

            return {
                "success": success,
                "returncode": result.returncode,
                "output": output,
                "command": " ".join(command),
            }

        except subprocess.TimeoutExpired:
            print(f"⏰ {name} timed out after 5 minutes")
            return {
                "success": False,
                "returncode": -1,
                "output": "Command timed out",
                "command": " ".join(command),
            }
        except Exception as e:
            print(f"💥 {name} failed with exception: {e}")
            return {
                "success": False,
                "returncode": -1,
                "output": str(e),
                "command": " ".join(command),
            }

    def run_python_checks(self):
        """Run all Python code quality checks"""
        print("\n🐍 Running Python Code Quality Checks...")

        # Code formatting
        self.results["black"] = self.run_command(
            [self.python_executable, "-m", "black", "--check", "--diff", "."],
            "Black (Code Formatting)",
        )

        self.results["isort"] = self.run_command(
            [
                self.python_executable,
                "-m",
                "isort",
                "--check-only",
                "--diff",
                ".",
            ],
            "isort (Import Sorting)",
        )

        # Linting
        self.results["flake8"] = self.run_command(
            [self.python_executable, "-m", "flake8", "."], "Flake8 (Linting)"
        )

        self.results["pylint"] = self.run_command(
            [
                self.python_executable,
                "-m",
                "pylint",
                "--rcfile=.pylintrc",
                "modules",
                "utils",
                "*.py",
            ],
            "Pylint (Advanced Linting)",
        )

        # Type checking
        self.results["mypy"] = self.run_command(
            [self.python_executable, "-m", "mypy", "."], "MyPy (Type Checking)"
        )

        # Security scanning
        self.results["bandit"] = self.run_command(
            [self.python_executable, "-m", "bandit", "-r", "."],
            "Bandit (Security Scanning)",
        )

        self.results["safety"] = self.run_command(
            [self.python_executable, "-m", "safety", "check"],
            "Safety (Dependency Security)",
        )

        # Code complexity and quality
        self.results["radon"] = self.run_command(
            [self.python_executable, "-m", "radon", "cc", "."],
            "Radon (Cyclomatic Complexity)",
        )

        self.results["vulture"] = self.run_command(
            [self.python_executable, "-m", "vulture", "."],
            "Vulture (Unused Code Detection)",
        )

        self.results["lizard"] = self.run_command(
            [self.python_executable, "-m", "lizard", "."],
            "Lizard (Code Metrics)",
        )

        # Static analysis
        self.results["pylama"] = self.run_command(
            [self.python_executable, "-m", "pylama", "."],
            "Pylama (Static Analysis)",
        )

    def run_node_checks(self):
        """Run Node.js code quality checks"""
        print("\n🟢 Running Node.js Code Quality Checks...")

        # Code duplication detection
        self.results["jscpd"] = self.run_command(
            ["jscpd", "."],
            "JSCPD (Code Duplication Detection)",
            cwd=self.project_root,
        )

    def run_tests(self):
        """Run tests with coverage"""
        print("\n🧪 Running Tests and Coverage...")

        self.results["pytest"] = self.run_command(
            [
                self.python_executable,
                "-m",
                "pytest",
                "--cov=backend",
                "--cov-report=term-missing",
                "--cov-report=html",
            ],
            "Pytest (Tests and Coverage)",
        )

    def run_documentation_checks(self):
        """Run documentation checks"""
        print("\n📚 Running Documentation Checks...")

        self.results["docformatter"] = self.run_command(
            [
                self.python_executable,
                "-m",
                "docformatter",
                "--check",
                "--recursive",
                ".",
            ],
            "Docformatter (Docstring Formatting)",
        )

    def generate_report(self):
        """Generate comprehensive quality report"""
        print(f"\n{'='*60}")
        print("📊 QUALITY CHECK REPORT")
        print(f"{'='*60}")

        total_checks = len(self.results)
        passed_checks = sum(1 for result in self.results.values() if result["success"])
        failed_checks = total_checks - passed_checks

        print("\nOverall Results:")
        print(f"✅ Passed: {passed_checks}/{total_checks}")
        print(f"❌ Failed: {failed_checks}/{total_checks}")
        print(f"📈 Success Rate: {(passed_checks/total_checks)*100:.1f}%")

        elapsed_time = time.time() - self.start_time
        print(f"⏱️  Total Time: {elapsed_time:.2f} seconds")

        if failed_checks > 0:
            print("\n❌ Failed Checks:")
            for name, result in self.results.items():
                if not result["success"]:
                    print(f"  - {name}: {result.get('output', 'Unknown error')[:100]}...")

        # Save detailed report
        report_file = self.backend_dir / "quality_report.json"
        with open(report_file, "w") as f:
            json.dump(
                {
                    "timestamp": time.time(),
                    "elapsed_time": elapsed_time,
                    "summary": {
                        "total_checks": total_checks,
                        "passed_checks": passed_checks,
                        "failed_checks": failed_checks,
                        "success_rate": (passed_checks / total_checks) * 100,
                    },
                    "results": self.results,
                },
                f,
                indent=2,
            )

        print(f"\n📄 Detailed report saved to: {report_file}")

        return failed_checks == 0

    def run_all_checks(self):
        """Run all quality checks"""
        print("🚀 Starting Comprehensive Code Quality Checks...")
        print(f"📁 Working directory: {self.backend_dir}")
        print(f"🐍 Using Python: {self.python_executable}")

        try:
            self.run_python_checks()
            self.run_node_checks()
            self.run_tests()
            self.run_documentation_checks()

            success = self.generate_report()

            if success:
                print("\n🎉 All quality checks passed!")
                return 0
            else:
                print("\n⚠️  Some quality checks failed. Please review the report above.")
                return 1

        except KeyboardInterrupt:
            print("\n⏹️  Quality checks interrupted by user")
            return 1
        except Exception as e:
            print(f"\n💥 Quality checks failed with exception: {e}")
            return 1


def main():
    """Main entry point"""
    checker = QualityChecker()
    return checker.run_all_checks()


if __name__ == "__main__":
    sys.exit(main())
