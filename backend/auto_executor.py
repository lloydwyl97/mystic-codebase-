import importlib.util
import os
from datetime import datetime


class AutoExecutor:
    """Autonomous module execution engine"""

    def __init__(self):
        self.executed_modules = []
        self.execution_history = []

    def run_generated_module(self, file_path):
        """Execute a generated module dynamically"""
        try:
            print(f"[EXECUTOR] Loading module: {file_path}")

            # Load the module dynamically
            spec = importlib.util.spec_from_file_location("autogen", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Execute the module
            result = None
            if hasattr(module, "run"):
                print("[EXECUTOR] Executing run() function...")
                result = module.run()
            elif hasattr(module, "execute_strategy"):
                print("[EXECUTOR] Executing execute_strategy() method...")
                # Create instance and execute
                class_name = [
                    name for name in dir(module) if not name.startswith("_") and name != "run"
                ][0]
                instance = getattr(module, class_name)()
                result = instance.execute_strategy()
            else:
                print(f"[EXECUTOR] No executable function found in {file_path}")
                return None

            # Log execution
            execution_record = {
                "timestamp": datetime.now().isoformat(),
                "file_path": file_path,
                "result": result,
                "status": "success",
            }

            self.execution_history.append(execution_record)
            self.executed_modules.append(file_path)

            print(f"[EXECUTOR] Module executed successfully with result: {result}")
            return result

        except Exception as e:
            print(f"[EXECUTOR] Error executing {file_path}: {e}")

            # Log failed execution
            execution_record = {
                "timestamp": datetime.now().isoformat(),
                "file_path": file_path,
                "result": None,
                "status": "failed",
                "error": str(e),
            }

            self.execution_history.append(execution_record)
            return None

    def execute_all_generated_modules(self):
        """Execute all modules in the generated_modules directory"""
        generated_dir = "generated_modules"

        if not os.path.exists(generated_dir):
            print(f"[EXECUTOR] No {generated_dir} directory found")
            return []

        results = []
        for file_name in os.listdir(generated_dir):
            if file_name.endswith(".py"):
                file_path = os.path.join(generated_dir, file_name)
                result = self.run_generated_module(file_path)
                if result:
                    results.append((file_name, result))

        return results

    def get_execution_stats(self):
        """Get execution statistics"""
        successful = len([r for r in self.execution_history if r["status"] == "success"])
        failed = len([r for r in self.execution_history if r["status"] == "failed"])

        return {
            "total_executions": len(self.execution_history),
            "successful": successful,
            "failed": failed,
            "success_rate": (
                successful / len(self.execution_history) if self.execution_history else 0
            ),
            "latest_executions": (self.execution_history[-5:] if self.execution_history else []),
        }


def run_generated_module(file_path):
    """Simple function interface for external calls"""
    executor = AutoExecutor()
    return executor.run_generated_module(file_path)


def execute_all_modules():
    """Execute all generated modules"""
    executor = AutoExecutor()
    return executor.execute_all_generated_modules()


if __name__ == "__main__":
    # Test execution
    print("[EXECUTOR] Testing auto-execution...")

    # Execute all modules in generated_modules directory
    results = execute_all_modules()

    if results:
        print(f"[EXECUTOR] Executed {len(results)} modules:")
        for file_name, result in results:
            print(f"  - {file_name}: {result}")
    else:
        print("[EXECUTOR] No modules to execute")


