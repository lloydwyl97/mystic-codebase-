#!/usr/bin/env python3
"""
Debug script to see what's being parsed from requirements file.
"""

def debug_requirements():
    with open('all_requirements_combined.txt') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('='):
                print(f"Line {i}: '{line}'")
                if '==' in line:
                    package_name = line.split('==')[0].strip()
                    print(f"  -> Package: '{package_name}'")
                elif '>=' in line:
                    package_name = line.split('>=')[0].strip()
                    print(f"  -> Package: '{package_name}'")

if __name__ == "__main__":
    debug_requirements()
