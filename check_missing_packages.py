#!/usr/bin/env python3
"""
Script to identify missing packages by comparing requirements with installed packages.
"""


import pkg_resources


def parse_requirements_file(file_path: str) -> set[str]:
    """Parse a requirements file and extract package names."""
    packages = set()
    try:
        # Try different encodings
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
            try:
                with open(file_path, encoding=encoding) as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments, empty lines, and section headers
                        if (line and 
                            not line.startswith('#') and 
                            not line.startswith('=') and
                            not line.startswith('=====')):
                            # Extract package name (before ==, >=, etc.)
                            if '==' in line:
                                package_name = line.split('==')[0].strip()
                            elif '>=' in line:
                                package_name = line.split('>=')[0].strip()
                            else:
                                # Skip lines without version specifiers
                                continue
                            
                            if package_name:
                                packages.add(package_name.lower())
                break  # If we get here, the encoding worked
            except UnicodeDecodeError:
                continue
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
    return packages

def get_installed_packages() -> dict[str, str]:
    """Get currently installed packages."""
    return {pkg.key.lower(): pkg.version for pkg in pkg_resources.working_set}

def main():
    # Get installed packages
    installed = get_installed_packages()
    print(f"Total installed packages: {len(installed)}")
    
    # Parse comprehensive requirements
    comprehensive_reqs = parse_requirements_file('all_requirements_combined.txt')
    print(f"Total packages in comprehensive requirements: {len(comprehensive_reqs)}")
    
    # Find missing packages
    missing = comprehensive_reqs - set(installed.keys())
    
    print(f"\nMissing packages ({len(missing)}):")
    for package in sorted(missing):
        print(f"  - {package}")
    
    # Check for version mismatches
    print("\nChecking for version mismatches...")
    mismatches = []
    
    # Re-parse requirements with versions
    for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
        try:
            with open('all_requirements_combined.txt', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if (line and 
                        not line.startswith('#') and 
                        not line.startswith('=') and
                        not line.startswith('=====') and
                        '==' in line):
                        package_name, required_version = line.split('==', 1)
                        package_name = package_name.strip().lower()
                        required_version = required_version.strip()
                        
                        if package_name in installed:
                            installed_version = installed[package_name]
                            if installed_version != required_version:
                                mismatches.append((package_name, required_version, installed_version))
            break  # If we get here, the encoding worked
        except UnicodeDecodeError:
            continue
    
    print(f"Version mismatches ({len(mismatches)}):")
    for package, required, installed in mismatches:
        print(f"  - {package}: required {required}, installed {installed}")

if __name__ == "__main__":
    main()
