#!/usr/bin/env python
"""
CDI Syntax & Static Check - No Dependencies Required
=====================================================

Performs compile-time checks on Python files:
  - Python syntax validation
  - AST parsing
  - Import statement analysis
  - Basic structure validation

Does NOT require torch, ethiobbpe, or datasets to be installed.
"""

import sys
import py_compile
import ast
import traceback
from pathlib import Path
from typing import Dict, List, Tuple


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class SyntaxCheckResult:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.passed = False
        self.error = None
        self.has_syntax_error = False
        self.has_import_error = False
        self.imports = []
        self.classes = []
        self.functions = []
    
    def __repr__(self):
        status = f"{Colors.GREEN}✓{Colors.ENDC}" if self.passed else f"{Colors.RED}✗{Colors.ENDC}"
        return f"{status} {Path(self.filepath).name}"


def check_syntax(filepath: str) -> SyntaxCheckResult:
    """Check Python file for syntax errors."""
    result = SyntaxCheckResult(filepath)
    
    try:
        # Compile check
        py_compile.compile(filepath, doraise=True)
        
        # Parse AST
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = ast.parse(code, filepath)
        
        # Extract metadata
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    result.imports.append(node.module)
            elif isinstance(node, ast.ClassDef):
                result.classes.append(node.name)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                result.functions.append(node.name)
        
        result.passed = True
        return result
    
    except SyntaxError as e:
        result.has_syntax_error = True
        result.error = f"Syntax Error at line {e.lineno}: {e.msg}"
        return result
    
    except Exception as e:
        result.error = f"{type(e).__name__}: {str(e)}"
        return result


def main():
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔" + "=" * 68 + "╗")
    print("║" + "  CDI SYNTAX & STATIC CHECK".center(68) + "║")
    print("║" + "  No Dependencies Required".center(68) + "║")
    print("╚" + "=" * 68 + "╝")
    print(f"{Colors.ENDC}")
    
    root_dir = Path(__file__).parent.parent  # Go up to CDI root
    cdi_dir = root_dir / "cdi"
    
    # Find all Python files
    python_files = list(cdi_dir.rglob("*.py"))
    python_files.extend([
        root_dir / "dataset.py",
        root_dir / "train.py",
    ])
    python_files = [f for f in python_files if f.exists()]
    
    print(f"\n{Colors.BOLD}Found {len(python_files)} Python files{Colors.ENDC}\n")
    
    # Check syntax for each file
    print(f"{Colors.CYAN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}SYNTAX CHECK{Colors.ENDC}")
    print(f"{Colors.CYAN}{'='*70}{Colors.ENDC}\n")
    
    results = []
    for filepath in sorted(python_files):
        result = check_syntax(str(filepath))
        results.append(result)
        
        print(f"  {result}")
        if result.error:
            print(f"    {Colors.RED}Error: {result.error}{Colors.ENDC}")
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    
    print(f"\n{Colors.CYAN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}SYNTAX CHECK SUMMARY{Colors.ENDC}")
    print(f"{Colors.CYAN}{'='*70}{Colors.ENDC}\n")
    
    print(f"  Total files:      {len(results)}")
    print(f"  {Colors.GREEN}✓ Passed:{Colors.ENDC}        {passed}")
    print(f"  {Colors.RED}✗ Failed:{Colors.ENDC}        {failed}")
    
    # File statistics
    print(f"\n{Colors.BOLD}File Statistics:{Colors.ENDC}")
    total_classes = sum(len(r.classes) for r in results if r.passed)
    total_functions = sum(len(r.functions) for r in results if r.passed)
    total_imports = len(set(imp for r in results if r.passed for imp in r.imports))
    
    print(f"  Classes:          {total_classes}")
    print(f"  Functions:        {total_functions}")
    print(f"  Unique imports:   {total_imports}")
    
    # Dependency analysis
    print(f"\n{Colors.BOLD}External Dependencies:{Colors.ENDC}")
    all_imports = set()
    for r in results:
        if r.passed:
            all_imports.update(r.imports)
    
    core_libs = {'os', 'sys', 'json', 'math', 'time', 'pathlib', 'typing', 
                 'dataclasses', 'traceback', 'abc', 'argparse', 'collections',
                 '__future__', 'datetime', 'itertools', 'functools'}
    
    external = sorted([imp for imp in all_imports 
                      if not imp.startswith('cdi') and imp not in core_libs])
    
    for imp in external:
        status = f"{Colors.YELLOW}→{Colors.ENDC}"
        print(f"  {status} {imp}")
    
    # Detailed file info
    if False:  # Set to True for verbose output
        print(f"\n{Colors.BOLD}Detailed File Info:{Colors.ENDC}\n")
        for result in sorted(results, key=lambda r: r.filepath):
            if result.passed:
                rel_path = Path(result.filepath).relative_to(Path(__file__).parent)
                print(f"  {Colors.GREEN}✓{Colors.ENDC} {rel_path}")
                if result.classes:
                    print(f"      Classes: {', '.join(result.classes)}")
                if result.functions:
                    func_preview = ', '.join(result.functions[:5])
                    if len(result.functions) > 5:
                        func_preview += f", ... ({len(result.functions) - 5} more)"
                    print(f"      Functions: {func_preview}")
    
    # Final status
    print(f"\n{Colors.CYAN}{'='*70}{Colors.ENDC}")
    if failed == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL FILES HAVE VALID SYNTAX!{Colors.ENDC}")
        print(f"  Ready for dependency installation and integration testing.")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ {failed} FILE(S) HAVE SYNTAX ERRORS{Colors.ENDC}")
        print(f"  Fix these issues before proceeding.")
    print(f"{Colors.CYAN}{'='*70}{Colors.ENDC}\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
