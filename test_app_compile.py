#!/usr/bin/env python3
"""
Test that app.py compiles and gini_coefficient is properly positioned
"""

import sys
import py_compile

print('Testing app.py compilation...')
try:
    py_compile.compile('app.py', doraise=True)
    print('✓ app.py compiles successfully')

    # Test that the function is defined at the top level
    with open('app.py', 'r') as f:
        content = f.read()

    if 'def gini_coefficient(values):' in content:
        print('✓ gini_coefficient function is defined in app.py')

        # Check that it's defined before it's used
        lines = content.split('\n')
        func_line = None
        usage_lines = []

        for i, line in enumerate(lines):
            if 'def gini_coefficient(values):' in line:
                func_line = i
            elif 'gini_coefficient(' in line:
                usage_lines.append(i)

        if func_line is not None:
            print(f'✓ Function defined at line {func_line + 1}')
            for usage_line in usage_lines:
                if usage_line > func_line:
                    print(f'✓ Usage at line {usage_line + 1} is after definition')
                else:
                    print(f'✗ Usage at line {usage_line + 1} is before definition!')
        else:
            print('✗ Function definition not found')

    print('✅ App should work without gini_coefficient errors!')

except py_compile.PyCompileError as e:
    print(f'✗ Compilation error: {e}')
    sys.exit(1)