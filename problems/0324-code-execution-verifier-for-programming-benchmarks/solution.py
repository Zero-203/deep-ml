import numpy as np

def verify_code_execution(
    test_cases: list[dict],
    numeric_tolerance: float = 1e-6
) -> dict:
    """
    Verify code execution results for a programming benchmark.
    
    Args:
        test_cases: List of dicts with keys:
            - 'expected': Expected output string
            - 'actual': Actual output string (or None if execution failed)
            - 'status': 'success', 'error', or 'timeout'
        numeric_tolerance: Tolerance for floating-point comparisons
        
    Returns:
        Dict with keys:
            - 'pass_rate': Proportion of passed tests (float, rounded to 4 decimals)
            - 'error_rate': Proportion of execution errors (float, rounded to 4 decimals)
            - 'passed_count': Number of passed tests (int)
            - 'total_count': Total number of tests (int)
            - 'verdicts': List of 'pass', 'fail', or 'error' for each test
    """
    # Your code here
    passed, errored, total = 0, 0, len(test_cases)
    verdicts = []
    for test_case in test_cases:
        expected, actual, status = test_case['expected'], test_case['actual'], test_case['status']
        expected = expected.strip()
        actual = actual.strip() if actual is not None else None
        if status == 'error':
            errored += 1
            verdicts.append('error')
        elif status == 'timeout':
            errored += 1
            verdicts.append('error')
        elif status == 'success':
            try: 
                float(expected)
                if actual is None or abs(float(actual) - float(expected)) > numeric_tolerance:
                    verdicts.append('fail')
                else:
                    passed += 1
                    verdicts.append('pass')
            except ValueError:
                if actual is None or actual != expected:
                    verdicts.append('fail')
                else:
                    passed += 1
                    verdicts.append('pass')
        else:
            raise ValueError("Wrong status type.")

    return {
        'pass_rate':round(passed/total if total > 0 else 0.0, 4),
        'error_rate':round(errored/total if total > 0 else 0, 4),
        'passed_count':passed,
        'total_count':total,
        'verdicts':verdicts
    }
