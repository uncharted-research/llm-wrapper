"""
Convenience script to run all grounding investigation tests in sequence.

This will:
1. Run test_grounding_investigation.py
2. Run test_grounding_comparison.py
3. Run test_url_context_vs_grounding.py
4. Generate a summary report
"""

import subprocess
import sys
from pathlib import Path
import time


def print_header(title):
    """Print a nice header."""
    width = 80
    print("\n" + "=" * width)
    print(f"  {title}".center(width))
    print("=" * width + "\n")


def run_test(script_name, description):
    """Run a test script and return success status."""
    print_header(f"Running: {script_name}")
    print(f"Description: {description}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=False
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✓ {script_name} completed successfully in {elapsed:.1f}s")
            return True, elapsed
        else:
            print(f"\n✗ {script_name} failed with return code {result.returncode} after {elapsed:.1f}s")
            return False, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {script_name} crashed: {e}")
        return False, elapsed


def main():
    """Run all tests."""
    print_header("GEMINI GROUNDING INVESTIGATION - FULL TEST SUITE")

    print("This will run all diagnostic tests to investigate:")
    print("  1. Response structure with Google Search grounding")
    print("  2. Comparison of responses with/without grounding")
    print("  3. Separation of url_context vs googleSearch features")
    print("\nEstimated total time: ~3-5 minutes")
    print("\nPress Enter to continue, or Ctrl+C to cancel...")

    try:
        input()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return

    # List of tests to run
    tests = [
        ("test_grounding_investigation.py", "Deep inspection of response structure with grounding"),
        ("test_grounding_comparison.py", "Compare responses with/without grounding enabled"),
        ("test_url_context_vs_grounding.py", "Verify url_context and googleSearch are separate")
    ]

    results = []
    total_start = time.time()

    # Run each test
    for script, description in tests:
        success, elapsed = run_test(script, description)
        results.append((script, success, elapsed))

        # Pause between tests
        if script != tests[-1][0]:  # Not the last test
            print("\nPausing 2 seconds before next test...")
            time.sleep(2)

    total_elapsed = time.time() - total_start

    # Print summary
    print_header("TEST SUITE SUMMARY")

    print("Individual test results:\n")
    print(f"{'Test Script':<45} {'Status':<10} {'Time':<10}")
    print("-" * 80)

    all_passed = True
    for script, success, elapsed in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{script:<45} {status:<10} {elapsed:>6.1f}s")
        if not success:
            all_passed = False

    print("-" * 80)
    print(f"{'Total':<45} {'':<10} {total_elapsed:>6.1f}s")

    # Overall status
    print("\n" + "=" * 80)
    if all_passed:
        print("✓✓✓ ALL TESTS COMPLETED SUCCESSFULLY ✓✓✓".center(80))
    else:
        print("✗✗✗ SOME TESTS FAILED - CHECK OUTPUT ABOVE ✗✗✗".center(80))
    print("=" * 80)

    # Check for test results directory
    test_results_dir = Path(__file__).parent / "test_results"
    if test_results_dir.exists():
        files = list(test_results_dir.glob("*"))
        print(f"\n{len(files)} output files saved to: {test_results_dir}")
        print("\nGenerated files:")
        for f in sorted(files):
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Review the test output above")
    print("2. Check the test_results/ directory for saved outputs")
    print("3. Read GROUNDING_TEST_README.md for how to interpret results")
    print("4. Based on findings, implement fixes to llm_wrapper/llm.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
