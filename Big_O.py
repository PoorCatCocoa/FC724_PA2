"""
This file provides tools for analyzing time complexity of functions.
"""
import time
from functools import wraps
import pandas as pd


def time_complexity(func):		#Decorator to measure and print a function's execution time.
	@wraps(func)
	def wrapper(*args, **kwargs):
		start = time.perf_counter()
		result = func(*args, **kwargs)
		end = time.perf_counter()
		print(f"\n⏱️ Function {func.__name__} executed in {end - start:.6f} seconds")
		return result

	return wrapper


def run_complexity_tests():					#Runs empirical scalability tests for process_data() at different data scales.
	print("\n=== Big-O Complexity Analysis ===")
	from backup import process_data			# Avoid circular import

	df_csv = pd.read_csv('Renewable_Energy_Adoption.csv')	# Load original dataset
	original_rows = len(df_csv)

	print("\n📊 Theoretical Complexity:")	# Theoretical complexity explanation
	print("1. process_item(): O(1) - Constant time (fixed column count)")
	print("2. process_data(): O(n*m) - Quadratic time")
	print("   (n = rows, m = columns)")
	print("3. Interpolation: O(n) - Linear search")

	print("\n🔬 Empirical Testing:")		# Empirical testing with scaled datasets
	test_scales = [100, 500, 1000]			# Test dataset sizes


	for scale in test_scales:				# Scale dataset by repeating original data
		repeat_times = (scale // original_rows) + 1
		test_df = pd.concat([df_csv] * repeat_times, ignore_index=True)[:scale]
		print(f"\nTesting with {len(test_df)} rows dataset:")
		_ = process_data(test_df.copy())	# Execute and measure time


if __name__ == "__main__":					# Prevent other files load the execute while other files import
	run_complexity_tests()