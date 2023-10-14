import time
import numpy as np

# Define the total number of iterations in your for loop
total_iterations = 100_000
start_time = time.time()  # Record the start time

# Function to print a text-based progress bar with estimated time remaining in minutes
def print_progress_bar(iteration, total, start_time, bar_length=100):
    progress = (iteration / total)
    arrow = '=' * int(round(bar_length * progress))
    spaces = ' ' * (bar_length - len(arrow))

    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60.0
    estimated_total_time = elapsed_time / progress if progress > 0 else 0
    estimated_remaining_time = (estimated_total_time - elapsed_time) / 60.0

    print(f'\r[{arrow}{spaces}] {np.round(progress * 100, 3)}% '
          f'\t\tElapsed: {np.round(elapsed_minutes, 2)} min '
          f'\t\tETA: {np.round(estimated_remaining_time, 2)} min', end='')

# Simulate a for loop
for i in range(total_iterations):
    # Your processing code here
    time.sleep(0.1)  # Simulate some work
    # Update the progress bar with estimated time in minutes
    print_progress_bar(i + 1, total_iterations, start_time)

# Print a new line to move the cursor to the next line after the loop finishes
print()
