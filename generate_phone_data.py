import pandas as pd
import numpy as np

# Set the random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate random data
# Assume friends_count ranges from 0 to 1000
friends_count = np.random.randint(0, 1001, n_samples)
# Assume daily_work_hours ranges from 0 to 16
daily_work_hours = np.random.randint(0, 17, n_samples)

# Assume phd (0 for no, 1 for yes)
phd = np.random.randint(0, 2, n_samples)

# Assume the time spent on phone is influenced by the number of friends, daily work hours, and having a PhD or not,
# but also include some random noise.
time_spent_on_phone = (
    60 + # Base time
    daily_work_hours * 8 + # More work, less time
    friends_count * 0.15 - # More friends, slightly more time
    phd * 30 + # Having a PhD increases time
    0*np.random.normal(0, 25, n_samples) # Random noise
)

# Create a DataFrame
data = pd.DataFrame({
    'Friends Count': friends_count,
    'Daily Work Hours': daily_work_hours,
    'Has PhD': phd,
    'Time Spent on Phone (min)': time_spent_on_phone
})


file_path = "machine_learning\\phone_usage.txt"


# Write the DataFrame to a text file
data.to_csv(file_path, sep='\t', index=False)
