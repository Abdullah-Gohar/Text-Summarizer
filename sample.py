import pandas as pd

# Load data from the original Excel file
original_file = 'OEL/Dataset.xlsx'
data = pd.read_excel(original_file)

# Take a random 20% sample of the data
sampled_data = data.sample(frac=0.02, random_state=42)

# Save the sampled data to a new Excel file
sampled_file = 'OEL/sampled_data.xlsx'
sampled_data.to_excel(sampled_file, index=False)

print(f"Sampled data saved to {sampled_file}")
