import pandas as pd

# Read the CSV file
df = pd.read_csv('project (4).csv')

# Select the desired columns
selected_columns = df[['id', 'url']]

# Save the new CSV file
selected_columns.to_csv('users.csv', index=False)