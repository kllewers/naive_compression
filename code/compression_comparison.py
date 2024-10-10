import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Convert to dataframe with headers
df = pd.read_csv("/Users/kitlewers/Desktop/naive_compression/imagery/Compression_Alexey.csv")

#Quick data check of first 5 (non-header columns)
print(df.head())

# Create a scatter plot
plt.figure(figsize=(9,8.5))
plt.bar(df['Method'], df['Size In MB'], color='blue')

# Add titles and labels
plt.title('Compression Methods vs. Size in MB')
plt.xlabel('Method')
plt.ylabel('Size in MB')

# Rotate x-axis labels for better readability
plt.xticks(rotation=20)

# Save the scatter plot as an image
plt.savefig('/Users/kitlewers/Desktop/naive_compression/imagery/bar.png')

