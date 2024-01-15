# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("uber-raw-data-sep14.csv")  # Replace with the actual path

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Check basic information about the dataset
print(data.info())

# Descriptive statistics of numerical columns
print(data.describe())


# Uncover hidden patterns in the data
# For example, analyze the distribution of ride counts over different days and hours.

# Date and time analysis
data['Date/Time'] = pd.to_datetime(data['Date/Time'])
data['hour'] = data['Date/Time'].dt.hour
data['day'] = data['Date/Time'].dt.day_name()


# Analyze ride count per day and hour
ride_count_by_day = data['day'].value_counts()
ride_count_by_hour = data['hour'].value_counts()

print(ride_count_by_day)
print(ride_count_by_hour)





# Data Visualization using Matplotlib and Seaborn

# Plot ride count by day
plt.figure(figsize=(12, 6))
sns.countplot(x='day', data=data, order=data['day'].value_counts().index)
plt.title('Ride Count by Day')
plt.show()

# Plot ride count by hour
plt.figure(figsize=(12, 6))
sns.countplot(x='hour', data=data, order=data['hour'].value_counts().index)
plt.title('Ride Count by Hour')
plt.show()




