import pandas as pd
import matplotlib.pyplot as plt

# Load the results
df = pd.read_csv('recognition_results.csv')

# Calculate accuracy over time
df['Accuracy'] = df['Correct Recognitions'] / df['Total Recognitions'] * 100

# Plot accuracy over time
plt.figure(figsize=(10, 5))
plt.plot(df['Elapsed Time'], df['Accuracy'], label='Accuracy')
plt.xlabel('Elapsed Time (seconds)')
plt.ylabel('Accuracy (%)')
plt.title('Recognition Accuracy Over Time')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_over_time.png')
plt.show()

# Generate cumulative attendance chart
attendance_df = pd.read_csv('Attendance.csv')
attendance_counts = attendance_df['Name'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
attendance_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Student Names')
plt.ylabel('Attendance Count')
plt.title('Cumulative Attendance Count')
plt.grid(axis='y')
plt.savefig('cumulative_attendance.png')
plt.show()
