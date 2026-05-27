import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# LOAD DATASET
df = pd.read_csv(
    r"C:\Users\shwet\Downloads\large_fitness_tracker_dataset.csv"
)

print("FIRST 5 ROWS")
print(df.head())

print("\nDATASET INFO")
print(df.info())

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# VISUAL STYLE
sns.set(style="whitegrid")


# 1. TOP 10 STEP COUNTS
top_steps = df.sort_values(
    by='Steps',
    ascending=False
).head(10)

plt.figure(figsize=(14,6))

sns.barplot(
    x='Name',
    y='Steps',
    data=top_steps
)

plt.title("Top 10 Highest Step Counts")
plt.xlabel("Person")
plt.ylabel("Steps Walked")

plt.xticks(rotation=45)

plt.savefig("graph_1_bar_chart.png",
            dpi=300,
            bbox_inches='tight')

plt.show()

# 2. WORKOUT HOURS VS CALORIES
plt.figure(figsize=(10,6))

sns.scatterplot(
    x='WorkoutHours',
    y='CaloriesBurned',
    hue='Gender',
    size='Steps',
    data=df,
    sizes=(20,300)
)

plt.title("Workout Hours vs Calories Burned")
plt.xlabel("Workout Hours")
plt.ylabel("Calories Burned")

plt.savefig("graph_2_scatter.png",
            dpi=300,
            bbox_inches='tight')

plt.show()

# 3. HEART RATE HISTOGRAM
plt.figure(figsize=(10,6))

sns.histplot(
    df['HeartRate'],
    bins=20,
    kde=True
)

plt.title("Heart Rate Distribution")
plt.xlabel("Heart Rate")
plt.ylabel("Frequency")

plt.savefig("graph_3_histogram.png",
            dpi=300,
            bbox_inches='tight')

plt.show()

# 4. BMI DISTRIBUTION
plt.figure(figsize=(8,6))

sns.boxplot(
    x='Gender',
    y='BMI',
    data=df
)

plt.title("BMI Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("BMI")

plt.savefig("graph_4_boxplot.png",
            dpi=300,
            bbox_inches='tight')

plt.show()

# 5. SLEEP HOURS DISTRIBUTION
plt.figure(figsize=(10,6))

sns.histplot(
    df['HoursSleep'],
    bins=10,
    kde=True
)

plt.title("Sleep Hours Distribution")
plt.xlabel("Hours Sleep")
plt.ylabel("Frequency")

plt.savefig("graph_5_sleep.png",
            dpi=300,
            bbox_inches='tight')

plt.show()

# 6. WATER INTAKE VS CALORIES
plt.figure(figsize=(10,6))

sns.lineplot(
    x='WaterIntake',
    y='CaloriesBurned',
    data=df,
    marker='o'
)

plt.title("Water Intake vs Calories Burned")
plt.xlabel("Water Intake")
plt.ylabel("Calories Burned")

plt.savefig("graph_6_lineplot.png",
            dpi=300,
            bbox_inches='tight')

plt.show()

# 7. AGE VS STEPS

plt.figure(figsize=(10,6))

sns.scatterplot(
    x='Age',
    y='Steps',
    hue='Gender',
    data=df
)

plt.title("Age vs Steps Count")
plt.xlabel("Age")
plt.ylabel("Steps")

plt.savefig("graph_7_age_steps.png",
            dpi=300,
            bbox_inches='tight')

plt.show()

# 8. GENDER PIE CHART
gender_count = df['Gender'].value_counts()

plt.figure(figsize=(7,7))

plt.pie(
    gender_count,
    labels=gender_count.index,
    autopct='%1.1f%%'
)

plt.title("Gender Distribution")

plt.savefig("graph_8_piechart.png",
            dpi=300,
            bbox_inches='tight')

plt.show()

# 9. AVERAGE FITNESS METRICS
average_metrics = df[
    ['Steps',
     'CaloriesBurned',
     'WaterIntake',
     'WorkoutHours',
     'HeartRate']
].mean()

plt.figure(figsize=(10,6))

average_metrics.plot(kind='bar')

plt.title("Average Fitness Metrics")
plt.ylabel("Average Values")

plt.savefig("graph_9_average_metrics.png",
            dpi=300,
            bbox_inches='tight')

plt.show()

# 10. CORRELATION HEATMAP
plt.figure(figsize=(10,8))

correlation = df[
    ['Age',
     'Steps',
     'CaloriesBurned',
     'HoursSleep',
     'WaterIntake',
     'WorkoutHours',
     'HeartRate',
     'BMI']
].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm'
)

plt.title("Fitness Dataset Correlation Heatmap")

plt.savefig("graph_10_heatmap.png",
            dpi=300,
            bbox_inches='tight')

plt.show()

# 11. PAIR PLOT
sns.pairplot(
    df[
        ['Steps',
         'CaloriesBurned',
         'WorkoutHours',
         'HeartRate',
         'BMI']
    ]
)

plt.savefig("graph_11_pairplot.png",
            dpi=300,
            bbox_inches='tight')

plt.show()

# 12. VIOLIN PLOT
df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[10,20,30,40,50,60],
    labels=['10-20',
            '20-30',
            '30-40',
            '40-50',
            '50-60']
)

plt.figure(figsize=(10,6))

sns.violinplot(
    x='AgeGroup',
    y='CaloriesBurned',
    data=df
)

plt.title("Calories Burned Across Age Groups")

plt.savefig("graph_12_violin.png",
            dpi=300,
            bbox_inches='tight')

plt.show()

# 13. BUBBLE CHART
plt.figure(figsize=(12,7))

sns.scatterplot(
    x='WorkoutHours',
    y='Steps',
    size='CaloriesBurned',
    hue='HeartRate',
    data=df,
    sizes=(20,400)
)

plt.title("Daily Fitness Activity Bubble Chart")

plt.savefig("graph_13_bubble_chart.png",
            dpi=300,
            bbox_inches='tight')

plt.show()

# 14. RADAR CHART
metrics = [
    'Steps',
    'CaloriesBurned',
    'WorkoutHours',
    'WaterIntake',
    'HeartRate'
]

values = df[metrics].mean().values

angles = np.linspace(
    0,
    2 * np.pi,
    len(metrics),
    endpoint=False
)

values = np.concatenate(
    (values, [values[0]])
)

angles = np.concatenate(
    (angles, [angles[0]])
)

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111, polar=True)

ax.plot(angles, values)

ax.fill(angles, values, alpha=0.3)

ax.set_xticks(angles[:-1])

ax.set_xticklabels(metrics)

plt.title("Average Fitness Metrics Radar Chart")

plt.savefig("graph_14_radar_chart.png",
            dpi=300,
            bbox_inches='tight')

plt.show()

# 15. HEATMAP BY GENDER
gender_avg = df.groupby('Gender')[
    ['Steps',
     'CaloriesBurned',
     'WorkoutHours',
     'HeartRate']
].mean()

plt.figure(figsize=(8,5))

sns.heatmap(
    gender_avg,
    annot=True,
    cmap='YlGnBu'
)

plt.title("Average Fitness Metrics by Gender")

plt.savefig("graph_15_gender_heatmap.png",
            dpi=300,
            bbox_inches='tight')

plt.show()
