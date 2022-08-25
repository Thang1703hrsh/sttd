from inspect import stack
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.express as px


df = pd.read_csv("Feasable_Solution.csv")
df = df[df["ST (Minutes)"] != 0]
df = df.reset_index(drop = True)

df["Task Number"] = pd.to_numeric(df["Task Number"])

max_work = max(df["Workstation"])
max_task = max(df["Task Number"])
arr = [[0 for i in range(max_task)] for j in range(max_work)]

df = df.sort_values(by=['Task Number'])

df = df.reset_index(drop = True)
 
fig = px.bar(df, x="Workstation", y="ST (Minutes)", color="Task Number",
             barmode = 'stack' , text = "Task Number")
print(df)
fig.show()
# print(df['ST (Minutes)'][32])

# for i in range(1, max_work + 1):
#     for j in range(1, max_task + 1):
#         for k in range(0, max_task):
#             if (df['Workstation'][k] == i) & (df['Task Number'][k] == j):
#                 arr[i-1][j-1] = df['ST (Minutes)'][k]

# df_new = pd.DataFrame(arr , index = [f'Work {i}' for i in range(0 , max_work)] , columns= [f'Task {j}' for j in range(0 , max_task)])

# create stacked bar chart for monthly temperatures
# df_new.plot(kind='bar', stacked=True, color= auto)
 
# # labels for x & y axis
# plt.xlabel('Months')
# plt.ylabel('Temp ranges in Degree Celsius')
 
# # title of plot
# plt.title('Monthly Temperatures in a year')

# plt.show()




# # a = []
# # for i in range(1 , max_work+1):
# #     if(df["Workstation"] == i):
# #         a[i] = 

# # plt.show()

