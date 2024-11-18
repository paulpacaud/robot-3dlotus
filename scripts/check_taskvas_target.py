import json

with open('../assets/taskvars_target_label_zrange_peract.json', 'r') as f:
    labels = json.load(f)

with open('../assets/taskvars_peract.json', 'r') as f:
    taskvar = json.load(f)

taskvar_train_peract = []
task_missing = []
for task in taskvar:
    if task in labels.keys():
        print(f"{task} is in both")
        taskvar_train_peract.append(task)
    else:
        task_missing.append(task)

print("Task missing in labels")
print(task_missing)
print("Task train")
print(taskvar_train_peract)