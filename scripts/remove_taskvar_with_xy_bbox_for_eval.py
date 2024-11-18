import json

with open('../assets/taskvars_target_label_zrange_peract.json', 'r') as f:
    labels = json.load(f)

with open('../assets/taskvars_test_peract.json', 'r') as f:
    taskvars = json.load(f)

taskvar_val_new = []
taskvar_to_remove = []
for task in taskvars:
    task_labels = labels[task]
    for episode in task_labels:
        plans = task_labels[episode]
        for plan in plans:
            if 'target' in plan and 'xy_bbox' in plan['target']:
                taskvar_to_remove.append(task)

taskvar_val_new = [task for task in taskvars if task not in taskvar_to_remove]

import json

print(f"Former taskvars_test_peract_wo_bbox.json length: {len(taskvars)}")
print(f"New taskvars_test_peract_wo_bbox.json length: {len(taskvar_val_new)}")
print(f"Taskvars removed: {taskvar_to_remove}")

with open('taskvars_test_peract_wo_bbox.json', 'w') as f:
    json.dump(taskvar_val_new, f, indent=4)
print("taskvars_test_peract_wo_bbox.json saved")