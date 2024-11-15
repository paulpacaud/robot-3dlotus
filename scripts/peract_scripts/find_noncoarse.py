import json

with open('../../assets/taskvars_target_label_zrange_peract.json', 'r') as f:
    labels = json.load(f)

coarse_count = 0
fine_count = 0
for task in labels:
    for episode in labels[task]:
        for plan in labels[task][episode]:
            obj_has_coarse = False
            obj_has_fine = False
            if 'object' in plan:
                object = plan['object']
            if 'target' in plan:
                object = plan['target']
            if 'coarse' in object:
                coarse_count += 1
                obj_has_coarse = True
            if 'fine' in object:
                fine_count += 1
                obj_has_fine = True
            if obj_has_fine and not obj_has_coarse:
                print(f"Task: {task}, Episode: {episode}, Plan: {plan}")

print(f"Coarse count: {coarse_count}")
print(f"Fine count: {fine_count}")
