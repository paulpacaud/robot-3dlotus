import os

dataset_dir = '../data/peract/test_dataset/microsteps/seed200/'
taskvars_val_peract = []
task_val_peract = os.listdir(dataset_dir)

# remove .zip files from the list
task_val_peract = [task for task in task_val_peract if not task.endswith('.zip') and not task.endswith('.sh')]

for task in task_val_peract:
    variations_list = os.listdir(f'{dataset_dir}{task}')
    for variation in variations_list:
        variation_id = variation[9:]
        taskvars_val_peract.append(f'{task}+{variation_id}')

# save the taskvars_val_peract as a json
import json

with open('../../assets/taskvars_test_peract.json', 'w') as f:
    json.dump(taskvars_val_peract, f, indent=4)
print("taskvars_test_peract.json saved")