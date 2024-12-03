import json
import numpy as np

set = "test"
output_file_path = f"../../prompts/rlbench/in_context_examples_{set}_peract.txt"

with open(f"../../assets/taskvars_{set}_peract.json", "r") as f:
    taskvars_val = json.load(f)
with open("../../assets/peract/taskvars_target_label_zrange_peract.json", "r") as f:
    labels = json.load(f)
with open("../../assets/peract/taskvars_instructions_peract.json", "r") as f:
    taskvars_instr = json.load(f)
with open(f"../../assets/objects_peract_{set}.json", "r") as f:
    taskvar_obj = json.load(f)

if set == "train":
    with open(f"../../assets/objects_peract_val.json", "r") as f:
        taskvar_obj = json.load(f)

    taskvar_obj_dict = {}
    for task in taskvars_val:
        taskvar_obj_list = []
        for episode in taskvar_obj[task].keys():
            objs = taskvar_obj[task][episode]
            for obj in objs:
                if obj not in taskvar_obj_list:
                    taskvar_obj_list.append(obj)
        # remove empty string
        taskvar_obj_list = [obj for obj in taskvar_obj_list if obj]
        taskvar_obj_dict[task] = taskvar_obj_list

    with open(f"../../assets/objects_peract_test.json", "r") as f:
        taskvar_obj = json.load(f)

    for task in taskvars_val:
        if len(taskvar_obj_dict[task]) != 0:
            continue
        taskvar_obj_list = []
        for episode in taskvar_obj[task].keys():
            objs = taskvar_obj[task][episode]
            for obj in objs:
                if obj not in taskvar_obj_list:
                    taskvar_obj_list.append(obj)
        # remove empty string
        taskvar_obj_list = [obj for obj in taskvar_obj_list if obj]
        taskvar_obj_dict[task] = taskvar_obj_list
else:
    with open(f"../../assets/objects_peract_{set}.json", "r") as f:
        taskvar_obj = json.load(f)

    taskvar_obj_dict = {}
    for task in taskvars_val:
        taskvar_obj_list = []
        for episode in taskvar_obj[task].keys():
            objs = taskvar_obj[task][episode]
            for obj in objs:
                if obj not in taskvar_obj_list:
                    taskvar_obj_list.append(obj)
        # remove empty string
        taskvar_obj_list = [obj for obj in taskvar_obj_list if obj]
        taskvar_obj_dict[task] = taskvar_obj_list

# "close_jar_peract+0": {"episode0": {"red jar": [81], "lime jar": [85], "gray lid": [87]}, "episode38": {"red jar": [81], "magenta jar": [85], "gray lid": [87]}, "episode39": {"red jar": [81], "cyan jar": [85], "gray lid": [87]}, "episode59": {"red jar": [81], "lime jar": [85], "gray lid": [87]}}, "close_jar_peract+1": {"episode7": {"white jar": [81], "maroon jar": [85], "gray lid": [87]}, "episode12": {"magenta jar": [81], "maroon jar": [85], "gray lid": [87]}, "episode51": {"gray jar": [81], "maroon jar": [85], "gray lid": [87]}, "episode77": {"green jar": [81], "maroon jar": [85], "gray lid": [87]}, "episode79": {"silver jar": [81], "maroon jar": [85], "gray lid": [87]}, "episode82": {"rose jar": [81], "maroon jar": [85], "gray lid": [87]}}, "close_jar_peract+2": {"episode2": {"lime jar": [81], "violet jar": [85], "gray lid": [87]}, "episode42": {"lime jar": [81], "orange jar": [85], "gray lid": [87]}, "episode45": {"lime jar": [81], "silver jar": [85], "gray lid": [87]}, "episode58": {"lime jar": [81], "black jar": [85], "gray lid": [87]}, "episode73": {"lime jar": [81], "maroon jar": [85], "gray lid": [87]}, "episode74": {"lime jar": [81], "silver jar": [85], "gray lid": [87]}, "episode97": {"lime jar": [81], "blue jar": [85], "gray lid": [87]}}, "close_jar_peract+3": {"episode1": {"violet jar": [81], "green jar": [85], "gray lid": [87]}, "episode11": {"silver jar": [81], "green jar": [85], "gray lid": [87]}, "episode49": {"lime jar": [81], "green jar": [85], "gray lid": [87]}, "episode71": {"blue jar": [81], "green jar": [85], "gray lid": [87]}, "episode72": {"yellow jar": [81], "green jar": [85], "gray lid": [87]}, "episode85": {"lime jar": [81], "green jar": [85], "gray lid": [87]}}, "close_jar_peract+4": {"episode64": {"blue jar": [81], "orange jar": [85], "gray lid": [87]}, "episode78": {"blue jar": [81], "olive jar": [85], "gray lid": [87]}}, "close_jar_peract+5": {"episode17": {"violet jar": [81], "navy jar": [85], "gray lid": [87]}, "episode24": {"maroon jar": [81], "navy jar": [85], "gray lid": [87]}, "episode66": {"red jar": [81], "navy jar": [85], "gray lid": [87]}, "episode88": {"magenta jar": [81], "navy jar": [85], "gray lid": [87]}, "episode93": {"green jar": [81], "navy jar": [85], "gray lid": [87]}}, "close_jar_peract+6": {"episode16": {"yellow jar": [81], "purple jar": [85], "gray lid": [87]}, "episode63": {"yellow jar": [81], "azure jar": [85], "gray lid": [87]}}, "close_jar_peract+7": {"episode5": {"silver jar": [81], "cyan jar": [85], "gray lid": [87]}, "episode13": {"azure jar": [81], "cyan jar": [85], "gray lid": [87]}}, "close_jar_peract+8": {"episode6": {"magenta jar": [81], "silver jar": [85], "gray lid": [87]}, "episode29": {"magenta jar": [81], "black jar": [85], "gray lid": [87]}, "episode68": {"magenta jar": [81], "lime jar": [85], "gray lid": [87]}}, "close_jar_peract+9": {"episode50": {"red jar": [81], "silver jar": [85], "gray lid": [87]}, "episode55": {"blue jar": [81], "silver jar": [85], "gray lid": [87]}, "episode81": {"white jar": [81], "silver jar": [85], "gray lid": [87]}, "episode84": {"orange jar": [81], "silver jar": [85], "gray lid": [87]}}
task_plan = {}
task_obj = {}
for task in taskvars_val:
    for episode in labels[task].keys():
        plans = labels[task][episode]
        task_plan[task] = plans
        break
    task_obj_list = []
    for episode in labels[task].keys():
        plans = labels[task][episode]
        for plan in plans:
            if "object" not in plan:
                continue
            obj = plan["object"]["name"]
            if obj not in task_obj_list:
                task_obj_list.append(obj)
    task_obj[task] = task_obj_list

with open(output_file_path, "w") as f:
    manipulation_actions = ["grasp", "move grasped object", "rotate grasped object"]
    for task in task_plan.keys():
        task_instr_list = taskvars_instr[task]
        # randomly select one instruction
        random_id = np.random.randint(0, len(task_instr_list))
        task_instr = task_instr_list[random_id]
        obj_list = taskvar_obj_dict[task]

        f.write(f"# taskvar: {task}\n")
        f.write(f"# query: {task_instr}.\n")
        f.write(f"# objects = {obj_list}\n")

        for i, plan in enumerate(task_plan[task]):
            if plan["action"] == "grasp":
                obj_name = plan["object"]["name"].replace(" ", "_")
                f.write(f"{obj_name} = grasp(object=\"{plan['object']['name']}\")\n")
                past_obj = obj_name
            elif plan["action"] == "move grasped object":
                target = plan["target"]["name"]
                if target == "bottom shelf":
                    target = "cupboard"
                f.write(
                    f"{past_obj} = move_grasped_object(target=\"{plan['target']['name']}\")\n"
                )

                # Check if there's a next action and if it's not rotate_grasped_object
                if (
                    i == len(task_plan[task]) - 1
                    or task_plan[task][i + 1]["action"] != "rotate grasped object"
                ):
                    f.write("release()\n")
            elif plan["action"] == "rotate grasped object":
                f.write(f"{past_obj} = rotate_grasped_object()\n")
            elif plan["action"] == "push down":
                obj_name = plan["object"]["name"].replace(" ", "_")
                f.write(
                    f"{obj_name} = push_down(object=\"{plan['object']['name']}\")\n"
                )
            elif plan["action"] == "push forward":
                obj_name = plan["object"]["name"].replace(" ", "_")
                f.write(
                    f"{obj_name} = push_forward(object=\"{plan['object']['name']}\", target=\"{plan['target']['name']}\")\n"
                )
            elif plan["action"] == "move grasped object out":
                f.write(f'{past_obj} = move_grasped_object(target="out")\n')
                if (
                    i == len(task_plan[task]) - 1
                    or task_plan[task][i + 1]["action"] != "rotate grasped object"
                ):
                    f.write("release()\n")

        if plan["action"] in manipulation_actions:
            if (
                plan["action"] != "move grasped object"
                and plan["action"] != "move grasped object out"
            ):
                f.write("release()\n")
        f.write("# done\n\n")

print(f"Plans have been written to {output_file_path}")
