def parse_plan(content):
    plans = {}
    current_taskvar = None
    current_plan = []

    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith('# taskvar:'):
            if current_taskvar:
                plans[current_taskvar] = current_plan
            current_taskvar = line.split('# taskvar:')[1].strip()
            current_plan = []
        elif not line.startswith('#'):
            current_plan.append(line)

    if current_taskvar:
        plans[current_taskvar] = current_plan

    return plans


def compare_plans(ground_truth_content, generated_content):
    gt_plans = parse_plan(ground_truth_content)
    gen_plans = parse_plan(generated_content)

    # Compare coverage
    gt_tasks = set(gt_plans.keys())
    gen_tasks = set(gen_plans.keys())

    missing_tasks = gt_tasks - gen_tasks
    extra_tasks = gen_tasks - gt_tasks
    common_tasks = gt_tasks.intersection(gen_tasks)

    # Compare plans for common tasks
    differences = []
    for task in common_tasks:
        gt_plan = gt_plans[task]
        gen_plan = gen_plans[task]

        if gt_plan != gen_plan:
            differences.append({
                'task': task,
                'ground_truth': gt_plan,
                'generated': gen_plan
            })

    return {
        'statistics': {
            'total_gt_tasks': len(gt_tasks),
            'total_gen_tasks': len(gen_tasks),
            'missing_tasks': len(missing_tasks),
            'extra_tasks': len(extra_tasks),
            'matching_tasks': len(common_tasks),
            'different_plans': len(differences)
        },
        'missing_tasks': list(missing_tasks),
        'extra_tasks': list(extra_tasks),
        'differences': differences
    }


# Example usage:
# with open('../prompts/rlbench/in_context_examples.txt', 'r') as f:
#     gt_content = f.read()
with open('../prompts/rlbench/in_context_examples_peract.txt', 'r') as f:
    gt_content = f.read()

with open('../prompts/rlbench/in_context_examples_peract.txt', 'r') as f:
    gen_content = f.read()

results = compare_plans(gt_content, gen_content)

print("\nComparison Statistics:")
print(f"Ground Truth Tasks: {results['statistics']['total_gt_tasks']}")
print(f"Generated Tasks: {results['statistics']['total_gen_tasks']}")
print(f"Missing Tasks: {results['statistics']['missing_tasks']}")
print(f"Extra Tasks: {results['statistics']['extra_tasks']}")
print(f"Matching Tasks: {results['statistics']['matching_tasks']}")
print(f"Tasks with Different Plans: {results['statistics']['different_plans']}")

if results['missing_tasks']:
    print("\nMissing Tasks:")
    for task in results['missing_tasks']:
        print(f"- {task}")

if results['extra_tasks']:
    print("\nExtra Tasks:")
    for task in results['extra_tasks']:
        print(f"- {task}")

if results['differences']:
    print("\nDifferent Plans:")
    for diff in results['differences']:
        print(f"\nTask: {diff['task']}")
        print("Ground Truth:")
        for line in diff['ground_truth']:
            print(f"  {line}")
        print("Generated:")
        for line in diff['generated']:
            print(f"  {line}")