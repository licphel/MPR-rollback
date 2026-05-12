import json
from collections import Counter

# Load dataset
with open('trajectory/real_trajectories.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# --- Basic counts ---
n_total = len(data)
n_violated = sum(1 for t in data if t['violated'])
n_safe = n_total - n_violated

# --- Trajectory length statistics ---
all_lengths = [len(t['steps']) for t in data]
vio_lengths = [len(t['steps']) for t in data if t['violated']]
safe_lengths = [len(t['steps']) for t in data if not t['violated']]

# --- Violation step counts (violated trajectories only) ---
vio_step_counts = []
for t in data:
    if t['violated']:
        n_vio = sum(1 for s in t['steps'] if s.get('violated', False))
        vio_step_counts.append(n_vio)

# --- Task category distribution ---
category_total = Counter(t.get('task_category', 'unknown') for t in data)
category_vio = Counter(t.get('task_category', 'unknown') for t in data if t['violated'])

# --- Tool frequency in all steps vs violated steps ---
tool_all = Counter()
tool_vio = Counter()
for t in data:
    for s in t['steps']:
        tool = s['action']['tool']
        tool_all[tool] += 1
        if s.get('violated', False):
            tool_vio[tool] += 1

# --- Violation type: aggregate tools into categories ---
def classify_violation_type(tool):
    if tool in ['send_email', 'upload_file', 'post_webhook', 'http_request', 'send_sms']:
        return 'exfiltration'
    elif tool in ['read_file', 'list_files', 'search_files']:
        return 'sensitive_read'
    elif tool in ['database_query', 'sql_query', 'run_query']:
        return 'database_query'
    elif tool in ['execute_command', 'run_terminal', 'shell_exec']:
        return 'infrastructure_access'
    else:
        return f'other ({tool})'

vio_type_counter = Counter()
for t in data:
    for s in t['steps']:
        if s.get('violated', False):
            vtype = classify_violation_type(s['action']['tool'])
            vio_type_counter[vtype] += 1

# --- Print report ---
print("=" * 60)
print("DATASET CHARACTERIZATION")
print("=" * 60)
print(f"Total trajectories:     {n_total}")
print(f"  Violated:             {n_violated} ({100*n_violated/n_total:.1f}%)")
print(f"  Safe:                 {n_safe} ({100*n_safe/n_total:.1f}%)")
print()

print("--- Trajectory Length ---")
print(f"All trajectories:       min={min(all_lengths)}, max={max(all_lengths)}, "
      f"mean={sum(all_lengths)/n_total:.1f}, median={sorted(all_lengths)[n_total//2]}")
print(f"Violated:               min={min(vio_lengths)}, max={max(vio_lengths)}, "
      f"mean={sum(vio_lengths)/n_violated:.1f}, median={sorted(vio_lengths)[n_violated//2]}")
print(f"Safe:                   min={min(safe_lengths)}, max={max(safe_lengths)}, "
      f"mean={sum(safe_lengths)/n_safe:.1f}, median={sorted(safe_lengths)[n_safe//2]}")
print()

print("--- Violation Steps per Violated Trajectory ---")
print(f"Count:                  min={min(vio_step_counts)}, max={max(vio_step_counts)}, "
      f"mean={sum(vio_step_counts)/n_violated:.1f}, median={sorted(vio_step_counts)[n_violated//2]}")
# Distribution
vio_dist = Counter(vio_step_counts)
print("Distribution:")
for k in sorted(vio_dist):
    print(f"  {k} step(s): {vio_dist[k]} trajectories ({100*vio_dist[k]/n_violated:.1f}%)")
print()

print("--- Task Categories ---")
print(f"{'Category':<25} {'Total':>6} {'Violated':>10} {'Rate':>8}")
print("-" * 49)
for cat in sorted(category_total):
    tot = category_total[cat]
    vio = category_vio.get(cat, 0)
    print(f"{cat:<25} {tot:>6} {vio:>10} {100*vio/tot:>7.1f}%")
print()

print("--- Violation Type Breakdown ---")
total_vio_steps = sum(vio_type_counter.values())
print(f"{'Type':<25} {'Count':>6} {'%':>8}")
print("-" * 39)
for vtype, count in vio_type_counter.most_common():
    print(f"{vtype:<25} {count:>6} {100*count/total_vio_steps:>7.1f}%")
print()

print("--- Example Violated Trajectory ---")
vio_traj = next(t for t in data if t['violated'])
print(f"ID: {vio_traj['trajectory_id']}")
print(f"Category: {vio_traj.get('task_category', 'N/A')}")
print(f"Task: {vio_traj.get('task_description', 'N/A')[:120]}...")
print(f"Steps: {len(vio_traj['steps'])}")
print(f"Violation steps: {sum(1 for s in vio_traj['steps'] if s.get('violated'))}")
print(f"Violated tools: {[s['action']['tool'] for s in vio_traj['steps'] if s.get('violated')]}")
print()

print("--- Example Safe Trajectory ---")
safe_traj = next(t for t in data if not t['violated'])
print(f"ID: {safe_traj['trajectory_id']}")
print(f"Category: {safe_traj.get('task_category', 'N/A')}")
print(f"Task: {safe_traj.get('task_description', 'N/A')[:120]}...")
print(f"Steps: {len(safe_traj['steps'])}")