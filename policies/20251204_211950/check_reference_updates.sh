#!/bin/bash
# Check policy file reference updates

POLICY_FILE="${1:-$(dirname "$0")/WASP_policies.json}"

echo "=========================================="
echo "Policy file reference check"
echo "=========================================="
echo "File: $POLICY_FILE"
echo ""

if [ ! -f "$POLICY_FILE" ]; then
    echo "Error: policy file not found"
    exit 1
fi

python3 << EOF
import json
from pathlib import Path
from datetime import datetime

policy_file = Path("$POLICY_FILE")

with open(policy_file, 'r', encoding='utf-8') as f:
    policies = json.load(f)

print(f"Total policies: {len(policies)}")
print(f"File mtime: {datetime.fromtimestamp(policy_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
print()

total_refs = 0
policies_with_refs = []

for policy in policies:
    policy_id = policy.get('policy_id', 'Unknown')
    refs = policy.get('reference', [])
    if isinstance(refs, list):
        ref_count = len(refs)
        total_refs += ref_count
        if ref_count > 0:
            policies_with_refs.append((policy_id, ref_count))

if policies_with_refs:
    print(f"Policies with references ({len(policies_with_refs)}):")
    for policy_id, count in policies_with_refs:
        print(f"  {policy_id}: {count} reference(s)")
    print()
    print("Latest reference examples:")
    for policy_id, count in policies_with_refs[:3]:
        policy = next((p for p in policies if p.get('policy_id') == policy_id), None)
        if policy and policy.get('reference'):
            latest_ref = policy['reference'][-1]
            print(f"\n  {policy_id} (latest):")
            print(f"    {latest_ref[:200]}...")
else:
    print("All policies have empty reference fields.")
    print("(If you ran the command and still see this, no policy violations were recorded.)")

print()
print(f"Total: {total_refs} reference(s)")
EOF

echo ""
echo "=========================================="
