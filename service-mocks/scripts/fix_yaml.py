import sys
import os
import yaml

def fix_yaml_dir(apis_dir):
    for fname in sorted(os.listdir(apis_dir)):
        if not fname.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(apis_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data is None:
                print(f"[WARN] {fname} is empty, skipping")
                continue
            print(f"[OK] {fname} — valid YAML, {len(data.get('paths', {}))} paths")
        except yaml.YAMLError as e:
            print(f"[ERROR] {fname}: {e}")

if __name__ == "__main__":
    apis_dir = sys.argv[1] if len(sys.argv) > 1 else "/apis"
    fix_yaml_dir(apis_dir)
