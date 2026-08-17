path = "demo/dashboard_state.py"
with open(path, "r") as f:
    content = f.read()

old_block = '''    def append_ee_path(self, p, cbf_active: bool = False, max_len: int = 2000):
    def push_conflict_marker(self, p, max_len: int = 200):
        with self.lock:
            self.conflict_markers.append(np.asarray(p).copy())
            if len(self.conflict_markers) > max_len:
                self.conflict_markers.pop(0)
        with self.lock:
            self.ee_path.append((np.asarray(p).copy(), cbf_active))
            self.conflict_markers = []  # list of ee positions where a conflict event fired
            if len(self.ee_path) > max_len:
                self.ee_path.pop(0)
'''

new_block = '''    def append_ee_path(self, p, cbf_active: bool = False, max_len: int = 2000):
        with self.lock:
            self.ee_path.append((np.asarray(p).copy(), cbf_active))
            if len(self.ee_path) > max_len:
                self.ee_path.pop(0)

    def push_conflict_marker(self, p, max_len: int = 200):
        with self.lock:
            self.conflict_markers.append(np.asarray(p).copy())
            if len(self.conflict_markers) > max_len:
                self.conflict_markers.pop(0)
'''

if old_block not in content:
    print("FIX 1 SKIPPED: exact broken block not found -- no changes made for this part.")
else:
    content = content.replace(old_block, new_block)
    print("FIX 1 APPLIED: repaired append_ee_path / push_conflict_marker.")

if "self.conflict_markers = []" in content:
    print("FIX 2 SKIPPED: self.conflict_markers = [] already present somewhere.")
else:
    marker = "        self.ee_path = []\n"
    if content.count(marker) != 1:
        print(f"FIX 2 SKIPPED: expected one 'self.ee_path = []' line, found {content.count(marker)}.")
    else:
        content = content.replace(
            marker,
            marker + "        self.conflict_markers = []  # list of ee positions where a conflict event fired\n"
        )
        print("FIX 2 APPLIED: added self.conflict_markers = [] to __init__.")

with open(path, "w") as f:
    f.write(content)

print("Done. Now run: python -m py_compile demo/dashboard_state.py")
