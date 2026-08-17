path = "demo/threaded_pipeline.py"
with open(path, "r") as f:
    content = f.read()

old_block = '''    def get_best(self):
    def set_start_execution(self):
        with self.lock:
            self.start_execution = True

    def can_execute(self):
        with self.lock:
            return self.start_execution
        with self.lock:
            return self.best_theta, self.best_goal_err
'''

new_block = '''    def get_best(self):
        with self.lock:
            return self.best_theta, self.best_goal_err

    def set_start_execution(self):
        with self.lock:
            self.start_execution = True

    def can_execute(self):
        with self.lock:
            return self.start_execution
'''

if old_block not in content:
    print("FIX SKIPPED: exact broken block not found -- no changes made.")
else:
    content = content.replace(old_block, new_block)
    print("FIX APPLIED: repaired get_best / set_start_execution / can_execute.")

if "self.start_execution = False" in content:
    print("INIT CHECK: self.start_execution = False already present.")
else:
    print("INIT CHECK WARNING: self.start_execution = False NOT found anywhere -- __init__ may be missing it. Will report separately, do not worry yet.")

with open(path, "w") as f:
    f.write(content)

print("Done. Now run: python -m py_compile demo/threaded_pipeline.py")
