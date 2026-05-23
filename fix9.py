with open("surveillance/views.py") as f:
    lines = f.readlines()

# Remove lines 136 and 137 (index 135 and 136) - old leftover code
# Check what they are first
print("Lines 134-140:")
for i,l in enumerate(lines[133:140],134):
    print(i, repr(l))

# Remove the duplicate frame=None and success=False after our block
new_lines = []
skip_next = False
for i, line in enumerate(lines):
    if i == 135 and line.strip() == "frame = None":
        print(f"Removing line {i+1}: {line.strip()}")
        continue
    if i == 136 and line.strip() == "success = False":
        print(f"Removing line {i+1}: {line.strip()}")
        continue
    new_lines.append(line)

with open("surveillance/views.py", "w") as f:
    f.writelines(new_lines)
print("Done!")
