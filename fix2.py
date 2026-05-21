with open("surveillance/views.py") as f:
    content = f.read()

# Find line 222 area and fix it
lines = content.split("\n")
for i, line in enumerate(lines):
    if "camera is None or" in line and "isOpened" in line:
        print(f"Found at line {i+1}: {line.strip()}")
        lines[i] = line.replace(
            line.strip(),
            "if camera is None:"
        ).rstrip()
        print(f"Fixed to: {lines[i].strip()}")

content = "\n".join(lines)
with open("surveillance/views.py", "w") as f:
    f.write(content)
print("Done!")
