with open("surveillance/views.py") as f:
    lines = f.readlines()

# Remove lines 148-161 (index 147-160)
new_lines = lines[:147] + lines[161:]

with open("surveillance/views.py", "w") as f:
    f.writelines(new_lines)

print("Done! Verifying...")
with open("surveillance/views.py") as f:
    lines2 = f.readlines()
for i,l in enumerate(lines2[130:152],131):
    print(i, l, end="")
