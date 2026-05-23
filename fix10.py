with open("surveillance/views.py") as f:
    lines = f.readlines()

# Remove lines 136-147 (index 135-146) - duplicate urllib block
new_lines = lines[:135] + lines[147:]

with open("surveillance/views.py", "w") as f:
    f.writelines(new_lines)

# Verify
print("Verifying lines 119-155:")
with open("surveillance/views.py") as f:
    lines2 = f.readlines()
for i,l in enumerate(lines2[118:150],119):
    print(i,l,end='')
