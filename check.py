with open("surveillance/views.py") as f:
    lines = f.readlines()

# Find and remove duplicate block starting at line 148 (index 147)
# Check what lines 147-163 contain
print("Lines 147-163:")
for i,l in enumerate(lines[146:163],147):
    print(i, repr(l))
