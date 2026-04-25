import os

with open("seed_data.txt", "r") as f:
    seed_content = f.read()

with open("database/schema.sql", "r") as f:
    schema = f.read()

# We want to replace the PYQ Resources block.
# Find the start of the block.
start_marker = "-- PYQ Resources\n"
end_marker = "\n-- University Info\n"

start_idx = schema.find(start_marker)
end_idx = schema.find(end_marker)

if start_idx != -1 and end_idx != -1:
    new_schema = schema[:start_idx + len(start_marker)] + seed_content + schema[end_idx:]
    with open("database/schema.sql", "w") as f:
        f.write(new_schema)
    print("Schema updated!")
else:
    print("Markers not found.")
