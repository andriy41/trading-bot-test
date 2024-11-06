import os

def print_directory_tree(startpath, max_depth=2):
    def tree(current_path, current_depth, indent=''):
        if current_depth > max_depth:
            return
        entries = os.listdir(current_path)
        entries.sort()
        for index, entry in enumerate(entries):
            path = os.path.join(current_path, entry)
            if index == len(entries) - 1:
                connector = '└── '
                new_indent = indent + '    '
            else:
                connector = '├── '
                new_indent = indent + '│   '
            if os.path.isdir(path):
                print(f"{indent}{connector}{entry}/")
                tree(path, current_depth + 1, new_indent)
            else:
                print(f"{indent}{connector}{entry}")

    print(f"{os.path.basename(startpath)}/")
    tree(startpath, 1)

# Set the path to your main folder
startpath = r"C:\Users\space\Desktop\pasta"

# Set the maximum depth level
max_depth = 2

# Call the function
print_directory_tree(startpath, max_depth)
