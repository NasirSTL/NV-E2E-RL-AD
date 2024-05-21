def save_first_200_lines(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as f:
        f.writelines(lines[:150])

def replace_path_in_file(file_path, old_path, new_path):
    # Read the content of the file
    with open(file_path, 'r') as file:
        file_contents = file.read()
    
    # Replace the old path with the new path
    updated_contents = file_contents.replace(old_path, new_path).replace('\\', '/')
    
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(updated_contents)

# Define the input file path, old path, and new path
input_file_path = '/home/research/m.owen/ondemand/data/lanenet/lanenet-lane-detection-pytorch/train_val_txt/val.txt'  # Update this to the path of your text file
new_path = '/home/research/m.owen/ondemand/data/lanenet/lanenet-lane-detection-pytorch/data/carla_data'
old_path = r'C:\Users\Owen\Desktop\WindowsNoEditor\lanenet\Carla-Lane-Detection-Dataset-Generation\src'

# Call the function with the specified paths
replace_path_in_file(input_file_path, old_path, new_path)

# Usage
# save_first_200_lines(r'C:\Users\Owen\Desktop\lanenet_and_controller\lanenet-lane-detection-pytorch\train_val_txt\val.txt', 
#                      r'C:\Users\Owen\Desktop\lanenet_and_controller\lanenet-lane-detection-pytorch\train_val_txt\val_small.txt')
