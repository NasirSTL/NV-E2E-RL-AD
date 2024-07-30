import os
import shutil

read_path = 'gt_image_7_10'
write_path = 'gt_image_7_10_512x256'

if os.path.exists(write_path):
    shutil.rmtree(write_path)

os.mkdir(write_path)

print("Processing images...")
for file in os.listdir(read_path):
    if file == '.DS_Store':
	    continue

    image = cv2.imread(os.path.join(read_path, file))
    image = cv2.resize(image, (512, 256), cv2.INTER_AREA)
    cv2.imwrite(os.path.join(write_path, file), image)

print("Done!")