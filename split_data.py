import os
import shutil
import random

# Paths to directories
base_dir = "/home/vmulukuri/Seminar_Impl/GNR638_Assgn_1_Bag_of_Words_Classification/UCMerced_LandUse/Images"
train_dir = "/home/vmulukuri/Seminar_Impl/GNR638_Assgn_1_Bag_of_Words_Classification/data/train"
test_dir = "/home/vmulukuri/Seminar_Impl/GNR638_Assgn_1_Bag_of_Words_Classification/data/test"

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Set the split ratio
train_ratio = 0.8

# Iterate through each class directory
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    
    if os.path.isdir(class_path):
        # Create corresponding class directories in train and test
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # List all images in the class directory
        images = os.listdir(class_path)
        
        # Shuffle images
        random.shuffle(images)
        
        # Split images into train and test sets
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Move images to the respective directories
        for img in train_images:
            src = os.path.join(class_path, img)
            dest = os.path.join(train_class_dir, img)
            shutil.copy(src, dest)
        
        for img in test_images:
            src = os.path.join(class_path, img)
            dest = os.path.join(test_class_dir, img)
            shutil.copy(src, dest)

print("Dataset split into training and testing sets successfully!")
