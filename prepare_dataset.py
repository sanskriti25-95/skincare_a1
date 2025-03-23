import os
import shutil
from sklearn.model_selection import train_test_split
import glob

def prepare_dataset(source_dir, dest_dir):
    # Create main directories
    train_dir = os.path.join(dest_dir, 'train')
    valid_dir = os.path.join(dest_dir, 'validation')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    
    # Process each class (oily, dry, normal)
    classes = ['oily', 'dry', 'normal']
    
    # First, create the class directories
    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)
    
    # Copy training data
    source_train = os.path.join(source_dir, 'train')
    for class_name in classes:
        source_class_dir = os.path.join(source_train, class_name)
        if os.path.exists(source_class_dir):
            # Copy all images from source train directory
            images = glob.glob(os.path.join(source_class_dir, '*.[jJ][pP][gG]')) + \
                    glob.glob(os.path.join(source_class_dir, '*.[jJ][pP][eE][gG]')) + \
                    glob.glob(os.path.join(source_class_dir, '*.[pP][nN][gG]'))
            
            for img in images:
                shutil.copy2(img, os.path.join(train_dir, class_name))
            print(f"Copied {len(images)} training images for {class_name}")
    
    # Copy validation data
    source_valid = os.path.join(source_dir, 'valid')
    for class_name in classes:
        source_class_dir = os.path.join(source_valid, class_name)
        if os.path.exists(source_class_dir):
            # Copy all images from source validation directory
            images = glob.glob(os.path.join(source_class_dir, '*.[jJ][pP][gG]')) + \
                    glob.glob(os.path.join(source_class_dir, '*.[jJ][pP][eE][gG]')) + \
                    glob.glob(os.path.join(source_class_dir, '*.[pP][nN][gG]'))
            
            for img in images:
                shutil.copy2(img, os.path.join(valid_dir, class_name))
            print(f"Copied {len(images)} validation images for {class_name}")

if __name__ == "__main__":
    # Update with your specific dataset path
    source_directory = r"C:\Users\shaur\OneDrive\Desktop\Oily-Dry-Skin-Dataset"
    destination_directory = "dataset"
    
    print(f"Source directory: {source_directory}")
    print(f"Checking if source directory exists: {os.path.exists(source_directory)}")
    
    # Print the structure of the source directory
    print("\nSource directory structure:")
    for root, dirs, files in os.walk(source_directory):
        level = root.replace(source_directory, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        if files:
            subindent = ' ' * 4 * (level + 1)
            for f in files[:3]:  # Show only first 3 files
                print(f"{subindent}{f}")
            if len(files) > 3:
                print(f"{subindent}... ({len(files)} files total)")
    
    prepare_dataset(source_directory, destination_directory)
    print("\nDataset preparation completed!") 