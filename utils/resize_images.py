#!/usr/bin/env python3

import os
import sys
from PIL import Image
import glob

def resize_with_padding(input_path, output_path, target_size=(1024, 576), padding_color=(255, 255, 255)):
    """
    Resize image to fit within target dimensions and pad with specified color.
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to save output image
        target_size (tuple): Target dimensions (width, height)
        padding_color (tuple): RGB color for padding (default: white)
    """
    try:
        # Open the image
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (handles RGBA, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate resize dimensions to fit within target while maintaining aspect ratio
            img_ratio = img.width / img.height
            target_ratio = target_size[0] / target_size[1]
            
            if img_ratio > target_ratio:
                # Image is wider, fit to width
                new_width = target_size[0]
                new_height = int(target_size[0] / img_ratio)
            else:
                # Image is taller, fit to height
                new_height = target_size[1]
                new_width = int(target_size[1] * img_ratio)
            
            # Resize the image
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create a new image with target dimensions and padding color
            padded_img = Image.new('RGB', target_size, padding_color)
            
            # Calculate position to center the resized image
            x_offset = (target_size[0] - new_width) // 2
            y_offset = (target_size[1] - new_height) // 2
            
            # Paste the resized image onto the padded background
            padded_img.paste(resized_img, (x_offset, y_offset))
            
            # Save the result
            padded_img.save(output_path, 'PNG', optimize=True)
            return True
            
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def main():
    # Get directory from command line argument or use current directory
    source_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    # Check if directory exists
    if not os.path.isdir(source_dir):
        print(f"Error: Directory '{source_dir}' does not exist.")
        sys.exit(1)
    
    # Find all PNG files in the directory
    png_pattern = os.path.join(source_dir, '*.png')
    png_files = glob.glob(png_pattern)
    
    if not png_files:
        print(f"No PNG files found in {source_dir}")
        return
    
    print(f"Found {len(png_files)} PNG files in: {source_dir}")
    print("Processing files...")
    
    processed = 0
    failed = 0
    
    for input_file in png_files:
        # Get filename without extension
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(source_dir, f"{base_name}_resized.png")
        
        print(f"Processing: {os.path.basename(input_file)} -> {os.path.basename(input_file)}")
        
        if resize_with_padding(input_file, input_file):
            print(f"✓ Successfully processed: {base_name}")
            processed += 1
        else:
            print(f"✗ Failed to process: {base_name}")
            failed += 1
    
    print(f"\nCompleted! Processed {processed} files successfully, {failed} failures.")

if __name__ == "__main__":
    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow (PIL) is not installed.")
        print("Install it with: pip install Pillow")
        sys.exit(1)
    
    main()