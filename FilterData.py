from PIL import Image
import os

def delete_small_images(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                if width < 30 or height < 30:
                    img.close()
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
        except (IOError, SyntaxError) as e:
            print(f"Error processing {file_path}: {e}")

# Đường dẫn tới thư mục chứa ảnh
image_directory = "./data_classify/valid/1"

# Gọi hàm để xóa những file ảnh nhỏ
delete_small_images(image_directory)