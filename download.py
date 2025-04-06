import gdown

# Thay bằng link chia sẻ đầy đủ của bạn
url = "https://drive.google.com/file/d/1AF_vaqjV_advfrVjN3VCukGLpGd5D5-o/view?usp=sharing"
output = "datasets.zip"

# Tải file
gdown.download(url, output, quiet=False, fuzzy=True)

# Kiểm tra kích thước file
import os
file_size = os.path.getsize(output)
print(f"File size: {file_size / 1024**2:.2f} MB")

print("Finished!")