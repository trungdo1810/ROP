import cv2
import numpy as np
import os

def crop_black_borders(image):
    # Trích xuất kênh màu xanh lá
    green_channel = image[:, :, 1]
    
    # Áp dụng phép mở rộng (morphological closing) để làm đầy các vùng hở
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(green_channel, cv2.MORPH_CLOSE, kernel)
    
    # Áp dụng phép giãn nở để loại bỏ nhiễu nhỏ
    dilated = cv2.dilate(closed, kernel, iterations=3)
    
    # Áp dụng threshold để nhị phân hóa ảnh
    _, thresh = cv2.threshold(dilated, 20, 255, cv2.THRESH_BINARY)
    
    # Loại bỏ nhiễu bằng phép mở (Opening)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.erode(cleaned, kernel, iterations=4)
    
    # Tìm contours để xác định vùng chứa thông tin
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Không tìm thấy vùng chứa thông tin, trả về ảnh gốc")
        return image
    
    # Lọc bỏ contours có diện tích nhỏ để tránh nhiễu
    min_area = 2000  # Tăng giá trị để lọc nhiễu mạnh hơn
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not contours:
        print("Không có vùng đủ lớn để crop, trả về ảnh gốc")
        return image, []
    
    # Tạo bounding box quanh vùng có thông tin lớn nhất
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cropped = image[y:y+h, x:x+w]
    
    return cropped

def process_dataset(dataset_path):
    categories = ["normal", "preplus", "plus"]
    
    for category in categories:
        input_folder = os.path.join(dataset_path, category)
        output_folder = os.path.join(dataset_path, f"crop_{category}")
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        os.makedirs(output_folder, exist_ok=True)
        
        count = 0  # Biến đếm số lượng ảnh đã crop
        
        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Đọc ảnh
            image = cv2.imread(input_path)
            if image is None:
                continue
            
            # Crop ảnh
            cropped_image = crop_black_borders(image)
            
            # Lưu ảnh đã crop
            cv2.imwrite(output_path, cropped_image)
            count += 1
        
        print(f"Đã crop {count} ảnh từ thư mục {category}")

# Gọi hàm xử lý toàn bộ dataset
dataset_path = r"C:\Users\trung\Documents\AI\ROP-o\Triplet-data\data"  # Thay bằng đường dẫn thực tế
process_dataset(dataset_path)