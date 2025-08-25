import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from matplotlib import pyplot as plt


AUGMENT_TIMES_DEFAULT = 0 # Số lần tăng cường mặc định nếu không được định nghĩa cụ thể

# Cập nhật đường dẫn gốc đến thư mục chứa các nhãn của dataset Alzheimer's
base_original_data = "your path"
# Tạo một thư mục riêng biệt cho tất cả ảnh đã tăng cường của dataset này
base_augmented_data = "your path"
os.makedirs(base_augmented_data, exist_ok=True)

print(f"Đường dẫn dữ liệu gốc: {base_original_data}")
print(f"Đường dẫn lưu dữ liệu tăng cường: {base_augmented_data}")


# Định nghĩa các phép biến đổi tăng cường dữ liệu
augmentation_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
        ),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
    ]
)

total_augmented_count = 0
original_total_count = 0
total_after_augmentation_count = 0

# Lấy danh sách các nhãn từ thư mục con và đếm số lượng ảnh gốc trong mỗi nhãn
labels_info = {}
for d in os.listdir(base_original_data):
    dir_path = os.path.join(base_original_data, d)
    if os.path.isdir(dir_path):
        image_files_in_label = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        labels_info[d] = len(image_files_in_label)

if not labels_info:
    print(f"Không tìm thấy thư mục con nào (nhãn) trong '{base_original_data}'. Vui lòng kiểm tra lại cấu trúc dataset.")
else:
    print("Thông tin số lượng ảnh gốc của các nhãn:")
    for label, count in labels_info.items():
        print(f"- {label}: {count} files")

    target_count_per_label = max(labels_info.values()) # Đặt mục tiêu bằng số lượng của lớp lớn nhất

    print(f"\nSố lượng ảnh mục tiêu cho mỗi nhãn sau tăng cường: {target_count_per_label}")

    augmentation_times_per_label = {}
    for label, current_count in labels_info.items():
        if current_count < target_count_per_label and current_count > 0: # Chỉ tăng cường nếu ít hơn mục tiêu và có ảnh gốc
            # Tính toán số lần tăng cường cần thiết cho mỗi ảnh gốc để đạt được mục tiêu
            num_augment_per_original = (target_count_per_label / current_count) - 1
            augmentation_times_per_label[label] = max(0, round(num_augment_per_original)) # không âm
        else:
            # Nếu đã đạt hoặc vượt mục tiêu, hoặc không có ảnh gốc, không tăng cường thêm
            augmentation_times_per_label[label] = AUGMENT_TIMES_DEFAULT # Có thể là 0 hoặc 1

    print("\nSố lần tăng cường cho mỗi ảnh gốc (tùy theo nhãn):")
    for label, aug_times in augmentation_times_per_label.items():
        print(f"- {label}: {aug_times} lần")

# Lặp qua từng nhãn để tăng cường dữ liệu
for label in labels_info.keys(): # Sử dụng keys từ labels_info để đảm bảo thứ tự
    original_dir_for_label = os.path.join(base_original_data, label)
    augmented_dir_for_label = os.path.join(base_augmented_data, label)

    # Tạo thư mục con cho nhãn này trong thư mục chứa ảnh tăng cường
    os.makedirs(augmented_dir_for_label, exist_ok=True)
    print(f"\n--- Đang xử lý nhãn: {label} ---")

    current_label_original_count = 0
    current_label_augmented_count = 0

    # Lấy số lần tăng cường cụ thể cho nhãn này
    current_augment_times = augmentation_times_per_label.get(label, AUGMENT_TIMES_DEFAULT)

    image_files = [f for f in os.listdir(original_dir_for_label) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if not image_files:
        print(f"Không tìm thấy file ảnh nào trong thư mục '{original_dir_for_label}'. Bỏ qua nhãn này.")
        continue

    for filename in tqdm(image_files, desc=f"Augmenting {label} (x{current_augment_times})"):
        img_path = os.path.join(original_dir_for_label, filename)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Cảnh báo: Không thể mở ảnh {img_path}. Lỗi: {e}")
            continue

        # Lưu ảnh gốc vào thư mục đã tăng cường
        original_save_path = os.path.join(augmented_dir_for_label, filename)
        if not os.path.exists(original_save_path):
            image.save(original_save_path)
        current_label_original_count += 1

        # Áp dụng tăng cường AUGMENT_TIMES_label lần
        for i in range(current_augment_times):
            transformed = augmentation_transform(image)
            base_name, ext = os.path.splitext(filename)
            if not ext.startswith('.'):
                ext = '.' + ext
            save_path = os.path.join(augmented_dir_for_label, f"{base_name}_aug{i}{ext}")
            transformed.save(save_path)
            current_label_augmented_count += 1
            total_augmented_count += 1

    original_total_count += current_label_original_count
    total_after_augmentation_count += (current_label_original_count + current_label_augmented_count)

    print(f"  - Số ảnh gốc của nhãn '{label}': {current_label_original_count}")
    print(f"  - Số ảnh tăng cường đã tạo cho nhãn '{label}': {current_label_augmented_count}")
    print(f"  - Tổng số ảnh của nhãn '{label}' sau tăng cường: {current_label_original_count + current_label_augmented_count}")

print("\n--- Tổng kết toàn bộ quá trình tăng cường dữ liệu ---")
print(f"Tổng số ảnh gốc ban đầu (tất cả các nhãn): {original_total_count}")
print(f"Tổng số ảnh tăng cường đã tạo: {total_augmented_count}")
print(f"Tổng số ảnh sau tăng cường (bao gồm cả gốc và tăng cường): {total_after_augmentation_count}")


# Phần biểu đồ thống kê tổng quan
labels_plot = ["Trước tăng cường", "Sau tăng cường"]
counts_plot = [original_total_count, total_after_augmentation_count]

plt.figure(figsize=(8, 6))
plt.bar(labels_plot, counts_plot, color=["skyblue", "lightgreen"])
plt.ylabel("Số lượng ảnh")
plt.title("Tổng số lượng ảnh (Dataset Alzheimer's) trước và sau tăng cường dữ liệu")
plt.grid(axis='y', linestyle='--')
plt.show()
