import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Tải mô hình đã huấn luyện
@st.cache_resource
def load_model():
    # Định nghĩa tên các lớp
    class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    
    # Tải mô hình ResNet-50
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # Tải trọng số từ file .pth
    try:
        model_path = 'final_best_model.pth'
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval() # Chuyển mô hình sang chế độ đánh giá
        return model, class_names
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file mô hình '{model_path}'. Vui lòng đảm bảo file nằm cùng thư mục.")
        return None, None

model, class_names = load_model()

# Định nghĩa các phép biến đổi cho ảnh đầu vào
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0) # Thêm một chiều batch

# Xây dựng giao diện web với Streamlit
st.title("Ứng dụng phân loại ảnh MRI não")
st.header("Phát hiện các giai đoạn của bệnh Alzheimer")

if model is not None:
    st.markdown("---")
    st.write("Vui lòng tải lên một ảnh MRI não để phân loại.")

    uploaded_file = st.file_uploader("Chọn một ảnh từ máy tính của bạn...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Đọc và hiển thị ảnh
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='Ảnh đã tải lên', use_column_width=True)
            st.write("")

            # Nút phân loại
            if st.button("Phân loại"):
                # Tiền xử lý ảnh
                with st.spinner('Đang phân tích ảnh...'):
                    input_tensor = preprocess_image(image)

                    # Phân loại ảnh
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(output[0], dim=0)

                    # Lấy kết quả
                    predicted_prob, predicted_index = torch.max(probabilities, 0)
                    predicted_class = class_names[predicted_index.item()]
                    confidence = predicted_prob.item() * 100

                    # Hiển thị kết quả
                    st.success("Phân loại hoàn tất!")
                    st.write("---")
                    st.write(f"**Kết quả phân loại:**")
                    st.write(f"Dự đoán: **{predicted_class}**")
                    st.write(f"Độ tin cậy: **{confidence:.2f}%**")

                    st.markdown("---")
                    st.write("Các độ tin cậy của từng lớp:")
                    for i, (prob, class_name) in enumerate(zip(probabilities, class_names)):
                        st.write(f"- {class_name}: {prob.item() * 100:.2f}%")

        except Exception as e:
            st.error(f"Có lỗi xảy ra: {e}")