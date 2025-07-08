# Hướng dẫn chạy Consensus Mechanism AI Agents với Streamlit

Ứng dụng Streamlit cung cấp giao diện người dùng thân thiện để tương tác với hệ thống Consensus Mechanism AI Agents.

## Cài đặt

1. Đảm bảo bạn đã cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

1. Từ thư mục gốc của dự án, chạy lệnh sau:

```bash
streamlit run app/streamlit_app.py
```

2. Trình duyệt web sẽ tự động mở và hiển thị giao diện Streamlit.

## Sử dụng

1. Trong sidebar bên trái, nhập API key cần thiết:
   - **IOINTELLIGENCE_API_KEY**: Bắt buộc phải có
   - **GOOGLE_API_KEY**: Tùy chọn, dùng cho tìm kiếm Google
   - **SERPAPI_API_KEY**: Tùy chọn, dùng cho tìm kiếm SerpAPI

2. Trong form chính, nhập thông tin bệnh nhân:
   - **Chủ đề y tế**: Chủ đề cần nghiên cứu (ví dụ: Tinnitus, COVID-19, Diabetes...)
   - **Triệu chứng**: Các triệu chứng bệnh nhân gặp phải
   - **Tiền sử bệnh**: Thông tin về tiền sử bệnh
   - **Kết quả xét nghiệm**: Thông tin về các xét nghiệm đã thực hiện

3. Tùy chọn:
   - **Sử dụng tìm kiếm thời gian thực**: Bật để sử dụng tìm kiếm web thời gian thực (yêu cầu API keys). Tắt để sử dụng kiến thức sẵn có của mô hình.

4. Nhấn nút **Bắt đầu chẩn đoán** để khởi động quy trình.

5. Theo dõi tiến trình qua thanh tiến độ và thông báo trạng thái.

6. Kết quả chẩn đoán sẽ hiển thị sau khi quy trình hoàn tất.

## Lưu ý

- Quá trình chẩn đoán có thể mất từ 1-5 phút tùy thuộc vào độ phức tạp của chủ đề và việc có sử dụng tìm kiếm thời gian thực hay không.
- Nếu sử dụng tìm kiếm thời gian thực, việc có API keys hợp lệ sẽ cải thiện đáng kể chất lượng kết quả.
- Kết quả chẩn đoán cũng được lưu vào file `medical_diagnosis_results.md` trong thư mục gốc của dự án. 