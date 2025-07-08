import streamlit as st
import os
import sys
import time
import markdown
from pathlib import Path
from dotenv import load_dotenv

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import các module cần thiết
from app.langraph.main import get_medical_diagnosis

st.set_page_config(
    page_title="Consensus Mechanism AI Agents",
    page_icon="🩺",
    layout="wide",
)

def main():
    st.title("Consensus Mechanism AI Agents")
    st.subheader("Hệ thống chẩn đoán y tế dựa trên đồng thuận AI")

    # Sidebar cho thông tin về hệ thống
    with st.sidebar:
        st.header("Thông tin")
        st.info("""
        **Consensus Mechanism AI Agents** là hệ thống sử dụng nhiều AI agent để thực hiện:
        1. Nghiên cứu y tế theo thời gian thực
        2. Xác minh độ tin cậy của nguồn
        3. Đưa ra nhiều chẩn đoán độc lập
        4. Tạo kế hoạch điều trị
        5. Đưa ra kết luận đồng thuận
        """)
        
        st.header("API Keys")
        # Hiển thị trạng thái API keys
        io_api_key = os.getenv("IOINTELLIGENCE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        serper_api_key = os.getenv("SERPER_API_KEY")
        
        if io_api_key:
            st.success("✅ IOINTELLIGENCE_API_KEY: Đã cấu hình")
        elif openai_api_key:
            st.success("✅ OPENAI_API_KEY: Đã cấu hình")
        else:
            st.error("❌ Không tìm thấy API key cho LLM. Cần có OPENAI_API_KEY hoặc IOINTELLIGENCE_API_KEY")
            
        if serper_api_key:
            st.success("✅ SERPER_API_KEY: Đã cấu hình")
        else:
            st.warning("⚠️ SERPER_API_KEY: Chưa cấu hình (tìm kiếm thời gian thực sẽ không hoạt động)")

    # Form nhập thông tin
    with st.form("diagnosis_form"):
        st.header("Nhập thông tin bệnh nhân")
        
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input("Chủ đề y tế cần nghiên cứu", "Tinnitus")
            symptoms = st.text_area("Triệu chứng", "Ringing in ears, difficulty sleeping")
        
        with col2:
            medical_history = st.text_area("Tiền sử bệnh", "Recent concert attendance, no prior hearing issues")
            test_results = st.text_area("Kết quả xét nghiệm", "None")
        
        use_realtime = st.checkbox("Sử dụng tìm kiếm web thời gian thực", value=False)
        
        submit_button = st.form_submit_button("Bắt đầu chẩn đoán")
    
    # Xử lý khi nhấn nút chẩn đoán
    if submit_button:
        if not (os.getenv("IOINTELLIGENCE_API_KEY") or os.getenv("OPENAI_API_KEY")):
            st.error("Thiếu API key cho LLM. Vui lòng cấu hình OPENAI_API_KEY hoặc IOINTELLIGENCE_API_KEY trong file .env")
            return
            
        if use_realtime and not os.getenv("SERPER_API_KEY"):
            st.warning("Thiếu SERPER_API_KEY cho tìm kiếm web. Hệ thống sẽ sử dụng dữ liệu mô phỏng.")
        
        # Hiển thị thanh tiến trình
        progress_bar = st.progress(0)
        status_text = st.empty()
        result_container = st.empty()
        
        try:
            # Các giai đoạn của quy trình chẩn đoán
            stages = [
                "Nghiên cứu y khoa",
                "Xác minh độ tin cậy nguồn",
                "Phân tích chẩn đoán",
                "Đề xuất điều trị",
                "Xây dựng đồng thuận",
                "Hoàn thành báo cáo"
            ]
            
            # Bắt đầu chạy quy trình chẩn đoán thật
            with st.spinner("Đang nghiên cứu và phân tích..."):
                status_text.text(f"Đang thực hiện: {stages[0]}...")
                progress_bar.progress(1/len(stages))
                
                result = get_medical_diagnosis(
                    topic=topic,
                    symptoms=symptoms,
                    medical_history=medical_history,
                    test_results=test_results,
                    realtime=use_realtime
                )
                
                # Hiển thị tiến trình cho các bước còn lại
                for i in range(1, len(stages)):
                    status_text.text(f"Đang thực hiện: {stages[i]}...")
                    progress_bar.progress((i+1)/len(stages))
                    time.sleep(0.5)  # Giả lập tiến trình
            
            status_text.text("Hoàn thành chẩn đoán!")
            progress_bar.progress(100)
            
            # Hiển thị kết quả
            with result_container.container():
                st.success("Chẩn đoán hoàn tất! Xem kết quả bên dưới:")
                st.divider()
                
                st.header(f"Chẩn đoán cho: {result['topic']}")
                
                st.subheader("Kết luận đồng thuận")
                st.markdown(result['consensus'])
                
                st.subheader("Chẩn đoán")
                if isinstance(result['diagnoses'], list):
                    for i, diag in enumerate(result['diagnoses']):
                        st.write(f"{i+1}. {diag}")
                else:
                    st.markdown(result['diagnoses'])
                
                st.subheader("Đề xuất điều trị")
                if isinstance(result['treatments'], list):
                    for i, treat in enumerate(result['treatments']):
                        st.write(f"{i+1}. {treat}")
                else:
                    st.markdown(result['treatments'])
                
                # Hiển thị thông tin nghiên cứu nếu có
                with st.expander("Xem thông tin nghiên cứu chi tiết"):
                    st.write(result['research_findings'])
                    
                    st.subheader("Nguồn tham khảo")
                    for source in result['verified_sources']:
                        st.write(f"- {source}")
                    
                    st.write(f"**Điểm tin cậy nguồn**: {result['source_credibility']:.2f}/1.0")
                
        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình chẩn đoán: {str(e)}")
            st.error("Vui lòng kiểm tra lại cấu hình và API keys trong file .env")

if __name__ == "__main__":
    main() 