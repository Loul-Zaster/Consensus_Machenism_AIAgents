#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script để chạy đánh giá RAG sử dụng RAGAS

Cách sử dụng:
    - Để chạy đánh giá từ dòng lệnh: python run_evaluation.py
    - Để chạy giao diện Streamlit: streamlit run streamlit_evaluation.py
"""

import os
import argparse
from dotenv import load_dotenv

# Tải biến môi trường từ file .env
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Đánh giá RAG với RAGAS")
    parser.add_argument(
        "--streamlit", 
        action="store_true", 
        help="Chạy đánh giá với giao diện Streamlit"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="evaluation_dataset.json",
        help="Đường dẫn đến file dataset đánh giá"
    )
    args = parser.parse_args()
    
    if args.streamlit:
        # Chạy giao diện Streamlit
        os.system(f"streamlit run streamlit_evaluation.py")
    else:
        # Chạy đánh giá từ dòng lệnh
        from evaluation_ragas import main as run_evaluation
        run_evaluation(use_streamlit=False)

if __name__ == "__main__":
    main()