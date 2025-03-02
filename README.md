# VN-SubMaker

VN-SubMaker là một ứng dụng web hiện đại giúp tạo phụ đề tự động từ các tệp âm thanh và video. Ứng dụng sử dụng công nghệ nhận dạng giọng nói tiên tiến để tạo phụ đề chính xác, hỗ trợ cả MP3 và MP4.

## 🚀 Tính năng nổi bật
- **Giao diện hiện đại**: Thiết kế UI thân thiện, dễ sử dụng.
- **Nhận diện giọng nói thông minh**: Sử dụng mô hình Transformer mạnh mẽ.
- **Xử lý MP3 & MP4**: Nhận diện phụ đề cho cả audio và video.
- **Tạo và nhúng phụ đề**: Nhúng file `.srt`  trực tiếp vào video.
- **Tải xuống kết quả**: Hỗ trợ tải file phụ đề và video có phụ đề.
- **Tiến trình theo thời gian thực**: Hiển thị trạng thái xử lý trực tiếp.

## 🖥️ Hướng dẫn sử dụng
### 1️⃣ Chạy ứng dụng
```sh
# Clone repository
git clone https://github.com/MinDuck404/VN-SubMaker.git
cd VN-SubMaker

# Cài đặt thư viện cần thiết (chưa up file nên giờ không cần chạy)
pip install -r requirements.txt

# Chạy ứng dụng Flask
python app.py
```
Ứng dụng sẽ chạy tại `http://127.0.0.1:5000/`.

### 2️⃣ Nhập file MP3 hoặc MP4
- **MP3**: Ứng dụng sử dụng mô hình nội bộ để nhận diện giọng nói.
- **MP4**: Ứng dụng trích xuất audio từ video và xử lý nhận diện.

### 3️⃣ Chỉnh sửa & tải xuống
- Sau khi xử lý, phụ đề sẽ hiển thị trên giao diện.
- Bạn có thể tải file text hoặc video đã có phụ đề nhúng.

## 📌 Công nghệ sử dụng
- **Backend**: Flask, Transformers (Whisper ASR), PyDub, MoviePy
- **Frontend**: Bootstrap, JavaScript (fetch API, EventSource)
- **Xử lý âm thanh/video**: FFmpeg, Pydub, MoviePy

## 📄 API Endpoint
| Phương thức | Đường dẫn | Chức năng |
|------------|----------------|----------------|
| `POST` | `/transcribe` | Tải lên file và bắt đầu nhận diện |
| `GET` | `/status/<session_id>` | Kiểm tra tiến trình xử lý |
| `GET` | `/download_text/<session_id>` | Tải file văn bản |
| `GET` | `/download_srt/<session_id>` | Tải file SRT |
| `GET` | `/download_video/<session_id>` | Tải video có phụ đề |

## 🤝 Đóng góp
Chúng tôi hoan nghênh mọi đóng góp từ cộng đồng! Nếu bạn muốn cải thiện VN-SubMaker, vui lòng gửi pull request hoặc báo lỗi trên GitHub.

## 📜 Giấy phép
VN-SubMaker được phát hành dưới giấy phép MIT.

