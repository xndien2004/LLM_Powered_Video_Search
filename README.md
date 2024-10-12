# AIC2024

Dự án AIC2024 là một ứng dụng web dựa trên Django. Dưới đây là các bước cài đặt và chạy ứng dụng.

## Yêu cầu

- Python 3.x
- Django 3.x hoặc cao hơn
- Git

## Cài đặt

1. **Clone Repository**

   Trước tiên, clone repository từ GitHub:

   ```bash
   git clone https://github.com/dienlamAI/AIC2024.git
   cd AIC2024
   ```

2. **Cài đặt các gói phụ thuộc**

   Đảm bảo rằng bạn đã cài đặt Python và Django. Sau đó, cài đặt các gói phụ thuộc khác (nếu có) được liệt kê trong `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Cấu hình `MEDIA_ROOT`**

   Mở file [settings.py](./AIC/settings.py#L130-L131) trong thư mục `AIC`, tìm dòng `MEDIA_ROOT` và chỉnh sửa nó để trỏ đến đường dẫn thư mục `media` trên máy của bạn:

   ```python
   MEDIA_ROOT = 'đường_dẫn_đến_thư_mục_media'
   ```

   Ví dụ:

   ```python
   MEDIA_ROOT = '/path/to/your/media'
   ```

4. **settings file path**

   Kiểm tra đảm bảo rằng các đường dẫn trong file [viewAPI.py](./app/viewAPI.py#L19-L68) là chính xác

5. **Chạy Migrations**

   Áp dụng các migrations để cập nhật cơ sở dữ liệu:

   ```bash
   python manage.py migrate
   ```

## Chạy ứng dụng

Để chạy ứng dụng, sử dụng lệnh sau:

```bash
python manage.py runserver
```

Ứng dụng sẽ chạy trên địa chỉ `http://127.0.0.1:8000/` theo mặc định. Mở trình duyệt và truy cập địa chỉ này để xem ứng dụng.
