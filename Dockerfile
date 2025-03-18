# Sử dụng image Python chính thức làm base
FROM python:3.12.6-slim
WORKDIR /app

# Cài đặt các công cụ biên dịch và header files
RUN apt-get update && apt-get install -y \
    gcc \
    libc-dev \
    linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]