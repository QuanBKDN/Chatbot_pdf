# Sử dụng hình ảnh Python phiên bản 3.9.12 làm hình ảnh cơ sở
FROM python:3.8.10


RUN apt-get update &&  apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 build-essential



RUN  apt install -y g++

RUN apt-get install -y build-essential libssl-dev ca-certificates libasound2 wget

RUN apt-get install -y libsndfile1

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install pyaudio && \
    pip install -r requirement.txt

EXPOSE 5000

CMD ["python", "app.py"]