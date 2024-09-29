FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install rembg[gpu] \
    && pip install fastapi \
    && pip install opencv \
    && pip install numpy \
    && pip install uvicorn


FROM nvidia/cuda:11.8.0-devel-ubi8

RUN uvicorn main:app -host 0.0.0.0 -port 9090