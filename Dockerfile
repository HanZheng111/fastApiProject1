FROM nvidia/cuda:11.8.0-devel-ubuntu20.04


# 安装基础包
RUN apt-get update && \
    apt-get install -y \
        wget build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
        libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /temp

# 下载python(注释是源站 改用华为镜像下载提升速度)
#RUN wget https://www.python.org/ftp/python/3.11.8/Python-3.11.8.tgz && \
#    tar -xvf Python-3.11.8.tgz
RUN wget https://mirrors.huaweicloud.com/python//3.11.8/Python-3.11.8.tgz && \
    tar -xvf Python-3.11.8.tgz

# 编译&安装python
RUN cd Python-3.11.8 && \
    ./configure --enable-optimizations && \
    make && \
    make install

WORKDIR /workspace

RUN rm -r /temp && \
    ln -s /usr/local/bin/python3 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3 /usr/local/bin/pip

WORKDIR /app
COPY . /app


# 原请求 改为请求镜像站
#RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://mirrors.aliyun.com/pytorch-wheels/cu118 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    && pip install --no-cache-dir -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/   --trusted-host mirrors.aliyun.com


#RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
#    &&pip --no-cache-dir install -r requirements.txt


#CMD ["python","main.py"]