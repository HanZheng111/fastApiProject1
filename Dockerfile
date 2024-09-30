FROM conda/miniconda3
 
# 设置工作目录
WORKDIR /app

# 复制本地文件到工作目录
COPY . /app

RUN conda env create --name newenv --file environment.yml
 
# 设置容器启动时执行的命令
CMD ["uvicorn", "main:app" ,"--port", "9090"]