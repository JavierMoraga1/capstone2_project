FROM python:3.8.12-slim

RUN pip install numpy 
RUN pip install --no-cache-dir tensorflow
RUN pip install fastapi
RUN pip install pydantic
RUN pip install pillow
RUN pip install uvicorn

WORKDIR /app

COPY ["model.h5", "predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]