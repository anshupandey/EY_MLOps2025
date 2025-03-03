# use a base image
FROM python:3.12-slim

# set the current working directory in the container
WORKDIR /app

# copy the file and install dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY app.py .

# expose port 8000
EXPOSE 8000

# run the app
CMD ["python","app.py"]