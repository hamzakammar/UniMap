# Set base image (host OS)
FROM python:3.8-slim

# By default, listen on port 5000
EXPOSE 5000/tcp

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any 
RUN pip install --upgrade pip
RUN pip3 install --upgrade pip setuptools && \
    pip3 install -r requirements.txt
RUN pip3 install numpy===1.19.3
RUN pip install --upgrade pip setuptools wheel
RUN pip install Pillow
RUN pip install matplotlib
RUN pip install requests
RUN pip install pathlib
# Copy the content of the local src directory to the working directory
COPY app.py .

# Specify the command to run on container start
CMD [ "python", "./app.py" ]
