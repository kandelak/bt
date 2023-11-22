# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:latest

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip

# Copy only the requirements file into the container
COPY requirements.txt .

# Install requirements
RUN pip install -r requirements.txt

# Copy the script file into the container
COPY run.py .

# No need to copy the data, as it will be mounted during runtime

