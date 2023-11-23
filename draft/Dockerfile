# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:latest

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip

# Copy files into container
COPY . .

# Install requirements
RUN pip install -r requirements.txt

