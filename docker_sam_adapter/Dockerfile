# Use the official Ubuntu 20.04 image as the base image
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8


# Install essential packages
RUN apt-get update && \
    apt-get install -y \
    ffmpeg libsm6 libxext6 \
    python3.8 \
    python3-pip \
    curl gnupg


# Add package repository for nvidia toolkit
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

RUN apt-get update && apt-get install -y nvidia-container-toolkit

# Add environment variables for GPU execution

RUN echo "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video" >> /etc/environment
RUN echo "NVIDIA_REQUIRE_CUDA=cuda=11.6" >> /etc/environment
RUN echo "NVIDIA_VISIBLE_DEVICES=all" >> /etc/environment

# Set Python 3.8 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Copy requirements.txt to the working directory
COPY requirements.txt /app/requirements.txt

# Set the working directory
WORKDIR /app

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    pip3 install -U openmim

# Install mmcv package
RUN mim install mmcv==1.7.0
