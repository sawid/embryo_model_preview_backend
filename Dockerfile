# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV CLOUDINARY_CLOUD_NAME da3kzhohj
ENV CLOUDINARY_API_KEY 388149659154645
ENV CLOUDINARY_API_SECRET wXCPRySL64s0B3sGa-2vJ-kWoxY
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask will run on
EXPOSE 8000

# Command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]
