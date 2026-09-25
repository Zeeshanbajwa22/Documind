# Start from a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (helps Docker cache dependencies)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# Tell Docker which port the app will run on
EXPOSE 8000

# Command to start the FastAPI app when the container runs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]