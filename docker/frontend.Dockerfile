# docker/frontend.Dockerfile
FROM node:20-slim

# Set working directory
WORKDIR /app

# Copy package files
COPY frontend/package.json frontend/yarn.lock ./

# Install dependencies
RUN yarn install

# Copy application code
COPY frontend .

# Set the API URL for production build
ENV VITE_API_URL=http://localhost:8000/api

# Build the application
RUN yarn build

# Install serve
RUN yarn global add serve

# Serve the built application
CMD ["serve", "-s", "dist", "-l", "3000"]
