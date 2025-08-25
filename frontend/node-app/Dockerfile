# Use official Node.js image
FROM node:18

# Set working directory
WORKDIR /app

# Copy package.json and lock file
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the app
COPY . .

# Expose Vite default dev server port
EXPOSE 5173

# Start the development server
CMD ["npm", "run", "dev"]
