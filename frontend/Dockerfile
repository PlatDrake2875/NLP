FROM node:18
ENV NODE_ENV=development
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies using npm
RUN npm install

# Copy the rest of your source code
COPY . .

# Expose the port that your dev server uses (Vite defaults to 5173)
EXPOSE 5173

# Run the development server using npm
CMD ["npm", "run", "dev", "--", "--host"]