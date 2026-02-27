# Use the full bookworm image so AI/ONNX models have all required C++ libraries
FROM node:22-bookworm

# Configure pnpm
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN npm install -g pnpm@10

# Create app directory
WORKDIR /app

# Install dependencies
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

# Copy the rest of the code
COPY . .

# Build the TypeScript code
RUN pnpm build

# Hugging Face runs containers as user 1000. 
# We need to give full read/write permissions so the agent can update clinic_data.json
RUN chown -R 1000:1000 /app
RUN chmod -R 777 /app

USER 1000

# Pre-download ML models
RUN pnpm download-files

# Set Production mode
ENV NODE_ENV=production

# FORCE Hugging Face Port
ENV PORT=7860

# Start the agent
CMD [ "pnpm", "start" ]