FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

FROM node:20-alpine
WORKDIR /app
ENV HOST=0.0.0.0 PORT=3000 NODE_ENV=production
COPY --from=build /app/.output ./.output
EXPOSE 3000
# адреса бэкендов читаются в рантайме: NUXT_ML_URL / NUXT_SOMMELIER_URL
CMD ["node", ".output/server/index.mjs"]
