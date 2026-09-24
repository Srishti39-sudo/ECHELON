# The DeepEcho dashboard: built by Vite, served as plain files by nginx.
#
# VITE_API_BASE_URL is read at BUILD time and compiled into the bundle. It is
# the address the BROWSER uses to reach the API, so it is the gateway as seen
# from the host (http://localhost:8000 under compose), never a compose service
# name: the browser is not on the compose network. Changing it means
# rebuilding this image.

FROM node:22-alpine AS build
WORKDIR /app

# The lockfile first, so the install layer survives a source edit.
COPY package.json package-lock.json ./
RUN npm ci

ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

COPY . .
RUN npm run build


FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
