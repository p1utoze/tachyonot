FROM debian:bullseye-slim
WORKDIR /app



ENTRYPOINT ["top", "-b"]