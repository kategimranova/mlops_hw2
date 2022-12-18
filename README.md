Домашнее задание 2

1. Для классического запуска докера:
```bash
docker build . -t mlops_hw2

docker run -d -p 8000:8000 mlops_hw2

```

(сервер запустится на http://localhost:8000/)

(для документации: http://localhost:8000/docs)

2. Через docker-compose
sudo docker-compose up -d

3. Через докерхаб: https://hub.docker.com/repository/docker/kattgim/mlops
docker pull kattgim/mlops

