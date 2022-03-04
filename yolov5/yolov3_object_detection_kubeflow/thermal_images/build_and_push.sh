docker build -t 192.168.152.34:5050/thermal_images:latest . || exit
docker push 192.168.152.34:5050/thermal_images:latest