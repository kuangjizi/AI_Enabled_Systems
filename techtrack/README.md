# Quick Start to Run the Object Detection Service
Before you get started, make sure you're at the project directory `techtrack`. 

```
cd techtrack
```

1. Run the command below to build and load a docker image on your machine using the provided Dockerfile.
   ```shell
   docker build --no-cache -t techtrack .
   ```

2. Run the command below to stream a video at that port 23000 for the Object Detection service to pickup.
   ```shell
   ffmpeg -re -i storage/test_videos/worker-zone-detection.mp4 -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000
   ``` 

3. Start a container from the docker image. If the video streaming stops, please repeat step 2.
   ```shell
   docker run -p 23000:23000/udp  --name techtrack-container techtrack
   ```

4. [Optional] Copy output frames from container to local host for review
    ```shell
    docker cp techtrack-container:/app/output/<file_name> output/
    ```
    * Replace the <file_name> to the generated frame image. For example:
        ```
        docker cp techtrack-container:/app/output/frame_30.jpg output/
        ```

5. [Optional] Manage the Docker Container and Image
    - Stop the container:
    ```shell
    docker stop techtrack-container
    ```
    - Remove the container:
    ```shell
    docker rm techtrack-container
    ```
    - Restart the container:
    ```shell
    docker start techtrack-container
    ```
    - Remove the Docker image:
    ```shell
    docker rmi techtrack
    ```
    - Clean up everything (stopped containers, unused networks, volumes, and images)
    ```shell
    docker system prune -a
    ```
