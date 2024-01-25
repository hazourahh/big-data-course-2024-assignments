# How to Run the Assignment Notebooks

To run the assignment notebooks for this course, we have prepared a docker image which contains the libraries needed for completing the assignments.

To get started, first clone this repository into your local machine.

```
git clone https://github.com/hazourahh/big-data-course-2024-assignments.git
```

If you're new to git, learn how to set up git [here](https://docs.github.com/en/get-started/quickstart/set-up-git).

Next, follow the steps below.

## Download Docker for Desktop

Download and Install [Docker for Desktop](https://www.docker.com/products/docker-desktop)

## Run Docker

There are two ways you can make use of this docker image:

1. By pulling the pre-built image from Docker Hub (recommended)
2. By building the image using Dockerfile

### Option 1: Using Pre-built Image from Docker Hub

**Step 1:** Pull the image

```
docker pull mtasnim/jupyter-pyspark-duckdb
```

**Step 2:** Set current directory

Make sure your working directory (pwd) is set to a directory containing the assignment notebooks from this repository.

```
cd big-data-course-2024-assignments
```

**Step 3:**  Run the image

```
docker run -p 8888:8888 -v $(pwd):/home/jovyan/work  -e JUPYTER_ENABLE_LAB=yes mtasnim/jupyter-pyspark-duckdb

```

If you have an error that JUPYTER_ENABLE_LAB is ignored, then use this alternative command to run the image:

```

docker run -p 8888:8888 -v $PWD:/home/jovyan/work -e DOCKER_STACKS_JUPYTER_CMD=lab mtasnim/jupyter-pyspark-duckdb
```

**Step 4:**  Access notebooks

To access Jupyter go to `http://localhost:8888/lab?token=REPLACE_ME_WITH_YOUR_TOKEN` from your browser. 

**NOTE:** Make sure you copy the notebook token from the terminal in order to access the notebooks, or directly click on the link displayed in the terminal. Under MacOS, you find the link as follows: 

![token](jupyter-links.png)


### Option 2: Building Image from Dockerfile

If downloading the image is not possible for you, then another option is to build the image using `docker build`

**Step 1:** Clone the Docker Setup Repository

```
git clone https://github.com/schelterlabs/bd22-docker-setup.git
```

**Step 2:** Build the Dockerfile

```
cd bd22-docker-setup
```

```
docker build -t mtasnim/jupyter-pyspark-duckdb .
```

Building the image for the first time should take a while.

**Step 3:**  Run the newly built image

To run the image and access the notebooks follow Step 2, 3 and 4 from Option 1 above.



## Troubleshooting

### Memory Issues: Recommended RAM Limit

In Assignment 2, Spark might randomly crash if it doesn't have enough memory available. In case you encounter issues like that, we suggest increasing the amount of memory available to the docker image to 4gb or more. If you are using docker-for-windows or docker-for-mac, you can easily increase it from the Whale 🐳 icon in the task bar, then go to Preferences -> Resources. 

### Space Issues: Cleaning up unused images & volumes to free up space

When using Docker for a while it's possible to run out of reserved diskspace, which gets filled with unused docker images and data volumes. 

One solution is to delete unused Docker images and volumes locally.

If you have Docker containers actively running, first stop them

```
docker container stop $(docker container ls -aq)
```

Then remove all dangling (unattached) volumes, networks and images.

```
docker volume rm $(docker volume ls -qf dangling=true)
```

```
docker system prune
```

```
docker image prune -a
```

This should have reclaimed back enough diskspace to resume operations.
