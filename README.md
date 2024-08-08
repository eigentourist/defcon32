# Hacking The Metal: A Spark of Intelligence
### Code and materials for Hacking The Metal: A Spark of Intelligence, a workshop for DEF CON 32.

## Welcome!

Setup for this workshop consists of:
- Cloning this repository to your local machine
- Installing Docker Desktop for your platform
- Pulling or building the container image for the workshop

You can find Docker Desktop for your platform at https://www.docker.com/products/docker-desktop.

Once Docker is installed and running, you can pull the container image for the workshop by launching Docker Desktop and then running the following command in a terminal or command window:

`docker pull eigentourist/defcon32:v1.0.0`

Once you have the container image, you can launch the container from a command window (for Windows) or a terminal window (Linux or MacOS) with the following command:

`docker run -it --rm --name defcon32-hm -v local-path:/shared defcon32-hm`

...where local-path is the path to the local folder or directory where you have cloned the repository.

This should launch the container with the directory

/shared

in the container's file system mapped to your code folder in your host environment. This way, you can view and edit files in your favorite editor or IDE, and then build and run them in the command/terminal window where you have launched the container.

**Note** - code is still being added to this repo! So if you're signed up for the workshop and decide to clone it before Saturday, be sure to do a fresh pull Saturday morning. As always, feel free to give us a shout in the Discord server about any issues you may run into with the Docker image, code, build scripts, or just the shock of walking into the Las Vegas summer heat. 