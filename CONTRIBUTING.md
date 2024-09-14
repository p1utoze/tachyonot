# Table of Contents
1. [Setting Up and Contribution](#Setting Up)
2. [Building](#Building)
3. [Packaging](#Packaging)

## Setting Up
To setup the project, follow the steps below:
1. Clone the repository.
    ```shell
    $ git clone git@github.com:p1utoze/tachyonot.git
    ```
2. Using Poetry, install the project dependencies.
    ```shell
    $ poetry install
    ```
3. Perform the required changes in the src directory. Here is Project tree:
    ```shell

    ├── src
    │   ├── tachyonot
    │   │   ├── commands # Contains the command line interface commands
    │   │   ├── models # Contains the LLM and TTS model logic
    │   │   ├── rag # Contains the RAG logic
    │   │   ├── resources # Contains Model weights in model directory and icons in icons directory with nltk data in nltk_data directory
    │   │   ├── utils ## Contains the utility functions
    │   │   ├── __init__.py
    │   │   ├── __main__.py
    │   │   ├── app.py ## Contains PyQT5 GUI logic
   ```
4. To run in dev mode, execute the following command:
    ```shell
    $ briefcase dev
    ```
Note: Briefcase has to be run as root user

## Building
To build the project, execute the following command:
```shell
$ briefcase create
```
This will create the project based on your native platform. To build for a specific platform, use docker by passing --target <image-name> to the create command.
```bash
briefcase create --target debian:bullseye-slim
```

## Packaging
To package the project, execute the following command:
```shell
$ briefcase package
```
This will create the project based on your native platform. To package for a specific platform, use docker by passing --target <image-name> to the package command.
```bash
briefcase package --target debian:bullseye-slim
```

Note:
Please add models, icons and nltk data in the resources directory before packaging the project.
