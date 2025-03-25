# DataMineTurfGrass-Patholagy-Database
Clone the github repository into VS Code and then follow the steps below to setup the vector database

This will take around 5GB of space on local device with Docker. 

Firstly you will need to install Docker on your local system by following the steps on the website below:

  https://docs.docker.com/get-started/get-docker/

After fully following the documentation guide, you should have a Docker application on your system

The next step is to clone the directory on VS Code. After getting the files on a directory, open up the docker application on your system. 

At this step you should have docker open and VS Code with all the files in one repository/directory. Now You need to open the terminal in VS Code and run the following command:

    docker-compose up -d

If this does not work try the following command to see if any of them work: 

    sudo docker-compose up -d
    
    docker compose up -d
    
    sudo docker compose up -d

After the command goes through, on the docker application you should see a new container.
There are some changes to be made to the docker Milvus Yaml.
In Docker open the required container, view files for the Milvus-Standalone container. In the folder milvus, find a folder configs which should have the yaml file. Change the value at Line 271 i.e. "maxVectorFieldNum: 4" from 4 to 10.
You can now start the ports and connection to the database by turning it on using docker. 

Now you want to make sure you have all the imports used in the test.py file installed on your system. If you do not have pip, install pip. This will allow you to install python moduls necessary for the vector database

Also make sure you are in the directory with all the files. You will also need to create a virtual environment for the application to run. Sometimes this means when you run the code below line-by-line you will see a prompt on VS Code to create a virtual environment at the bottom right. Accept this and pick an interpretor. After that, you should be able to re-run the import installs below. You will now they are installed if on VS Code, the imports are highlighted in green/blue text instead of white:

    # Install pymilvus for Milvus interaction
    pip install pymilvus

    # Install PyTOrch as a Prerquisite
    pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0

    # Install sentence-transformers for generating embeddings
    pip install sentence-transformers

    # Install SQLite3 (comes pre-installed with Python, but you can install additional SQLite tools if needed)
    # This usually doesn't require installation, but just in case:
    pip install pysqlite3

    # Install numpy for numerical operations
    pip install numpy

If the installation goes through, you should be able to run the command:

    python test.py
    
From here you should be able to see a connection between Milvus and an output should be pushed out. If not please contact the Database Team. 


# Possible Errors:

If you got a docker error saying that the username does not match, this was resolved by making sure you make a docker account that is linked to your github. Make sure when you log in to docker you are using your username and not email. I don't know why this was the solution, but it did work.





