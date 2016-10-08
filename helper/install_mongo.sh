if ! type "mongod" > /dev/null; then
    echo "begin instal mongo-db 3.2"

    #Import the public key used by the package management system
    sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv EA312927

    #Create a list file for MongoDB
    echo "deb http://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/3.2 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.2.list

    #Reload local package database
    sudo apt-get update

    #Install the MongoDB packages
    sudo apt-get install -y mongodb-org=3.2.10 mongodb-org-server=3.2.10 mongodb-org-shell=3.2.10 mongodb-org-mongos=3.2.10 mongodb-org-tools=3.2.10

#     echo "[Unit]
# Description=High-performance, schema-free document-oriented database
# After=network.target
# Documentation=https://docs.mongodb.org/manual

# [Service]
# User=mongodb
# Group=mongodb
# ExecStart=/usr/bin/mongod --quiet --config /etc/mongod.conf

# [Install]
# WantedBy=multi-user.target" >> /lib/systemd/system/mongod.service

else
    echo "mongo-db already installed"
fi

mongod_installed=$(pydoc modules | grep -F pymongo)

if [ ${#mongod_installed} -le 0 ]; then
    echo "begin instal pymongo"
    sudo pip install pymongo

else
    echo "pymongo already installed"
fi
