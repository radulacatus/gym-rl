def get_db():
    from pymongo import MongoClient
    client = MongoClient('localhost:27017')
    db = client.smokeDb
    return db

if __name__ == "__main__":

    db = get_db() 
    db.ping.insert({"ping" : "Ping!"})
    print db.ping.find_one()['ping']
    print db.collection_names()
    db.ping.drop()
    print db.collection_names()