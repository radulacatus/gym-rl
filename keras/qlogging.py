import datetime
from bson import ObjectId
from pymongo import MongoClient

class qlogger(object):
    
    def __init__(self, dbName = "experimentsDb"):
        self.mongoClient = MongoClient('localhost:27017')
        self.db = self.mongoClient.db[dbName]
        self.init_properties()

    def init_properties(self):
        self.currentExperimentId = None
        self.score = series('scores')
        self.random_hits = series('random_hits')
        self.errors = []
        self.comments = []

    def start_experiment(self, params, name = None, desc = None):
        timestamp = datetime.datetime.now()
        self.currentExperimentId = self.db.experiments.insert({'startTime': timestamp, 'params': params, 'name': name, 'desc': desc })

    def log_error(self, errorMessage):
        self.errors.append(errorMessage)

    def log_comment(self, message):
        self.comments.append(message)

    def end_experiment(self):
        timestamp = datetime.datetime.now()
        self.save_series(self.score)
        self.save_series(self.random_hits)
        self.db.experiments.update_one({'_id': self.currentExperimentId}, {'$set': 
        {
            'endTime': timestamp,
            'errors': self.errors,
            'comments': self.comments
        }}, upsert=True)
        rez = self.currentExperimentId
        self.init_properties()
        return rez

    def save_series(self, series):
        if self.currentExperimentId == None:
            raise Exception('You have to start an experiment before adding timeseries data.')
        if series == None or series.is_empty():
            return
        
        self.db[series.name].insert(
            {
                '_id': self.currentExperimentId,
                'name': series.name,
                'episodeIndexes': series.episodeIndexes,
                'timestepIndexes': series.timestepIndexes,
                'recorderdValues': series.recorderdValues,
            })

    def log_score(self, val, ep):
        self.score.add_timestep(val,ep)

    def log_random_hits(self, val, ep, ts):
        self.random_hits.add_timestep(val, ep, ts)


class series(object):

    def __init__(self, name):
        self.name = name
        self.episodeIndexes = []
        self.timestepIndexes = []
        self.recorderdValues = []

    def is_empty(self):
        return len(self.episodeIndexes) == 0

    def add_timestep(self, val, ep, ts = -1):
        self.recorderdValues.append(val)
        self.episodeIndexes.append(ep)
        if ts > 0:
            self.timestepIndexes.append(ts)
    

class experimentsRepository(object):
    
    def __init__(self, dbName = "experimentsDb"):
        self.mongoClient = MongoClient('localhost:27017')
        self.db = self.mongoClient.db[dbName]

    def find_by_id(self, collectionName, id):
        return [i for i in self.db[collectionName].find({'_id': ObjectId(id)})][0]

    def get_experiment(self, id):
        return self.find_by_id('experiments', id)

    def get_scores(self, id):
        return self.find_by_id('scores', id)

    def get_random_hits(self, id):
        return self.find_by_id('random_hits', id)

    def get_full_log(self, id):
        return {
            'experiment': self.get_experiment(id),
            'scores': self.get_scores(id),
            'random_hits': self.get_random_hits(id)
        }