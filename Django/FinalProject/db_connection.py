import pymongo
import os

db_id = os.environ.get('DATABASE_USERNAME')
db_pwd = os.environ.get('DATABASE_PWD')
url = f'mongodb+srv://{db_id}:{db_pwd}@cluster0.g4t9j0f.mongodb.net/'
client = pymongo.MongoClient(url)

db = client['adhd']

