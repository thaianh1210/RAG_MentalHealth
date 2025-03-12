import os

directories = [
    "data/cache",
    "data/images",
    "data/index_storage",
    "data/ingestion_storage",
    "data/user_storage"
]

for dir in directories:
    os.makedirs(dir, exist_ok=True) 