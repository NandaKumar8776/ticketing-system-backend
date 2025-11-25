URI = "http://localhost:19530"

from langchain_milvus import Milvus
from utils.helpers import embeddings

### Milvus Vector DB Creating if it does not exist


from pymilvus import Collection, MilvusException, connections, db, utility

try:
    conn = connections.connect(host="localhost", port=19530)

    db_name = "milvus_assignment_test"
    try:
        existing_databases = db.list_database()
        db_was_deleted = False

        if db_name in existing_databases:
            print(f"Database '{db_name}' already exists.")

            # Use the database context
            db.using_database(db_name)

            # Drop all collections in the database
            collections = utility.list_collections()
            for collection_name in collections:
                collection = Collection(name=collection_name)
                collection.drop()
                print(f"Collection '{collection_name}' has been dropped.")

            # Drop the database
            db.drop_database(db_name)
            db_was_deleted = True
            print(f"Database '{db_name}' has been deleted.")
        else:
            print(f"Database '{db_name}' does not exist.")
            db_was_deleted = True

        # Recreate the database if it was deleted
        if db_was_deleted:
            db.create_database(db_name)
            print(f"Database '{db_name}' created successfully.")

    except MilvusException as e:
        print(f"An error occurred: {e}")


    #### Milvus DB Local Instance running on Docker- connected below

    # Creating a FLAT index vector store with L2 similarity check


    flat_milvus_vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI, "db_name": "milvus_assignment_test"},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
        collection_name="Flat_Index_PC_Troubleshooting_PDF",
        collection_description= "Its the contents from the PDF, explaining how to troubleshoot issues with a PC. It uses Flat Index.",
        consistency_level="Strong"

    )

except Exception as e:
    print(f"Warning: Could not initialize Milvus vector store: {e}")
    print("Make sure Milvus is running on localhost:19530")
    flat_milvus_vector_store = None

"""
## NOT NEEDED FOR NOW

# Web Crawler data's vector store 

web_crawler_milvus_vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI, "token": "root:Milvus", "db_name": "milvus_assignment_test"},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    collection_name="Flat_Index_Web_Crawler_Data",
    collection_description= "Its the website scraped contents from a PC troubleshooting Website.",
    consistency_level="Strong"

)

print("\nEmpty vector stores has been created successfully")
"""