import configparser
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
import pymysql
import pandas as pd
from loguru import logger

# load config files
config =  configparser.ConfigParser()
config.read('pipeline.ini')


# load mysql credentials
host = config['mysql.config']['hostname']
username = config['mysql.config']['username']
password = config['mysql.config']['password']
db = config['mysql.config']['database']
port = config['mysql.config']['port']

# columns or features of interests
features = ['id','title','price','description','category']

def document_loader():
    """ Loads the record from the mysql db, convert selected fields in each record to document
        and return the document object
    """
    # read record from database
    # connect to mysql db
    try:
        conn = pymysql.connect(
            host=host,
            user=username,
            password=password,
            database=db,
            charset='utf8mb4',
            )
        logger.info('connection created successfully...')
    except ConnectionError:
        logger.exception('Error occured while creating connection!')
    
    # sql query to retrieve records
    retrieve_data_query = "select * from products"

    # creating a csv file  and writing data in chunks for memory efficiency
    csv_file = './rag_chain/output.csv'
    headers =  True

    # read data in chunks
    try:
        for chunk in pd.read_sql_query(retrieve_data_query, conn, chunksize=20): # read 20 rows at a time
            chunk[features].to_csv(csv_file, mode='a', header=headers, index=False) # save the records to csv by adding each record
            headers = False # ensure header isn't included in the second and subsequent calls
        logger.log(10, f'Data successfully written to {csv_file}')
    finally:
        conn.close()
        logger.info('Connection to database closed...')
    # convert each record to a document
    loader =  CSVLoader('./rag_chain/output.csv', )
    docs = loader.load()
    return docs


def document_embeddings(docs):
    """ generate the embeddings for each document and save it to our vectorstore chromadb """
    # create the embedding object
    embeddings = OllamaEmbeddings(model='llama3.1')

    # creating the vector store
    vectorstore_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name='amazon_product_collection',
        persist_directory='./rag_chain/'
    )
    logger.info('Returning the vectorstore with its embeddings...')
    return vectorstore_db