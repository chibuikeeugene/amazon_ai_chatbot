import requests
import configparser
import pandas as pd
import ast
import pymysql
import csv

# load the config object to read configurations
config = configparser.ConfigParser()

config.read('pipeline.ini')

# load mysql credentials
host = config['mysql.config']['hostname']
username = config['mysql.config']['username']
password = config['mysql.config']['password']
db = config['mysql.config']['database']
port = config['mysql.config']['port']

# load the url links from config
product_url = config['platzi.config']['product_url']

# ---------- extract data from source api -------------- #
def extract_data() -> pd.DataFrame:
    """ load the data via amazon api"""
    product = requests.get(
        url= product_url,
        timeout=60
    )
    product_data = product.json()
    print("Data retrieved successfully...")

    # create a dataframe
    df = pd.DataFrame(product_data)
    print('dataframe created successfully...')
    return df


# ---------- transform the data -------------- #
def transform_data(data: pd.DataFrame):
    """ transform daata and save file to a csv format"""
    # load the dataframe
    # df =  pd.read_csv(data)
    df_copy = data.copy()

    # retrieve just the category name
    category_list = []
    for cat_dict in df_copy['category']:
        # cat_dict = ast.literal_eval(cat_str)
        category_list.append(cat_dict['name'])

    df_copy['category'] =  category_list

    # retrieve just the first product image from the list
    image_list = []
    # extract the first image
    for image in df_copy['images']:
        # image_list.append(ast.literal_eval(image)[0])
        image_list.append(image[0])

    df_copy['images'] = image_list


    # convert the createdAT and updatedAT to timestamp
    createdAtList = [date.date()for date in pd.to_datetime(df_copy['creationAt'])]
    updateAtList =  [date.date()for date in pd.to_datetime(df_copy['updatedAt'])]

    df_copy['creationAt'] =  createdAtList
    df_copy['updatedAt'] =  updateAtList

    df_copy.to_csv('./data_pipeline/transformed_product_data.csv', index=False,)

# -----------load data to our database -------------- #
def load_data_to_mysqldb():
    """ loads a csv data into the mysql database"""

    # connect to mysql db
    conn = pymysql.connect(
        host=host,
        user=username,
        password=password,
        database=db,
        charset='utf8mb4',
        )
    
    # load the csv data to mysql database
    try:
        with open('./data_pipeline/transformed_product_data.csv', 'r', encoding='utf-8') as file:
            csv_reader  = csv.reader(file)

            # extract the header(first row) and the rest of the data
            headers =  next(csv_reader) 
            rows = [row for row in csv_reader]

            create_table_query = """ create table if not exists products
            (
                id int primary key, 
                title varchar(225) , 
                slug varchar(225), 
                price int, 
                description varchar(1000), 
                category varchar(225), 
                images varchar(225),
                creationAt timestamp, 
                updatedAt timestamp
            );
            """
            # instantiate the cursor object
            cursor = conn.cursor()

            cursor.execute(create_table_query)
        
            conn.commit()

            # inserting records into the table
            for row in rows:
                #  Build the insert query dynamically using the headers and values
                insert_data_query = f"""
                    INSERT INTO products ({', '.join(headers)}) 
                    VALUES ({', '.join(['%s'] * len(headers))}); 
                    """
            # Execute the insert query for each row
                cursor.execute(insert_data_query, row)
                conn.commit()

    except FileNotFoundError as e:
        print(f'The file: {e} not found! ')
        
    finally:
        # Close the connection
        conn.close()

    print("CSV data has been successfully inserted into the database!")


def etl():
    """ etl pipeline for data extraction, processing and storage"""
    dataframe = extract_data()
    transform_data(dataframe)
    load_data_to_mysqldb()
