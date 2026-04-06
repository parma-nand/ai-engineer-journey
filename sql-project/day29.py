import psycopg2
from config import DB_CONFIG

conn = psycopg2.connect(**DB_CONFIG)

cursor = conn.cursor()

# cursor.execute("create table employees (id int,name varchar (255),subject varchar, anount int, year int)")

# # Insert sample data
# cursor.execute("INSERT INTO employees VALUES (%s,%s,%s,%s,%s)", [
#     (1, 'Parma',   'ML',      90000, 2022),
#     (2, 'Ankit',   'Data',    75000, 2021),
#     (3, 'Sneha',   'ML',      95000, 2023),
#     (4, 'Rahul',   'DevOps',  80000, 2022),
#     (5, 'Priya',   'Data',    70000, 2023),
#     (6, 'Vikram',  'ML',      85000, 2021),
# ])
# conn.commit()
cursor.execute("SELECT * FROM employees where id=1")
# rows = cursor.fetchone()
rows = cursor.fetchall()
print(rows)

conn.close()

print("SQL")
