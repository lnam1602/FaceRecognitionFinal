import mysql.connector

class MySql:
    def __init__(self):
        self.connector = mysql.connector.connect(host="localhost", user="root", password="ledinhnam21001569", database="attendance")
        self.cur = self.connector.cursor()
        
    def insert(self, sql_query, values):
        self.cur.execute(sql_query, values)
        self.connector.commit()
        
    def read(self, sql_query):
        self.cur.execute(sql_query)
        rows = self.cur.fetchall()
        return rows
    
    def create(self, sql_query):
        self.cur.execute(sql_query)
        self.connector.commit()
    
    def update(self, sql_query, values):
        self.cur.execute(sql_query, values)
        self.connector.commit()
        
    def delete(self, sql_query, values):
        self.cur.execute(sql_query, values)
        self.connector.commit()
    
    def close(self):
        self.cur.close()
        self.connector.close()
        

mySql = MySql()