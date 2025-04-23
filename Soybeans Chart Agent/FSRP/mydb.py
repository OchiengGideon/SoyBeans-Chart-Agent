import mysql.connector

dataBase= mysql.connector.connect(
  host = 'localhost',
  user = 'root',
  passwd = '6241Gideon@' 
  )


#prepare a cursor object
cursorObject = dataBase.cursor()
#Create a database
cursorObject.execute("CREATE DATABASE FSRP")

print("All Done!")