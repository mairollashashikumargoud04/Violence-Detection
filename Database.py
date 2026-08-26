import pymysql

def getConnection():
    con = pymysql.connect(host='localhost',user='root',password='root',database='violance')

    return con