import sqlite3
import datetime
from persiantools.jdatetime import JalaliDate


class FunctionsQuery():

    def __init__(self,name) -> None:
        self.name = name

    def create_database(self):

        connect = sqlite3.connect('D:\Project-VSCode\Face-Recognition-RealTime-Attendance-System-SQL-main\database.db')
        cursor = connect.cursor()
        cursor.execute('''
    CREATE TABLE IF NOT EXISTS database(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        names NVARCHAR(50) NOT NULL,
        dates NVARCHAR(10) NOT NULL,
        clocks NVARCHAR(8) NOT NULL,
        statuses NVARCHAR(4) NOT NULL
    );
    ''')
        connect.commit()
        connect.close()
        


    def time_now(self):
        time = datetime.datetime.now()
        time = time.time()
        time = str(time)[:8]
        return time

    def date_now(self):
        date = JalaliDate.today()
        return date


    def clockOpen(self):
        connect = sqlite3.connect('D:\Project-VSCode\Face-Recognition-RealTime-Attendance-System-SQL-main\database.db')
        cursor = connect.cursor()
        cursor.execute(f'SELECT clocks FROM database WHERE names = ? AND statuses = "OPEN" AND clocks < (SELECT MAX(clocks) FROM database WHERE statuses = "EXIT");',(self.name,))
        connect.commit()
        rows = cursor.fetchall()
        connect.close()
        return rows
        
    def clockExit(self):
        connect = sqlite3.connect('D:\Project-VSCode\Face-Recognition-RealTime-Attendance-System-SQL-main\database.db')
        cursor = connect.cursor()
        cursor.execute(f'SELECT clocks FROM database WHERE names = ? AND statuses = "EXIT" AND id = (SELECT MAX(id) FROM database WHERE names = ? AND statuses = "EXIT");',(self.name,self.name))
        connect.commit()
        rows = cursor.fetchall()
        connect.close()
        return rows

    def insertOpen(self):
        date = self.date_now()
        clock = self.time_now()
        connect = sqlite3.connect('D:\Project-VSCode\Face-Recognition-RealTime-Attendance-System-SQL-main\database.db')
        cursor = connect.cursor()
        cursor.execute(f'INSERT INTO database (names, dates, clocks, statuses) VALUES (?,?,?,"OPEN");',(str(self.name),str(date),str(clock)))
        connect.commit()
        connect.close()

    def insertExit(self):
        connect = sqlite3.connect('D:\Project-VSCode\Face-Recognition-RealTime-Attendance-System-SQL-main\database.db')
        cursor = connect.cursor()
        date = str(self.date_now())
        clock = str(self.time_now())
        cursor.execute(f'INSERT INTO database (names, dates, clocks, statuses) VALUES (?,?,?,"EXIT");',(str(self.name),str(date),str(clock)))
        connect.commit()
        connect.close()

    def checkOpen(self):
        date = self.date_now()
        connect = sqlite3.connect('D:\Project-VSCode\Face-Recognition-RealTime-Attendance-System-SQL-main\database.db')
        cursor = connect.cursor()
        cursor.execute(f'SELECT id FROM database WHERE names = ? AND statuses = "OPEN" AND dates = ?;',(self.name,str(date)))
        connect.commit()
        rows = cursor.fetchall()
        connect.close()
        if rows:
            return True
        else:
            return False

    def checkExit(self):
        date = self.date_now()
        connect = sqlite3.connect('D:\Project-VSCode\Face-Recognition-RealTime-Attendance-System-SQL-main\database.db')
        cursor = connect.cursor()
        cursor.execute(f'SELECT statuses FROM database WHERE names = ? AND statuses = "EXIT" AND dates = ?;',(self.name,str(date)))
        connect.commit()
        rows = cursor.fetchall()
        connect.close()
        if rows:
            return True
        else:
            return False
        
    def insert(self):

        FMT = '%H:%M:%S'

        if self.checkOpen() and not self.checkExit():
            self.insertExit()
        elif self.checkOpen() and self.checkExit():
            self.insertOpen()


if __name__ == '__main__':
    FunctionsQuery('Mahdi').create_database()