import sqlite3

def main():
    conn = sqlite3.connect('Student.db')
    print('Connection successfully established to Student Database')
    cur = conn.cursor()
     
    cur.execute('Drop table Student_data')
    print('Student_data table dropped successfully')
    cur.execute('Create table Student_data (\
              Roll_No Number,\
              Name Varchar2(255));')
    print('Student_data table created succefully')
    cur.execute("Insert into Student_data Values (1,'Rahul')")
    cur.execute("Insert into Student_data Values (2,'Priyanka')")
    print('Data inserted into Student_data table successfully') 
    cur.execute('Select Roll_No, Name from Student_data')
    for row in cur:
        print(row)
    
    print('Data retrieved successfully from Student_data')
    
    conn.close()
if __name__ == '__main__':
    main()
