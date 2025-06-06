import sqlite3
from datetime import datetime
import os
class TaskManager:
    def __init__(self, db_path='tasks.db'):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
    
    def _create_tables(self):
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_id TEXT NOT NULL UNIQUE,
            audio_path TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            result_path TEXT
        )
        ''')
        self.conn.commit()
    
    def add_task(self, audio_id, audio_path):
        try:
            
            result_dir="/root/res/"
            result_path = os.path.join(result_dir, audio_id+".json")
            self.conn.execute(
                "INSERT INTO tasks (audio_id, audio_path, result_path) VALUES (?, ?, ?)",
                (audio_id, audio_path, result_path)
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            # 音频ID已存在  
            return False
    
    def update_audio_status(self, audio_id, status):
        update_data = {
            'status': status,
            'audio_id': audio_id
        }
        query = "UPDATE tasks SET " + ", ".join(f"{k}=?" for k in update_data if k != 'audio_id') + " WHERE audio_id=?"
        params = [v for k, v in update_data.items() if k != 'audio_id'] + [audio_id]
        
        self.conn.execute(query, params)
        self.conn.commit()
    

    def get_audio_path(self, audio_id):
        cursor = self.conn.execute("select audio_path, status from tasks where audio_id=?",
        (audio_id,))
        return cursor.fetchone()


    def get_audio_status(self, audio_id):
        cursor = self.conn.execute(
            "SELECT status, result_path FROM tasks WHERE audio_id=?",
            (audio_id,)
        )
        return cursor.fetchone()
    
    def close(self):
        self.conn.close()

if __name__ == "__main__":
    tm = TaskManager()
    tm.add_task(1,"/upload/1.mp3")
    print(tm.get_audio_status(1))
    tm.update_audio_status(1,"completed")
    print(tm.get_audio_status(1))