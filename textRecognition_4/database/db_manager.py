import sqlite3
from datetime import datetime, timedelta
import secrets
from database.user import User


class DatabaseManager:
    def __init__(self, db_name):
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT,
            first_name TEXT,
            last_name TEXT,
            email TEXT
        )
        ''')

        # Create images table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            filename TEXT,
            upload_date TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')

        # Create analysis_results table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY,
            image_id INTEGER,
            result TEXT,
            analysis_date TEXT,
            FOREIGN KEY (image_id) REFERENCES images (id)
        )
        ''')

        # Create sessions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            username TEXT,
            session_token TEXT,
            expiry_date TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')

        conn.commit()
        conn.close()

    def create_session(self, user_id, username):
        session_token = secrets.token_hex(16)
        expiry_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")

        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (user_id, username, session_token, expiry_date)
            VALUES (?, ?, ?, ?)
        """, (user_id, username, session_token, expiry_date))
        conn.commit()
        conn.close()

        return session_token

    def check_session(self, page):
        session_token = page.client_storage.get("session_token")
        if session_token:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, username FROM sessions
                WHERE session_token = ? AND expiry_date > ?
            """, (session_token, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            result = cursor.fetchone()
            conn.close()

            if result:
                return result
        return None

    def authenticate_user(self, username, password):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                       (username, password))
        user_data = cursor.fetchone()
        conn.close()

        if user_data:
            return User(user_data[0], user_data[1], user_data[3], user_data[4], user_data[5])
        return None

    def register_user(self, username, password, first_name, last_name, email):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO users (username, password, first_name, last_name, email)
                VALUES (?, ?, ?, ?, ?)
                """, (username, password, first_name, last_name, email))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def save_image(self, user_id, filename):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO images (user_id, filename, upload_date)
            VALUES (?, ?, ?)
            """, (user_id, filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        image_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return image_id

    def save_analysis_result(self, image_id, result):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analysis_results (image_id, result, analysis_date)
            VALUES (?, ?, ?)
            """, (image_id, result, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

    def get_user_history(self, user_id):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT i.filename, ar.result, ar.analysis_date
            FROM images i
            JOIN analysis_results ar ON i.id = ar.image_id
            WHERE i.user_id = ?
            ORDER BY ar.analysis_date DESC
            """, (user_id,))
        history = cursor.fetchall()
        conn.close()
        return history
