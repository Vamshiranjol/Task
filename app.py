from flask import Flask
import mysql.connector
from flask import jsonify

app = Flask(__name__)

# Function to connect to MySQL
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",   
        password="vamshi",  
        database="Vamshi"
    )

@app.route("/locations", methods=["GET"])
def get_locations():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM locations")
    locations = cursor.fetchall()
    conn.close()
    return jsonify(locations)  # Returns JSON response

if __name__ == "__main__":
    app.run(debug=True)
    