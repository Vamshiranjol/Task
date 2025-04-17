from flask import Flask, render_template, jsonify
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Function to connect to MySQL
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",   
            password="vamshi",  
            database="Vamshi"
        )
        return conn
    except Error as e:
        print(f"Error: {e}")
        return None

# Route to get all locations
@app.route("/locations", methods=["GET"])
def get_locations():
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500
    
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM locations")
    locations = cursor.fetchall()
    conn.close()
    
    return jsonify(locations)

# Route to get a specific location by ID
@app.route("/locations/<int:id>", methods=["GET"])
def get_location(id):
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500
    
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM locations WHERE id = %s", (id,))
    location = cursor.fetchone()
    conn.close()
    
    if location:
        return jsonify(location)
    else:
        return jsonify({"error": "Location not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
