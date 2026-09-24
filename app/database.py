import sqlite3
from datetime import datetime


DATABASE_PATH = "predictions.db"

MODEL_VERSION = "resnet18-v1"


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def init_database():
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'accepted',
            model_version TEXT NOT NULL DEFAULT 'resnet18-v1',
            created_at TEXT NOT NULL
        )
    """)

    # Add new columns if the database was created using the older schema.
    columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(predictions)"
        ).fetchall()
    }

    if "status" not in columns:
        connection.execute(
            "ALTER TABLE predictions "
            "ADD COLUMN status TEXT NOT NULL DEFAULT 'accepted'"
        )

    if "model_version" not in columns:
        connection.execute(
            "ALTER TABLE predictions "
            "ADD COLUMN model_version TEXT NOT NULL DEFAULT 'resnet18-v1'"
        )

    connection.commit()
    connection.close()


def save_prediction(
    filename,
    prediction,
    confidence,
    status,
    model_version=MODEL_VERSION
):
    connection = get_connection()

    connection.execute(
        """
        INSERT INTO predictions
        (
            filename,
            prediction,
            confidence,
            status,
            model_version,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            prediction,
            confidence,
            status,
            model_version,
            datetime.utcnow().isoformat()
        )
    )

    connection.commit()
    connection.close()


def get_predictions(
    prediction=None,
    min_confidence=None
):
    connection = get_connection()

    query = """
        SELECT
            id,
            filename,
            prediction,
            confidence,
            status,
            model_version,
            created_at
        FROM predictions
        WHERE 1=1
    """

    parameters = []

    if prediction:
        query += " AND prediction = ?"
        parameters.append(prediction)

    if min_confidence is not None:
        query += " AND confidence >= ?"
        parameters.append(min_confidence)

    query += " ORDER BY id DESC"

    cursor = connection.execute(query, parameters)

    rows = cursor.fetchall()
    connection.close()

    return [
        {
            "id": row[0],
            "filename": row[1],
            "prediction": row[2],
            "confidence": row[3],
            "status": row[4],
            "model_version": row[5],
            "created_at": row[6]
        }
        for row in rows
    ]