import os
import psycopg

def get_connection():
    return psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=5432,
        dbname="ttc_ml",
        user="ttc",
        password="ttc_dev",
    )


def create_predictions_table():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    day TEXT NOT NULL,
                    line TEXT NOT NULL,
                    code TEXT NOT NULL,
                    bound TEXT NOT NULL,
                    month SMALLINT NOT NULL CHECK (month BETWEEN 1 AND 12),
                    hour SMALLINT NOT NULL CHECK (hour BETWEEN 0 AND 23),
                    predicted_delay_minutes DOUBLE PRECISION NOT NULL
                );
                """
            )


def save_prediction(
    day: str,
    line: str,
    code: str,
    bound: str,
    month: int,
    hour: int,
    predicted_delay_minutes: float,
):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO predictions (
                    day,
                    line,
                    code,
                    bound,
                    month,
                    hour,
                    predicted_delay_minutes
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    day,
                    line,
                    code,
                    bound,
                    month,
                    hour,
                    predicted_delay_minutes,
                ),
            )

