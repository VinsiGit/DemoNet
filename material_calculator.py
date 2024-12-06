import pandas as pd
import numpy as np
from psycopg2 import connect
from dotenv import load_dotenv
import os

def get_materials_by_year_and_area(year, surface_area, db_host="localhost", db_port=15432, db_name="demo_db"):
    """
    Calculates the material amounts for a given year and surface area using data from the database.

    Args:
        year (int): The year for which materials are to be calculated.
        surface_area (float): The surface area to scale the materials.
        db_host (str): The host of the database. Default is "localhost".
        db_port (int): The port of the database. Default is 15432.
        db_name (str): The name of the database. Default is "demo_db".

    Returns:
        dict: A dictionary containing materials and their calculated amounts.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Fetch database connection parameters
    db_host = "localhost"
    db_port = 15432
    db_name = "demo_db"
    db_user = os.getenv('POSTGRES_USER')
    db_password = os.getenv('POSTGRES_PASSWORD')

    try:
        # Connect to the PostgreSQL database
        connection = connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )

        # Create a cursor to interact with the database
        cursor = connection.cursor()

        # Query to fetch materials for the specified year
        query = """
        SELECT build_year, aconcrete_s, bitumen, brick, concrete, expclay_s, minwool_l, 
               mortar, plasterboard, polystyrene, woodchip, woodfiber_s, aluminium, 
               glass, mdf
        FROM materials
        WHERE build_year = %s;
        """

        # Execute the query with the given year
        cursor.execute(query, (year,))
        result = cursor.fetchone()

        # Check if the result is empty
        if result is None:
            raise ValueError(f"No data found for the year {year}.")

        # Extract column names
        column_names = [desc[0] for desc in cursor.description]

        # Map result to column names
        material_data = dict(zip(column_names, result))

        # Calculate materials based on the surface area
        calculated_materials = {
            material: value * surface_area if material != "build_year" else value
            for material, value in material_data.items()
            if material != "build_year"
        }

        return calculated_materials

    except Exception as e:
        raise RuntimeError(f"Error retrieving materials: {e}")

    finally:
        # Ensure resources are released
        if cursor:
            cursor.close()
        if connection:
            connection.close()

