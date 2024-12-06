-- Step 1: Create a new database
CREATE DATABASE demo_db;

-- Step 2: Connect to the new database
\c demo_db

-- Step 3: Create the existing materials table within the database
CREATE TABLE materials (
    aconcrete_s FLOAT,
    bitumen FLOAT,
    brick FLOAT,
    concrete FLOAT,
    expclay_l FLOAT,
    expclay_s FLOAT,
    minwool_l FLOAT,
    mortar FLOAT,
    plasterboard FLOAT,
    polystyrene FLOAT,
    woodchip FLOAT,
    woodfiber_s FLOAT,
    aluminium FLOAT,
    glass FLOAT,
    mdf FLOAT,
    build_year INTEGER
);

-- Step 4: Create a new table for image, year, and area amount
CREATE TABLE house (
    image VARCHAR(1024),  -- Increased column size for storing the image reference or URL
    build_year INTEGER,         -- Year for the image
    area_amount FLOAT     -- Area amount in float
);
