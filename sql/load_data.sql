-- Olist Data Loading Script
-- Note: This script assumes 'local_infile' is enabled in your MySQL configuration.
-- Replace '{BASE_PATH}' with your absolute project directory, or use relative paths if running from the data/raw directory.

USE olist;

-- 1. Load Customers
LOAD DATA LOCAL INFILE 'd:/DATA SCIENCE/ecommerce_delay_prediction/data/raw/olist_customers_dataset.csv'
INTO TABLE customers
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- 2. Load Geolocation
LOAD DATA LOCAL INFILE 'd:/DATA SCIENCE/ecommerce_delay_prediction/data/raw/olist_geolocation_dataset.csv'
INTO TABLE geolocation
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- 3. Load Sellers
LOAD DATA LOCAL INFILE 'd:/DATA SCIENCE/ecommerce_delay_prediction/data/raw/olist_sellers_dataset.csv'
INTO TABLE sellers
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- 4. Load Products
LOAD DATA LOCAL INFILE 'd:/DATA SCIENCE/ecommerce_delay_prediction/data/raw/olist_products_dataset.csv'
INTO TABLE products
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- 5. Load Orders
-- Note: MySQL needs NULL for empty strings. We use SET to handle empty timestamps correctly.
LOAD DATA LOCAL INFILE 'd:/DATA SCIENCE/ecommerce_delay_prediction/data/raw/olist_orders_dataset.csv'
INTO TABLE orders
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(order_id, customer_id, order_status, @v_purchase, @v_approved, @v_carrier, @v_delivered, @v_estimated)
SET 
    order_purchase_timestamp = NULLIF(@v_purchase, ''),
    order_approved_at = NULLIF(@v_approved, ''),
    order_delivered_carrier_date = NULLIF(@v_carrier, ''),
    order_delivered_customer_date = NULLIF(@v_delivered, ''),
    order_estimated_delivery_date = NULLIF(@v_estimated, '');

-- 6. Load Order Items
LOAD DATA LOCAL INFILE 'd:/DATA SCIENCE/ecommerce_delay_prediction/data/raw/olist_order_items_dataset.csv'
INTO TABLE order_items
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- 7. Load Order Payments
LOAD DATA LOCAL INFILE 'd:/DATA SCIENCE/ecommerce_delay_prediction/data/raw/olist_order_payments_dataset.csv'
INTO TABLE order_payments
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- 8. Load Order Reviews
LOAD DATA LOCAL INFILE 'd:/DATA SCIENCE/ecommerce_delay_prediction/data/raw/olist_order_reviews_dataset.csv'
INTO TABLE order_reviews
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(review_id, order_id, review_score, review_comment_title, review_comment_message, @v_creation, @v_answer)
SET 
    review_creation_date = NULLIF(@v_creation, ''),
    review_answer_timestamp = NULLIF(@v_answer, '');

-- 9. Load Category Translation
LOAD DATA LOCAL INFILE 'd:/DATA SCIENCE/ecommerce_delay_prediction/data/raw/product_category_name_translation.csv'
INTO TABLE product_category_name_translation
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
