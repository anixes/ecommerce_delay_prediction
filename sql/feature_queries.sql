-- Analytical Dataset Construction
-- Goal: Transform relational tables into a single order-level dataset for machine learning.

USE olist;

DROP VIEW IF EXISTS analytical_dataset;

CREATE VIEW analytical_dataset AS
WITH zip_coordinates AS (
    -- Consolidate geolocation: take the average coordinates per zip code prefix
    -- This prevents many-to-many join explosions
    SELECT 
        geolocation_zip_code_prefix AS zip_code,
        AVG(geolocation_lat) AS lat,
        AVG(geolocation_lng) AS lng
    FROM geolocation
    GROUP BY geolocation_zip_code_prefix
),
order_item_summary AS (
    -- Aggregate item-level data to order-level
    SELECT 
        oi.order_id,
        COUNT(oi.order_item_id) AS total_items,
        SUM(oi.price) AS total_price,
        SUM(oi.freight_value) AS total_freight,
        MAX(oi.seller_id) AS seller_id,
        MAX(p.product_category_name) AS product_category
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    GROUP BY oi.order_id
),
payment_summary AS (
    SELECT 
        order_id,
        SUM(payment_value) AS total_payment,
        MAX(payment_installments) AS max_installments,
        MAX(payment_type) AS primary_payment_type
    FROM order_payments
    GROUP BY order_id
),
base_orders AS (
    -- Pre-calculate the is_late flag so we can use it in window functions
    SELECT 
        o.order_id,
        o.customer_id,
        o.order_purchase_timestamp,
        CASE 
            WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 
            ELSE 0 
        END AS is_late,
        o.order_estimated_delivery_date,
        o.order_delivered_customer_date
    FROM orders o
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
),
seller_history AS (
    -- Calculate historical delay rate for each seller UP TO the point of the current order
    -- This prevents target leakage (using future delays to predict current delays)
    SELECT
        bo.order_id,
        ois.seller_id,
        AVG(bo.is_late) OVER(
            PARTITION BY ois.seller_id 
            ORDER BY bo.order_purchase_timestamp 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS seller_historical_delay_rate
    FROM base_orders bo
    JOIN order_item_summary ois ON bo.order_id = ois.order_id
)
SELECT 
    o.order_id,
    o.customer_id,
    c.customer_city,
    c.customer_state,
    s.seller_city,
    s.seller_state,
    
    -- Haversine Distance Calculation (approximate distance in KM)
    (6371 * 2 * ASIN(SQRT(
        POWER(SIN((gc.lat - gs.lat) * PI() / 180 / 2), 2) +
        COS(gc.lat * PI() / 180) * COS(gs.lat * PI() / 180) *
        POWER(SIN((gc.lng - gs.lng) * PI() / 180 / 2), 2)
    ))) AS distance_km,
    
    -- Target Variable
    o.is_late,
    
    -- Temporal Features
    o.order_purchase_timestamp,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    WEEKDAY(o.order_purchase_timestamp) AS purchase_day_of_week,
    HOUR(o.order_purchase_timestamp) AS purchase_hour,
    
    -- Metrics
    ois.total_items,
    ois.total_price,
    ois.total_freight,
    ois.product_category,
    ps.total_payment,
    ps.max_installments,
    ps.primary_payment_type,
    
    -- Business Logic Metrics & Seller History
    DATEDIFF(o.order_estimated_delivery_date, o.order_purchase_timestamp) AS lead_time_days_estimated,
    COALESCE(sh.seller_historical_delay_rate, 0) AS seller_historical_delay_rate
    
FROM base_orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_item_summary ois ON o.order_id = ois.order_id
JOIN sellers s ON ois.seller_id = s.seller_id
LEFT JOIN payment_summary ps ON o.order_id = ps.order_id
LEFT JOIN seller_history sh ON o.order_id = sh.order_id
-- Join coordinates for customer and seller
LEFT JOIN zip_coordinates gc ON c.customer_zip_code_prefix = gc.zip_code
LEFT JOIN zip_coordinates gs ON s.seller_zip_code_prefix = gs.zip_code;
