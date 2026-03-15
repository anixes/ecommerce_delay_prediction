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
        MAX(p.product_category_name) AS product_category,
        SUM(p.product_weight_g) AS total_weight_g,
        AVG(p.product_length_cm * p.product_height_cm * p.product_width_cm) AS avg_product_volume_cm3,
        -- NEW: Product listing quality
        AVG(COALESCE(p.product_photos_qty, 0)) AS avg_product_photos,
        AVG(COALESCE(p.product_description_lenght, 0)) AS avg_description_length,
        -- NEW: Shipping limit gap (seller's committed buffer in days)
        DATEDIFF(MAX(oi.shipping_limit_date), MIN(oi.shipping_limit_date)) AS shipping_limit_spread,
        MAX(oi.shipping_limit_date) AS max_shipping_limit_date
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
    -- Pre-calculate the is_late flag
    SELECT 
        o.order_id,
        o.customer_id,
        o.order_purchase_timestamp,
        CASE 
            WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 
            ELSE 0 
        END AS is_late,
        o.order_estimated_delivery_date
    FROM orders o
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
),
seller_review_history AS (
    -- Average review score for seller leading up to the order
    SELECT
        bo.order_id,
        ois.seller_id,
        AVG(rv.review_score) OVER(
            PARTITION BY ois.seller_id 
            ORDER BY bo.order_purchase_timestamp 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS seller_avg_review_score
    FROM base_orders bo
    JOIN order_item_summary ois ON bo.order_id = ois.order_id
    LEFT JOIN order_reviews rv ON bo.order_id = rv.order_id
),
seller_history AS (
    -- Calculate historical delay rate for each seller UP TO the point of the current order
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
    CASE WHEN c.customer_state = s.seller_state THEN 1 ELSE 0 END AS is_same_state,
    
    -- Haversine Distance Calculation
    (6371 * 2 * ASIN(SQRT(
        POWER(SIN((gc.lat - gs.lat) * PI() / 180 / 2), 2) +
        COS(gc.lat * PI() / 180) * COS(gs.lat * PI() / 180) *
        POWER(SIN((gc.lng - gs.lng) * PI() / 180 / 2), 2)
    ))) AS distance_km,
    
    -- Target Variable
    o.is_late,
    
    -- Temporal Features
    o.order_purchase_timestamp,
    WEEKDAY(o.order_purchase_timestamp) AS purchase_day_of_week,
    HOUR(o.order_purchase_timestamp) AS purchase_hour,
    MONTH(o.order_purchase_timestamp) AS purchase_month,
    
    -- Metrics
    ois.total_items,
    ois.total_price,
    ois.total_freight,
    ois.total_weight_g,
    ois.avg_product_volume_cm3,
    ois.total_freight / NULLIF(ois.total_price, 0) AS freight_ratio,
    ois.product_category,
    ps.total_payment,
    ps.max_installments,
    ps.primary_payment_type,
    
    -- NEW: Product listing quality
    ois.avg_product_photos,
    ois.avg_description_length,
    
    -- NEW: Seller shipping buffer (days between purchase and shipping limit)
    DATEDIFF(ois.max_shipping_limit_date, o.order_purchase_timestamp) AS seller_shipping_buffer_days,
    ois.shipping_limit_spread,
    
    -- Business Logic Metrics & History
    DATEDIFF(o.order_estimated_delivery_date, o.order_purchase_timestamp) AS lead_time_days_estimated,
    COALESCE(sh.seller_historical_delay_rate, 0) AS seller_historical_delay_rate,
    COALESCE(srh.seller_avg_review_score, 0) AS seller_avg_review_score,
    COALESCE(sb.seller_state_backlog, 0) AS seller_state_backlog,
    COALESCE(cb.customer_state_backlog, 0) AS customer_state_backlog,
    COALESCE(ch.customer_total_orders, 1) AS customer_total_orders,
    COALESCE(srp.seller_recent_delay_rate, 0) AS seller_recent_delay_rate,
    
    -- NEW: Seller burstiness (time since last order in seconds)
    COALESCE(slb.seconds_since_last_seller_order, 2592000) AS seconds_since_last_seller_order,
    
    -- NEW: Seller intensity (Backlog relative to their average volume)
    COALESCE(sb.seller_state_backlog, 0) / NULLIF(COUNT(ois.seller_id) OVER(PARTITION BY ois.seller_id), 0) AS seller_intensity_score,
    
    -- NEW: Route-level historical delay rate (seller_state -> customer_state)
    COALESCE(rd.route_delay_rate, 0) AS route_delay_rate

FROM base_orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_item_summary ois ON o.order_id = ois.order_id
JOIN sellers s ON ois.seller_id = s.seller_id
LEFT JOIN payment_summary ps ON o.order_id = ps.order_id
LEFT JOIN seller_history sh ON o.order_id = sh.order_id
LEFT JOIN seller_review_history srh ON o.order_id = srh.order_id
LEFT JOIN zip_coordinates gc ON c.customer_zip_code_prefix = gc.zip_code
LEFT JOIN zip_coordinates gs ON s.seller_zip_code_prefix = gs.zip_code
LEFT JOIN (
    -- NEW: Time between seller orders
    SELECT 
        bo.order_id,
        TIMESTAMPDIFF(SECOND, 
            LAG(bo.order_purchase_timestamp) OVER(PARTITION BY ois.seller_id ORDER BY bo.order_purchase_timestamp),
            bo.order_purchase_timestamp
        ) AS seconds_since_last_seller_order
    FROM base_orders bo
    JOIN order_item_summary ois ON bo.order_id = ois.order_id
) slb ON o.order_id = slb.order_id
LEFT JOIN (
    -- Customer Purchase History: Count total orders for this unique customer identity
    -- Note: customers table has customer_unique_id for this, but base_orders uses customer_id (order-specific)
    -- Joining back to customers to get unique_id
    SELECT 
        bo.order_id,
        COUNT(bo.order_id) OVER(
            PARTITION BY cu.customer_unique_id 
            ORDER BY bo.order_purchase_timestamp 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) + 1 AS customer_total_orders
    FROM base_orders bo
    JOIN customers cu ON bo.customer_id = cu.customer_id
) ch ON o.order_id = ch.order_id
LEFT JOIN (
    -- Recent Seller Performance: Average delay in the last 5 orders
    SELECT
        bo.order_id,
        AVG(bo.is_late) OVER(
            PARTITION BY ois.seller_id 
            ORDER BY bo.order_purchase_timestamp 
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS seller_recent_delay_rate
    FROM base_orders bo
    JOIN order_item_summary ois ON bo.order_id = ois.order_id
) srp ON o.order_id = srp.order_id
LEFT JOIN (
    -- Backlog: Order volume in the seller state in the last 7 days
    SELECT 
        bo.order_id,
        COUNT(bo.order_id) OVER(
            PARTITION BY s.seller_state 
            ORDER BY bo.order_purchase_timestamp 
            RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND INTERVAL 1 SECOND PRECEDING
        ) AS seller_state_backlog
    FROM base_orders bo
    JOIN order_item_summary ois ON bo.order_id = ois.order_id
    JOIN sellers s ON ois.seller_id = s.seller_id
) sb ON o.order_id = sb.order_id
LEFT JOIN (
    -- Backlog: Order volume in the customer state in the last 3 days
    SELECT 
        bo.order_id,
        COUNT(bo.order_id) OVER(
            PARTITION BY c.customer_state 
            ORDER BY bo.order_purchase_timestamp 
            RANGE BETWEEN INTERVAL 3 DAY PRECEDING AND INTERVAL 1 SECOND PRECEDING
        ) AS customer_state_backlog
    FROM base_orders bo
    JOIN customers c ON bo.customer_id = c.customer_id
) cb ON o.order_id = cb.order_id
LEFT JOIN (
    -- NEW: Route-level delay rate (seller_state -> customer_state historical performance)
    SELECT
        bo.order_id,
        AVG(bo.is_late) OVER(
            PARTITION BY s.seller_state, c.customer_state
            ORDER BY bo.order_purchase_timestamp
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS route_delay_rate
    FROM base_orders bo
    JOIN order_item_summary ois ON bo.order_id = ois.order_id
    JOIN sellers s ON ois.seller_id = s.seller_id
    JOIN customers c ON bo.customer_id = c.customer_id
) rd ON o.order_id = rd.order_id;
