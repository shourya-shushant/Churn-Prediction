-- name: Overall churn rate
SELECT COUNT(*) AS customers,
       SUM(churn = 'Yes') AS churned,
       ROUND(100.0 * SUM(churn = 'Yes') / COUNT(*), 1) AS churn_rate_pct
FROM customers;

-- name: Churn rate by contract type
SELECT contract,
       COUNT(*) AS customers,
       ROUND(100.0 * SUM(churn = 'Yes') / COUNT(*), 1) AS churn_rate_pct,
       ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;

-- name: Churn rate by internet service and tech support
SELECT internet_service, tech_support,
       COUNT(*) AS customers,
       ROUND(100.0 * SUM(churn = 'Yes') / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY internet_service, tech_support
ORDER BY churn_rate_pct DESC;

-- name: Churn by tenure bucket
SELECT CASE WHEN tenure <= 12 THEN '0-12m'
            WHEN tenure <= 24 THEN '13-24m'
            WHEN tenure <= 48 THEN '25-48m'
            ELSE '49-72m' END AS tenure_bucket,
       COUNT(*) AS customers,
       ROUND(100.0 * SUM(churn = 'Yes') / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY tenure_bucket
ORDER BY MIN(tenure);

-- name: Monthly revenue lost to churn by payment method
SELECT payment_method,
       SUM(churn = 'Yes') AS churned_customers,
       ROUND(SUM(CASE WHEN churn = 'Yes' THEN monthly_charges ELSE 0 END), 2) AS monthly_revenue_lost
FROM customers
GROUP BY payment_method
ORDER BY monthly_revenue_lost DESC;
