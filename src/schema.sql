CREATE TABLE IF NOT EXISTS companies (
    id INT AUTO_INCREMENT PRIMARY KEY,
    website_uri VARCHAR(255) NOT NULL UNIQUE,
    company_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS company_info_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    company_id INT,
    type VARCHAR(50), -- public, private, non-profit
    location TEXT,
    phone_number VARCHAR(50),
    email_address VARCHAR(255),
    ceo_name VARCHAR(255),
    coo_name VARCHAR(255),
    cfo_name VARCHAR(255),
    cto_name VARCHAR(255),
    thinking_process TEXT,
    retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (company_id) REFERENCES companies(id) ON DELETE CASCADE
);
