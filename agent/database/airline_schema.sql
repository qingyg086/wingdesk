-- =============================================================================
-- 航空客服数据库 airline（数据来源：τ-bench airline 领域，MIT 协议）
--
-- 设计要点：
-- 1. 编码和枚举值保持 τ-bench 英文原值（JFK、economy、mia_li_3668 …），
--    评测模式和产品模式共用一套数据，结果可以对比。
-- 2. *_zh 列是 MySQL 生成列（GENERATED ... VIRTUAL）：根据英文枚举实时算出中文，
--    不占存储、不会和英文值不一致。对照表来自 data/airline/enums_zh.json，
--    改了对照表要重新生成本文件。
-- 3. 库默认排序规则用 utf8mb4_bin（区分大小写的精确匹配），
--    和 τ-bench 原版"Python dict 按 key 精确查找"的语义一致。
-- 4. 原版一个 JSON 文档被拆成多张关系表；seq / user_seq 列用来保留原 JSON 里的列表顺序，
--    MySQLBackend 拼回 JSON 时要和原版顺序一致。
-- 5. 用户地址拆成 6 个普通列，不用 JSON 类型：MySQL 8 的 JSON 列会按"key 长度 + 字母序"
--    重新排序对象的 key，读出来的字段顺序和原版不同（往返校验因此失败过）。
-- 6. 补充列：payment_methods.status（代金券软删除）、airports.seq（保持原版机场顺序）。
--
-- 用法：python agent/database/import_airline.py（会自动执行本文件，可重复执行）
-- =============================================================================

CREATE DATABASE IF NOT EXISTS airline DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;
USE airline;

DROP TABLE IF EXISTS payment_history;
DROP TABLE IF EXISTS reservation_passengers;
DROP TABLE IF EXISTS reservation_flights;
DROP TABLE IF EXISTS reservations;
DROP TABLE IF EXISTS flight_instances;
DROP TABLE IF EXISTS flights;
DROP TABLE IF EXISTS airports;
DROP TABLE IF EXISTS saved_passengers;
DROP TABLE IF EXISTS payment_methods;
DROP TABLE IF EXISTS users;

-- ---------------------------------------------------------------- 用户
CREATE TABLE users (
    user_id     VARCHAR(32)  NOT NULL,
    first_name  VARCHAR(64)  NOT NULL,
    last_name   VARCHAR(64)  NOT NULL,
    email       VARCHAR(128) NOT NULL,
    dob         DATE         NOT NULL,
    address1    VARCHAR(128) NOT NULL,
    address2    VARCHAR(128) NOT NULL,
    city        VARCHAR(64)  NOT NULL,
    country     VARCHAR(64)  NOT NULL,
    state       VARCHAR(32)  NOT NULL,
    zip         VARCHAR(16)  NOT NULL,
    membership  VARCHAR(16)  NOT NULL COMMENT 'regular / silver / gold',
    membership_zh VARCHAR(16) GENERATED ALWAYS AS (CASE membership WHEN 'regular' THEN '普通会员' WHEN 'silver' THEN '银卡会员' WHEN 'gold' THEN '金卡会员' END) VIRTUAL,
    PRIMARY KEY (user_id)
) ENGINE=InnoDB;

CREATE TABLE payment_methods (
    payment_id  VARCHAR(32)   NOT NULL COMMENT '如 credit_card_4421486，全局唯一',
    user_id     VARCHAR(32)   NOT NULL,
    seq         INT           NOT NULL COMMENT '在用户档案 payment_methods 里的顺序',
    source      VARCHAR(16)   NOT NULL COMMENT 'credit_card / gift_card / certificate',
    brand       VARCHAR(16)   NULL     COMMENT '仅 credit_card',
    last_four   CHAR(4)       NULL     COMMENT '仅 credit_card',
    amount      DECIMAL(10,2) NULL     COMMENT '仅 gift_card / certificate：余额',
    status      VARCHAR(16)   NOT NULL DEFAULT 'active' COMMENT 'active / used。代金券用过后软删除（τ-bench 原版是从档案里删掉），保留支付流水的外键',
    source_zh VARCHAR(16) GENERATED ALWAYS AS (CASE source WHEN 'credit_card' THEN '信用卡' WHEN 'gift_card' THEN '礼品卡' WHEN 'certificate' THEN '旅行代金券' END) VIRTUAL,
    PRIMARY KEY (payment_id),
    KEY idx_pm_user (user_id, seq),
    CONSTRAINT fk_pm_user FOREIGN KEY (user_id) REFERENCES users (user_id)
) ENGINE=InnoDB;

CREATE TABLE saved_passengers (
    id          INT          NOT NULL AUTO_INCREMENT,
    user_id     VARCHAR(32)  NOT NULL,
    first_name  VARCHAR(64)  NOT NULL,
    last_name   VARCHAR(64)  NOT NULL,
    dob         DATE         NOT NULL,
    PRIMARY KEY (id),
    KEY idx_sp_user (user_id),
    CONSTRAINT fk_sp_user FOREIGN KEY (user_id) REFERENCES users (user_id)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------- 机场与航班
CREATE TABLE airports (
    code     CHAR(3)      NOT NULL,
    seq      INT          NOT NULL COMMENT 'τ-bench list_all_airports 的返回顺序',
    city     VARCHAR(64)  NOT NULL COMMENT '与 τ-bench list_all_airports 原样一致',
    city_zh  VARCHAR(32)  NOT NULL,
    name_en  VARCHAR(128) NOT NULL,
    name_zh  VARCHAR(64)  NOT NULL,
    PRIMARY KEY (code)
) ENGINE=InnoDB;

-- 航班"模板"：每个航班号固定的航线和计划时刻
CREATE TABLE flights (
    flight_number                VARCHAR(8)  NOT NULL,
    origin                       CHAR(3)     NOT NULL,
    destination                  CHAR(3)     NOT NULL,
    scheduled_departure_time_est VARCHAR(11) NOT NULL COMMENT 'HH:MM:SS',
    scheduled_arrival_time_est   VARCHAR(11) NOT NULL COMMENT 'HH:MM:SS，带 +1 表示次日到达',
    PRIMARY KEY (flight_number),
    KEY idx_flights_od (origin, destination),
    CONSTRAINT fk_flights_origin FOREIGN KEY (origin) REFERENCES airports (code),
    CONSTRAINT fk_flights_dest   FOREIGN KEY (destination) REFERENCES airports (code)
) ENGINE=InnoDB;

-- 航班"实例"：某航班号在某一天的状态。不同状态带的字段不同，不适用的列为 NULL：
--   available            -> seats_* / price_*
--   on time / delayed    -> estimated_departure / estimated_arrival
--   flying               -> actual_departure / estimated_arrival
--   landed               -> actual_departure / actual_arrival
--   cancelled            -> 只有 status
CREATE TABLE flight_instances (
    flight_number                VARCHAR(8)  NOT NULL,
    date                         DATE        NOT NULL,
    status                       VARCHAR(16) NOT NULL,
    status_zh VARCHAR(16) GENERATED ALWAYS AS (CASE status WHEN 'available' THEN '可预订' WHEN 'on time' THEN '准点' WHEN 'delayed' THEN '延误' WHEN 'flying' THEN '飞行中' WHEN 'landed' THEN '已降落' WHEN 'cancelled' THEN '已取消' END) VIRTUAL,
    seats_basic_economy          INT         NULL,
    seats_economy                INT         NULL,
    seats_business               INT         NULL,
    price_basic_economy          INT         NULL,
    price_economy                INT         NULL,
    price_business               INT         NULL,
    actual_departure_time_est    DATETIME    NULL,
    actual_arrival_time_est      DATETIME    NULL,
    estimated_departure_time_est DATETIME    NULL,
    estimated_arrival_time_est   DATETIME    NULL,
    PRIMARY KEY (flight_number, date),
    KEY idx_fi_date_status (date, status),
    CONSTRAINT fk_fi_flight FOREIGN KEY (flight_number) REFERENCES flights (flight_number)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------- 订单
CREATE TABLE reservations (
    reservation_id   CHAR(6)     NOT NULL,
    user_id          VARCHAR(32) NOT NULL,
    user_seq         INT         NOT NULL COMMENT '在用户档案 reservations 列表里的顺序',
    origin           CHAR(3)     NOT NULL,
    destination      CHAR(3)     NOT NULL,
    flight_type      VARCHAR(16) NOT NULL COMMENT 'one_way / round_trip',
    cabin            VARCHAR(16) NOT NULL COMMENT 'basic_economy / economy / business',
    created_at       DATETIME    NOT NULL,
    total_baggages   INT         NOT NULL,
    nonfree_baggages INT         NOT NULL,
    insurance        VARCHAR(3)  NOT NULL COMMENT 'yes / no',
    status           VARCHAR(16) NOT NULL DEFAULT 'active' COMMENT '原版数据没有该字段；cancel 后为 cancelled',
    flight_type_zh VARCHAR(16) GENERATED ALWAYS AS (CASE flight_type WHEN 'one_way' THEN '单程' WHEN 'round_trip' THEN '往返' END) VIRTUAL,
    cabin_zh VARCHAR(16) GENERATED ALWAYS AS (CASE cabin WHEN 'basic_economy' THEN '基础经济舱' WHEN 'economy' THEN '经济舱' WHEN 'business' THEN '商务舱' END) VIRTUAL,
    insurance_zh VARCHAR(8) GENERATED ALWAYS AS (CASE insurance WHEN 'yes' THEN '已购买' WHEN 'no' THEN '未购买' END) VIRTUAL,
    status_zh VARCHAR(16) GENERATED ALWAYS AS (CASE status WHEN 'active' THEN '有效' WHEN 'cancelled' THEN '已取消' END) VIRTUAL,
    PRIMARY KEY (reservation_id),
    KEY idx_res_user (user_id, user_seq),
    CONSTRAINT fk_res_user FOREIGN KEY (user_id) REFERENCES users (user_id)
) ENGINE=InnoDB;

CREATE TABLE reservation_flights (
    reservation_id CHAR(6)    NOT NULL,
    seq            INT        NOT NULL,
    flight_number  VARCHAR(8) NOT NULL,
    date           DATE       NOT NULL,
    origin         CHAR(3)    NOT NULL,
    destination    CHAR(3)    NOT NULL,
    price          INT        NOT NULL COMMENT '下单时的单人价格',
    PRIMARY KEY (reservation_id, seq),
    KEY idx_rf_flight (flight_number, date),
    CONSTRAINT fk_rf_res FOREIGN KEY (reservation_id) REFERENCES reservations (reservation_id),
    CONSTRAINT fk_rf_fi  FOREIGN KEY (flight_number, date) REFERENCES flight_instances (flight_number, date)
) ENGINE=InnoDB;

CREATE TABLE reservation_passengers (
    reservation_id CHAR(6)     NOT NULL,
    seq            INT         NOT NULL,
    first_name     VARCHAR(64) NOT NULL,
    last_name      VARCHAR(64) NOT NULL,
    dob            DATE        NOT NULL,
    PRIMARY KEY (reservation_id, seq),
    CONSTRAINT fk_rp_res FOREIGN KEY (reservation_id) REFERENCES reservations (reservation_id)
) ENGINE=InnoDB;

-- 支付流水：付款为正数；退票时 τ-bench 会追加一条同 payment_id 的负数记录（冲正）
CREATE TABLE payment_history (
    id             INT           NOT NULL AUTO_INCREMENT,
    reservation_id CHAR(6)       NOT NULL,
    payment_id     VARCHAR(32)   NOT NULL,
    amount         DECIMAL(10,2) NOT NULL,
    PRIMARY KEY (id),
    KEY idx_ph_res (reservation_id),
    CONSTRAINT fk_ph_res FOREIGN KEY (reservation_id) REFERENCES reservations (reservation_id),
    CONSTRAINT fk_ph_pm  FOREIGN KEY (payment_id) REFERENCES payment_methods (payment_id)
) ENGINE=InnoDB;
