-- =============================================================================
-- NBA DRAFT ANALYTICS ENGINE: Database Schema & Master View
-- =============================================================================
-- Description: This script initializes the raw tables for NBA Draft history, 
-- NCAA college stats, physical combine measurements, and NBA career stats/awards. 
-- It then compiles this data into a master view (`vw_draft_master`) to calculate 
-- Value Over Expectation (VOE) and assign ML target labels (Bust, Steal, Star).
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. DROP EXISTING OBJECTS (To allow clean re-runs)
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS vw_draft_master CASCADE;
DROP TABLE IF EXISTS raw_nba_career_stats CASCADE;
DROP TABLE IF EXISTS raw_ncaa_stats CASCADE;
DROP TABLE IF EXISTS raw_player_awards CASCADE;
DROP TABLE IF EXISTS raw_combine_stats CASCADE;
DROP TABLE IF EXISTS raw_draft_history CASCADE;

-- -----------------------------------------------------------------------------
-- 2. CREATE RAW DATA TABLES
-- -----------------------------------------------------------------------------

-- DRAFT HISTORY (The "Spine" of our Database)
CREATE TABLE raw_draft_history (
    person_id INT PRIMARY KEY,
    player_name VARCHAR(100),
    season INT,
    round_number INT,
    round_pick INT,
    overall_pick INT,
    draft_type VARCHAR(50),
    team_id BIGINT,
    team_city VARCHAR(100),
    team_name VARCHAR(100),
    team_abbreviation VARCHAR(10),
    organization VARCHAR(100),
    organization_type VARCHAR(50),
    player_profile_flag INT
);

-- NBA CAREER STATS (Year-by-year performance)
CREATE TABLE raw_nba_career_stats (
    player_id INT,
    season_id VARCHAR(20),
    league_id INT,
    team_id BIGINT,
    team_abbreviation VARCHAR(10),
    player_age NUMERIC,
    gp INT,
    gs INT,
    min INT,
    fgm INT,
    fga INT,
    fg_pct NUMERIC,
    fg3m INT,
    fg3a INT,
    fg3_pct NUMERIC,
    ftm INT,
    fta INT,
    ft_pct NUMERIC,
    oreb INT,
    dreb INT,
    reb INT,
    ast INT,
    stl INT,
    blk INT,
    tov INT,
    pf INT,
    pts INT,
    person_id INT REFERENCES raw_draft_history(person_id)
);

-- NCAA SCRAPED STATS (Advanced metrics from college)
CREATE TABLE raw_ncaa_stats (
    season VARCHAR(50),
    team VARCHAR(100),
    conf VARCHAR(50),
    class VARCHAR(20),
    pos VARCHAR(20),
    g NUMERIC,
    gs NUMERIC,
    mp NUMERIC,
    per NUMERIC,
    ts_pct NUMERIC,
    "3par" NUMERIC,
    ftr NUMERIC,
    pprod NUMERIC,
    orb_pct NUMERIC,
    drb_pct NUMERIC,
    trb_pct NUMERIC,
    ast_pct NUMERIC,
    stl_pct NUMERIC,
    blk_pct NUMERIC,
    tov_pct NUMERIC,
    usg_pct NUMERIC,
    ows NUMERIC,
    dws NUMERIC,
    ws NUMERIC,
    ws_40 NUMERIC,
    obpm NUMERIC,
    dbpm NUMERIC,
    bpm NUMERIC,
    awards VARCHAR(255),
    player_name VARCHAR(100)
);

-- PLAYER AWARDS (Used to identify Stars and Superstars)
CREATE TABLE raw_player_awards (
    person_id VARCHAR(255),
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    team VARCHAR(255),
    description VARCHAR(255),
    all_nba_team_number VARCHAR(255),
    season VARCHAR(255),
    month VARCHAR(255),
    week VARCHAR(255),
    conference VARCHAR(255),
    type VARCHAR(255),
    subtype1 VARCHAR(255),
    subtype2 VARCHAR(255),
    subtype3 VARCHAR(255),
    person_id1 VARCHAR(255)
);

-- NBA COMBINE STATS (Physical measurements)
CREATE TABLE raw_combine_stats (
    player_name VARCHAR(100),
    height_no_shoes NUMERIC,
    wingspan NUMERIC,
    max_vertical NUMERIC
);

-- -----------------------------------------------------------------------------
-- 3. CREATE MASTER MACHINE LEARNING VIEW
-- -----------------------------------------------------------------------------
CREATE VIEW vw_draft_master AS
WITH nba_career_totals AS (
    -- Aggregate total career games and minutes for each drafted player
    SELECT 
        person_id, 
        SUM(gp) AS career_games, 
        SUM(min) AS career_minutes
    FROM raw_nba_career_stats 
    GROUP BY person_id
),
draft_curve AS (
    -- Calculate the historical expected minutes per season for each draft slot
    SELECT 
        d.overall_pick, 
        AVG(COALESCE(n.career_minutes, 0) / (2024 - d.season)) AS expected_mins_per_season
    FROM raw_draft_history d
    LEFT JOIN nba_career_totals n ON d.person_id = n.person_id
    GROUP BY d.overall_pick
),
player_hardware AS (
    -- Count major accolades to flag Elite outcomes
    SELECT 
        CAST(person_id AS INT) AS person_id,
        SUM(CASE WHEN description ILIKE '%Most Valuable Player%' THEN 1 ELSE 0 END) AS mvp_count,
        SUM(CASE WHEN description ILIKE '%All-NBA%' THEN 1 ELSE 0 END) AS all_nba_count,
        SUM(CASE WHEN description ILIKE '%All-Star%' THEN 1 ELSE 0 END) AS all_star_count
    FROM raw_player_awards 
    GROUP BY CAST(person_id AS INT)
)
SELECT 
    d.person_id,
    d.player_name,
    d.overall_pick,
    c.class AS ncaa_class,
    
    -- ADVANCED NCAA METRICS
    (c.stl_pct + c.blk_pct) AS ncaa_stocks,
    CASE WHEN c.tov_pct > 0 THEN (c.ast_pct / c.tov_pct) ELSE 0 END AS ncaa_ast_tov_ratio,
    c.per AS ncaa_per,
    c.bpm AS ncaa_bpm,
    c.ts_pct AS ncaa_ts_pct,
    c.usg_pct AS ncaa_usg_pct,
    
    -- PHYSICALS (COALESCE handles missing combine data by providing positional averages)
    COALESCE(co.height_no_shoes, 78) AS height_in, 
    COALESCE(co.wingspan - co.height_no_shoes, 3.5) AS ape_index, 
    COALESCE(co.max_vertical, 32) AS vertical_in,
    
    -- TARGET LABEL DEFINITIONS (Bust, Steal, Star, Superstar, Expected)
    CASE 
        WHEN COALESCE(ph.mvp_count, 0) > 0 OR COALESCE(ph.all_nba_count, 0) >= 2 OR COALESCE(ph.all_star_count, 0) >= 3 THEN 'Superstar'
        WHEN COALESCE(ph.all_nba_count, 0) > 0 OR COALESCE(ph.all_star_count, 0) > 0 THEN 'Star'
        WHEN d.overall_pick <= 14 AND ((COALESCE(n.career_minutes, 0) / (2024 - d.season)) - dc.expected_mins_per_season) < -700 THEN 'Bust'
        WHEN d.overall_pick <= 14 AND ((COALESCE(n.career_minutes, 0) / (2024 - d.season)) - dc.expected_mins_per_season) > 1200 THEN 'Star'
        WHEN d.overall_pick BETWEEN 15 AND 30 AND ((COALESCE(n.career_minutes, 0) / (2024 - d.season)) - dc.expected_mins_per_season) > 700 THEN 'Steal'
        WHEN d.overall_pick > 30 AND ((COALESCE(n.career_minutes, 0) / (2024 - d.season)) - dc.expected_mins_per_season) > 500 THEN 'Steal'
        ELSE 'Expected'
    END AS draft_value_label

FROM raw_draft_history d
LEFT JOIN raw_ncaa_stats c ON d.player_name = c.player_name
LEFT JOIN nba_career_totals n ON d.person_id = n.person_id
LEFT JOIN draft_curve dc ON d.overall_pick = dc.overall_pick
LEFT JOIN player_hardware ph ON d.person_id = ph.person_id
LEFT JOIN raw_combine_stats co ON d.player_name = co.player_name;