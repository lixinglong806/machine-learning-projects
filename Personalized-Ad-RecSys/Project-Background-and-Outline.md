# Introduction to Project: the Personalized E-commerce Advertising Recommendation System

## 1 Dataset Introduction

- **Dataset**: JD.com provides an advertising click-through rate (CTR) prediction dataset.  

- **Raw Sample (`raw_sample`)**  
  A random sample of 1.14M users’ ad display/click logs over 8 days (26M records).  
  Fields:  
  1. `user_id`: anonymized user ID  
  2. `adgroup_id`: anonymized ad unit ID  
  3. `time_stamp`: timestamp  
  4. `pid`: ad placement ID  
  5. `noclk`: 1 = no click, 0 = click  
  6. `clk`: 0 = no click, 1 = click  

  Training: first 7 days (20170506–20170512).  
  Testing: day 8 (20170513).  

- **Ad Feature (`ad_feature`)**  
  Basic info of all ads in raw_sample (~800K entries).  
  Fields: adgroup_id, cate_id, campaign_id, customer_id, brand_id, price.  

- **User Profile (`user_profile`)**  
  Basic info of all users in raw_sample (~1M users).  
  Fields: userid, cms_segid, cms_group_id, gender, age_level, pvalue_level, shopping_level, occupation, new_user_class_level.  

- **User Behavior Log (`behavior_log`)**  
  Shopping behavior of all users over 22 days (700M records).  
  Fields: user, time_stamp, btag (pv, cart, fav, buy), cate_id, brand_id.  

---

## 2 Project Implementation

### 2.1 Processing Raw Tables

#### (1) Raw Sample (`raw_sample`)  
- Goal: predict clicks (`clk` vs `noclk`).  
- Feature: ad placement `pid`.  
- Techniques: one-hot encoding (Spark), timestamp to datetime conversion.

#### (2) Ad Feature (`ad_feature`)  
- Keep `price` as feature (others are high-cardinality IDs).  
- Techniques: SparkSQL `dropna`, type casting.

#### (3) User Profile (`user_profile`)  
- All fields usable as features.  
- Missing values in `pvalue_level`, `new_user_class_level`:  
  - Option 1: Fill via prediction (Random Forest).  
  - Option 2: Map to high-dimensional space (e.g., one-hot + missing).  
- Techniques: Random Forest in Spark ML, merging tables, LabeledPoint, SparseVector.

#### (4) User Behavior Log (`behavior_log`)  
- Transform raw logs into rating tables for categories and brands:  
  - user_id–cate_id–behavior → user_id–cate_id–score  
  - user_id–brand_id–behavior → user_id–brand_id–score  
- Train ALS for recommendations.  
- Techniques: pivot in Spark, ALS model.

---

### 2.2 CTR Model Training

- Features = `pid` (raw_sample) + `price` (ad_feature) + all fields (user_profile).  
- Train Logistic Regression (LR) models:  
  - CTRModel_Normal: basic feature vectors.  
  - CTRModel_AllOneHot: improved one-hot version.  
- Insight: CTR depends heavily on user interest, ad features, and context.  

---

### 2.3 Offline Data Caching

#### (1) Offline Recall Set  
- Use ALS-predicted user–cate scores.  
- Recommend top 3 categories per user, then sample ~500 items for recall.  
- Store in Redis; save model in HDFS.  

#### (2) Offline Features  
- Cache user and item features in Redis.  
- Recall set (500 items) + features → LR CTR prediction → ranking.  

---

### 2.4 Real-time Log Analysis & Recommendation

#### (1) Real-time Log Processing  
- Log format: `time, location, userID, itemID, cateID, brandID, price`.  
- Update features and recall sets in real time:  
  - Location, price → update user features in Redis.  
  - Category, brand → online recall (add ~50 items).  
- Techniques: Kafka.  

#### (2) Real-time Recommendation Task  
- Recall set = offline + online.  
- Input: recall set + CTR model + features.  
- Output: top-N recommendations (e.g., top 20).  

---

### 2.5 Technologies Used

- **Flume**: log collection  
- **Kafka**: real-time log queue  
- **HDFS**: storage  
- **Spark SQL**: offline processing  
- **Spark ML**: model training  
- **Redis**: caching  

---

## 3 Project Results

![image-20210312224748835](%E9%A1%B9%E7%9B%AE%E7%AC%94%E8%AE%B0.assets/项目效果展示.png)

