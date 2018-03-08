--入组标准为
--  1.年龄大于16周岁
--  2.ICU停留时间大于24小时
--  3.第一次入院、第一次入ICU
--  4.ICU期间未发生呼吸、心率骤停（值为0）
--  5.ICU前24小时未发生少尿（尿量>0.5ml/kg/h持续6小时）
--  6.ICU期间肌酐测量次数大于2次
--  7.入ICU时未发生终末期肾病（首次肌酐测量值<4mg/dl以及诊断编码不是5856）
DROP MATERIALIZED VIEW IF EXISTS yj_exclusion_flag_per6  CASCADE;
CREATE MATERIALIZED VIEW yj_exclusion_flag_per6 AS
WITH age_los AS (
         SELECT icu.subject_id,
            icu.hadm_id,
            icu.icustay_id,
            icu.intime,
            icu.outtime,
            (date_part('epoch'::text, (icu.outtime - icu.intime)) / (((60 * 60) * 24))::double precision) AS icu_length_of_stay,
            (date_part('epoch'::text, (icu.intime - pat.dob)) / (((((60 * 60) * 24))::numeric * 365.242))::double precision) AS age,
            rank() OVER (PARTITION BY icu.subject_id ORDER BY icu.intime) AS icustay_id_order
           FROM (mimiciii.icustays icu
             JOIN mimiciii.patients pat ON ((icu.subject_id = pat.subject_id)))
        ), ex_age AS (
         SELECT age_los.subject_id,
            age_los.hadm_id,
            age_los.icustay_id,
            age_los.intime,
            age_los.outtime,
            age_los.icu_length_of_stay,
            age_los.age,
            age_los.icustay_id_order,
                CASE
                    WHEN (age_los.icu_length_of_stay < (1)::double precision) THEN 1
                    ELSE 0
                END AS exclusion_los,
                CASE
                    WHEN (age_los.age < (16)::double precision) THEN 1
                    ELSE 0
                END AS exclusion_age,
                CASE
                    WHEN (age_los.icustay_id_order <> 1) THEN 1
                    ELSE 0
                END AS exclusion_first_stay
           FROM age_los
        ), hr_val AS (
         SELECT ea.subject_id,
            ea.hadm_id,
            ea.icustay_id,
            ea.intime,
            ea.outtime,
            ea.icu_length_of_stay,
            ea.age,
            ea.icustay_id_order,
            ea.exclusion_los,
            ea.exclusion_age,
            ea.exclusion_first_stay,
            ce.valuenum,
            ce.charttime,
                CASE
                    WHEN (ce.itemid = ANY (ARRAY[211, 220045])) THEN 'HR'::text
                    ELSE NULL::text
                END AS label
           FROM (ex_age ea
             LEFT JOIN mimiciii.chartevents ce ON (((ea.subject_id = ce.subject_id) AND (ea.icustay_id = ce.icustay_id) AND ((ce.charttime >= ea.intime) AND (ce.charttime <= ea.outtime)) AND (ce.error IS DISTINCT FROM 1))))
          WHERE (ce.itemid = ANY (ARRAY[211, 220045]))
        ), min_hr AS (
         SELECT DISTINCT ea.subject_id,
            ea.hadm_id,
            ea.icustay_id,
            min(hv.valuenum) OVER (PARTITION BY hv.subject_id, hv.icustay_id ORDER BY hv.charttime ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS min_hr
           FROM (hr_val hv
             RIGHT JOIN ex_age ea ON (((hv.subject_id = ea.subject_id) AND (hv.icustay_id = ea.icustay_id))))
        ), ex_hr AS (
         SELECT ea.subject_id,
            ea.hadm_id,
            ea.icustay_id,
            ea.intime,
            ea.outtime,
            ea.icu_length_of_stay,
            ea.age,
            ea.icustay_id_order,
            ea.exclusion_los,
            ea.exclusion_age,
            ea.exclusion_first_stay,
                CASE
                    WHEN (mh.min_hr = (0)::double precision) THEN 1
                    ELSE 0
                END AS exclusion_hr
           FROM (ex_age ea
             LEFT JOIN min_hr mh ON (((ea.subject_id = mh.subject_id) AND (ea.icustay_id = mh.icustay_id))))
        ), rr_val AS (
         SELECT ea.subject_id,
            ea.hadm_id,
            ea.icustay_id,
            ea.intime,
            ea.outtime,
            ea.icu_length_of_stay,
            ea.age,
            ea.icustay_id_order,
            ea.exclusion_los,
            ea.exclusion_age,
            ea.exclusion_first_stay,
            ce.valuenum,
            ce.charttime,
                CASE
                    WHEN (ce.itemid = ANY (ARRAY[615, 618, 220210, 224690])) THEN 'RR'::text
                    ELSE NULL::text
                END AS label
           FROM (ex_age ea
             LEFT JOIN mimiciii.chartevents ce ON (((ea.subject_id = ce.subject_id) AND (ea.icustay_id = ce.icustay_id) AND ((ce.charttime >= ea.intime) AND (ce.charttime <= ea.outtime)) AND (ce.error IS DISTINCT FROM 1))))
          WHERE (ce.itemid = ANY (ARRAY[615, 618, 220210, 224690]))
        ), min_rr AS (
         SELECT DISTINCT ea.subject_id,
            ea.hadm_id,
            ea.icustay_id,
            min(hv.valuenum) OVER (PARTITION BY hv.subject_id, hv.icustay_id ORDER BY hv.charttime ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS min_rr
           FROM (hr_val hv
             RIGHT JOIN ex_age ea ON (((hv.subject_id = ea.subject_id) AND (hv.icustay_id = ea.icustay_id))))
        ), ex_rr AS (
         SELECT eh.subject_id,
            eh.hadm_id,
            eh.icustay_id,
            eh.intime,
            eh.outtime,
            eh.icu_length_of_stay,
            eh.age,
            eh.icustay_id_order,
            eh.exclusion_los,
            eh.exclusion_age,
            eh.exclusion_first_stay,
            eh.exclusion_hr,
                CASE
                    WHEN (mr.min_rr = (0)::double precision) THEN 1
                    ELSE 0
                END AS exclusion_rr
           FROM (ex_hr eh
             LEFT JOIN min_rr mr ON (((eh.subject_id = mr.subject_id) AND (eh.icustay_id = mr.icustay_id))))
        ),uo_6_all AS
(
  SELECT er.subject_id,
            er.hadm_id,
            er.icustay_id,
            er.intime,
            er.outtime,
            ku.charttime,
            ku.urineoutput_6hr,
            ku.weight,
            ku.urineoutput_6hr/ku.weight/6 AS uo_per_hour
       FROM kdigo_uo ku
  RIGHT JOIN  ex_rr er ON ku.charttime BETWEEN  er.intime AND er.intime +  INTERVAL '18' HOUR
  AND ku.icustay_id=er.icustay_id
),uo_6_min AS
(
  SELECT DISTINCT
    u6a.subject_id,
    u6a.icustay_id,
    u6a.hadm_id,
    u6a.intime,
    u6a.outtime,
    min(uo_per_hour) OVER (PARTITION BY  u6a.subject_id,u6a.icustay_id ORDER BY u6a.charttime  ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS  uo_per_min
  FROM  uo_6_all u6a
)
  ,ex_uf AS
(
SELECT
 er.*,
  CASE
    WHEN  u6m.uo_per_min <0.5 THEN 1 ELSE  0
    END  AS  exclusion_uf
  FROM  uo_6_min u6m
  RIGHT JOIN ex_rr er
    ON  u6m.subject_id=er.subject_id AND u6m.icustay_id=er.icustay_id AND  u6m.hadm_id=er.hadm_id
), cre_value AS (
         SELECT eu.subject_id,
            eu.icustay_id,
            eu.hadm_id,
            le.valuenum AS crevalue,
            le.charttime,
            count(le.valuenum) OVER (PARTITION BY eu.subject_id, eu.icustay_id ORDER BY le.charttime ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS count_cre,
            rank() OVER (PARTITION BY eu.subject_id, eu.icustay_id ORDER BY le.charttime ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS cre_order
           FROM (mimiciii.labevents le
             RIGHT JOIN ex_uf eu ON (((le.subject_id = eu.subject_id) AND (le.hadm_id = eu.hadm_id) AND ((le.charttime >= eu.intime) AND (le.charttime <= eu.outtime)) AND (le.itemid = 50912))))
        ), ex_cre AS (
         SELECT eu2.subject_id,
            eu2.hadm_id,
            eu2.icustay_id,
            eu2.intime,
            eu2.outtime,
            eu2.icu_length_of_stay,
            eu2.age,
            eu2.icustay_id_order,
            eu2.exclusion_los,
            eu2.exclusion_age,
            eu2.exclusion_first_stay,
            eu2.exclusion_hr,
            eu2.exclusion_rr,
            eu2.exclusion_uf,
                CASE
                    WHEN ((cv.count_cre < 2) OR (cv.count_cre IS NULL)) THEN 1
                    ELSE 0
                END AS exclusion_cre_num,
                CASE
                    WHEN ((cv.crevalue IS NULL) OR (cv.crevalue > (4)::double precision)) THEN 1
                    ELSE 0
                END AS exclusion_cre_4
           FROM (ex_uf eu2
             LEFT JOIN cre_value cv ON (((eu2.subject_id = cv.subject_id) AND (eu2.hadm_id = cv.hadm_id) AND (eu2.icustay_id = cv.icustay_id))))
          WHERE (cv.cre_order = 1)
        ), esrd AS (
         SELECT di.row_id,
            di.subject_id,
            di.hadm_id,
            di.seq_num,
            di.icd9_code
           FROM mimiciii.diagnoses_icd di
          WHERE ((di.icd9_code)::text = '5856'::text)
        ), ex_icd AS (
         SELECT ec.subject_id,
            ec.hadm_id,
            ec.icustay_id,
            ec.intime,
            ec.outtime,
            ec.icu_length_of_stay,
            ec.age,
            ec.icustay_id_order,
            ec.exclusion_los,
            ec.exclusion_age,
            ec.exclusion_first_stay,
            ec.exclusion_hr,
            ec.exclusion_rr,
            ec.exclusion_uf,
            ec.exclusion_cre_num,
            ec.exclusion_cre_4,
                CASE
                    WHEN (esrd.icd9_code IS NULL) THEN 0
                    ELSE 1
                END AS exclusion_icd
           FROM (ex_cre ec
             LEFT JOIN esrd ON (((ec.subject_id = esrd.subject_id) AND (ec.hadm_id = esrd.hadm_id))))
        ), ex_hos AS (
         SELECT ei2.subject_id,
            ei2.hadm_id,
            ei2.icustay_id,
            ei2.intime,
            ei2.outtime,
            ei2.icu_length_of_stay,
            ei2.age,
            ei2.icustay_id_order,
            ei2.exclusion_los,
            ei2.exclusion_age,
            ei2.exclusion_first_stay,
            ei2.exclusion_hr,
            ei2.exclusion_rr,
            ei2.exclusion_uf,
            ei2.exclusion_cre_num,
            ei2.exclusion_cre_4,
            ei2.exclusion_icd,
                CASE
                    WHEN (id.hospstay_seq = '1'::bigint) THEN 0
                    ELSE 1
                END AS exclusion_hos
           FROM (ex_icd ei2
             LEFT JOIN mimiciii.icustay_detail id ON (((ei2.icustay_id = id.icustay_id) AND (ei2.hadm_id = id.hadm_id) AND (ei2.subject_id = id.subject_id))))
        ), exclusion AS (
         SELECT eh.subject_id,
            eh.hadm_id,
            eh.icustay_id,
            eh.intime,
            eh.outtime,
            eh.icu_length_of_stay,
            eh.age,
            eh.icustay_id_order,
            eh.exclusion_los,
            eh.exclusion_age,
            eh.exclusion_first_stay,
            eh.exclusion_hr,
            eh.exclusion_rr,
            eh.exclusion_uf,
            eh.exclusion_cre_num,
            eh.exclusion_cre_4,
            eh.exclusion_icd,
            eh.exclusion_hos,
            (((((((((eh.exclusion_los + eh.exclusion_first_stay) + eh.exclusion_age) + eh.exclusion_cre_4) + eh.exclusion_cre_num) + eh.exclusion_hr) + eh.exclusion_rr) + eh.exclusion_icd) + eh.exclusion_uf) + eh.exclusion_hos) AS exclusion
           FROM ex_hos eh
        )
 SELECT DISTINCT  exclusion.subject_id,
    exclusion.hadm_id,
    exclusion.icustay_id,
    exclusion.intime,
    exclusion.outtime,
    exclusion.icu_length_of_stay,
    exclusion.age,
    exclusion.icustay_id_order,
    exclusion.exclusion_los,
    exclusion.exclusion_age,
    exclusion.exclusion_first_stay,
    exclusion.exclusion_hr,
    exclusion.exclusion_rr,
    exclusion.exclusion_uf,
    exclusion.exclusion_cre_num,
    exclusion.exclusion_cre_4,
    exclusion.exclusion_icd,
    exclusion.exclusion_hos,
    exclusion.exclusion
   FROM exclusion;

--根据6小时平均尿量小于0.5筛选出来的全部可用样本，包括发生少尿及未发生少尿的，
-- 排除入ICU第一天未发生少尿时用24小时总尿量时，筛选出的总样本量为25789，用6小时平均尿量筛选出来的总样本为16273，同时满足上述两个标准的为15861
--此处只是同时满足上述两个标准的，共15861个样本
DROP MATERIALIZED VIEW IF EXISTS yj_cohort_all_per6 CASCADE;
CREATE MATERIALIZED VIEW yj_cohort_all_per6 AS
SELECT  DISTINCT  yp.*
  from yj_exclusion_flag_per6 yp
  INNER JOIN
    yj_exclusion_flag  yf ON yp.subject_id=yf.subject_id
    AND yp.icustay_id=yf.icustay_id
    AND yp.hadm_id=yf.hadm_id
WHERE yp.exclusion=0
and yf.exclusion=0


--sec是second缩写，表示入组标准按照6小时的平均尿量来限制时，全部入组样本打标签结果
DROP MATERIALIZED VIEW IF EXISTS yj_cohort_sec CASCADE;
CREATE MATERIALIZED VIEW yj_cohort_sec AS WITH uo_6_all AS (
         SELECT yca.subject_id,
            yca.hadm_id,
            yca.icustay_id,
            yca.intime,
            yca.outtime,
            ku.charttime,
            ku.urineoutput_6hr,
            ku.weight,
            ((ku.urineoutput_6hr / ku.weight) / (6.0)::double precision) AS uo_per_hour
           FROM (mimiciii.kdigo_uo ku
             RIGHT JOIN mimiciii.yj_cohort_all_per6  yca ON (((yca.icustay_id = ku.icustay_id) AND ((ku.charttime >= (yca.intime + '1 day'::interval day)) AND (ku.charttime <= (yca.outtime - '06:00:00'::interval hour))))))
        ), uo_6_min AS (
         SELECT u6a.subject_id,
            u6a.icustay_id,
            u6a.hadm_id,
            u6a.intime,
            u6a.outtime,
            u6a.charttime,
            u6a.urineoutput_6hr,
            u6a.weight,
            u6a.uo_per_hour,
            rank() OVER (PARTITION BY u6a.subject_id, u6a.icustay_id ORDER BY u6a.charttime) AS uo_order
           FROM uo_6_all u6a
          WHERE (u6a.uo_per_hour < (0.5)::double precision)
        ), uo_6_per AS (
         SELECT DISTINCT uo_6_min.subject_id,
            uo_6_min.icustay_id,
            uo_6_min.hadm_id,
            uo_6_min.intime,
            uo_6_min.outtime,
            uo_6_min.charttime,
            uo_6_min.urineoutput_6hr,
            uo_6_min.weight,
            uo_6_min.uo_per_hour,
            uo_6_min.uo_order
           FROM uo_6_min
          WHERE (uo_6_min.uo_order = 1)--第一次发生少尿的
        ), class_per6 AS (
         SELECT DISTINCT yca2.subject_id,
            yca2.icustay_id,
            yca2.hadm_id,
            yca2.intime,
            yca2.outtime,
            yca2.age,
            yca2.icu_length_of_stay,
           u6p.charttime,
                CASE
                    WHEN ((yca2.hadm_id = u6p.hadm_id) AND (yca2.icustay_id = u6p.icustay_id) AND (yca2.subject_id = u6p.subject_id)) THEN 1
                    ELSE 0
                END AS classlabel
           FROM (mimiciii.yj_cohort_all_per6 yca2
             LEFT JOIN uo_6_per u6p ON (((yca2.hadm_id = u6p.hadm_id) AND (yca2.icustay_id = u6p.icustay_id) AND (yca2.subject_id = u6p.subject_id))))
        )
 SELECT class_per6.subject_id,
    class_per6.icustay_id,
    class_per6.hadm_id,
    class_per6.intime,
    class_per6.outtime,
    class_per6.age,
    class_per6.icu_length_of_stay,
    class_per6.classlabel
   FROM class_per6;


--少尿的标准全部修改为6小时平均尿量后，在入组的15861个样本中，发生少尿的样本为4457个，占28.1%
SELECT  DISTINCT  count(*)
FROM yj_cohort_sec
WHERE  classlabel=1