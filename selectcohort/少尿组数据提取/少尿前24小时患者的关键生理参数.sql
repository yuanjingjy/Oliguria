CREATE MATERIALIZED VIEW yj_vital24 AS SELECT pvt.subject_id,
    pvt.hadm_id,
    pvt.icustay_id,
    min(
        CASE
            WHEN (pvt.vitalid = 1) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS heartrate_min,
    max(
        CASE
            WHEN (pvt.vitalid = 1) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS heartrate_max,
    avg(
        CASE
            WHEN (pvt.vitalid = 1) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS heartrate_mean,
    min(
        CASE
            WHEN (pvt.vitalid = 2) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS sysbp_min,
    max(
        CASE
            WHEN (pvt.vitalid = 2) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS sysbp_max,
    avg(
        CASE
            WHEN (pvt.vitalid = 2) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS sysbp_mean,
    min(
        CASE
            WHEN (pvt.vitalid = 3) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS diasbp_min,
    max(
        CASE
            WHEN (pvt.vitalid = 3) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS diasbp_max,
    avg(
        CASE
            WHEN (pvt.vitalid = 3) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS diasbp_mean,
    min(
        CASE
            WHEN (pvt.vitalid = 4) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS meanbp_min,
    max(
        CASE
            WHEN (pvt.vitalid = 4) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS meanbp_max,
    avg(
        CASE
            WHEN (pvt.vitalid = 4) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS meanbp_mean,
    min(
        CASE
            WHEN (pvt.vitalid = 5) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS resprate_min,
    max(
        CASE
            WHEN (pvt.vitalid = 5) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS resprate_max,
    avg(
        CASE
            WHEN (pvt.vitalid = 5) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS resprate_mean,
    min(
        CASE
            WHEN (pvt.vitalid = 6) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS tempc_min,
    max(
        CASE
            WHEN (pvt.vitalid = 6) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS tempc_max,
    avg(
        CASE
            WHEN (pvt.vitalid = 6) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS tempc_mean,
    min(
        CASE
            WHEN (pvt.vitalid = 7) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS spo2_min,
    max(
        CASE
            WHEN (pvt.vitalid = 7) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS spo2_max,
    avg(
        CASE
            WHEN (pvt.vitalid = 7) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS spo2_mean,
    min(
        CASE
            WHEN (pvt.vitalid = 8) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS glucose_min,
    max(
        CASE
            WHEN (pvt.vitalid = 8) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS glucose_max,
    avg(
        CASE
            WHEN (pvt.vitalid = 8) THEN pvt.valuenum
            ELSE NULL::double precision
        END) AS glucose_mean
   FROM ( SELECT yu.subject_id,
            yu.hadm_id,
            yu.icustay_id,
                CASE
                    WHEN ((ce.itemid = ANY (ARRAY[211, 220045])) AND (ce.valuenum > (0)::double precision) AND (ce.valuenum < (300)::double precision)) THEN 1
                    WHEN ((ce.itemid = ANY (ARRAY[51, 442, 455, 6701, 220179, 220050])) AND (ce.valuenum > (0)::double precision) AND (ce.valuenum < (400)::double precision)) THEN 2
                    WHEN ((ce.itemid = ANY (ARRAY[8368, 8440, 8441, 8555, 220180, 220051])) AND (ce.valuenum > (0)::double precision) AND (ce.valuenum < (300)::double precision)) THEN 3
                    WHEN ((ce.itemid = ANY (ARRAY[456, 52, 6702, 443, 220052, 220181, 225312])) AND (ce.valuenum > (0)::double precision) AND (ce.valuenum < (300)::double precision)) THEN 4
                    WHEN ((ce.itemid = ANY (ARRAY[615, 618, 220210, 224690])) AND (ce.valuenum > (0)::double precision) AND (ce.valuenum < (70)::double precision)) THEN 5
                    WHEN ((ce.itemid = ANY (ARRAY[223761, 678])) AND (ce.valuenum > (70)::double precision) AND (ce.valuenum < (120)::double precision)) THEN 6
                    WHEN ((ce.itemid = ANY (ARRAY[223762, 676])) AND (ce.valuenum > (10)::double precision) AND (ce.valuenum < (50)::double precision)) THEN 6
                    WHEN ((ce.itemid = ANY (ARRAY[646, 220277])) AND (ce.valuenum > (0)::double precision) AND (ce.valuenum <= (100)::double precision)) THEN 7
                    WHEN ((ce.itemid = ANY (ARRAY[807, 811, 1529, 3745, 3744, 225664, 220621, 226537])) AND (ce.valuenum > (0)::double precision)) THEN 8
                    ELSE NULL::integer
                END AS vitalid,
                CASE
                    WHEN (ce.itemid = ANY (ARRAY[223761, 678])) THEN ((ce.valuenum - (32)::double precision) / (1.8)::double precision)
                    ELSE ce.valuenum
                END AS valuenum
           FROM (mimiciii.yj_uolesscohort yu
             LEFT JOIN mimiciii.chartevents ce ON (((yu.subject_id = ce.subject_id) AND (yu.hadm_id = ce.hadm_id) AND (yu.icustay_id = ce.icustay_id) AND ((ce.charttime >= yu.starttime) AND (ce.charttime <= yu.endtime)) AND (ce.error IS DISTINCT FROM 1))))
          WHERE (ce.itemid = ANY (ARRAY[211, 220045, 51, 442, 455, 6701, 220179, 220050, 8368, 8440, 8441, 8555, 220180, 220051, 456, 52, 6702, 443, 220052, 220181, 225312, 618, 615, 220210, 224690, 646, 220277, 807, 811, 1529, 3745, 3744, 225664, 220621, 226537, 223762, 676, 223761, 678]))) pvt
  GROUP BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id
  ORDER BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id;
