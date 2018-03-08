CREATE MATERIALIZED VIEW yj_uolesscohort AS SELECT ycs.hadm_id,
    ycs.icustay_id,
    ycs.subject_id,
    ycs.age,
    ycs.charttime AS endtime,
    (ycs.charttime - '24:00:00'::interval hour) AS starttime,
    ycs.classlabel
   FROM mimiciii.yj_cohort_sec ycs
  WHERE (ycs.classlabel = 1);
