CREATE MATERIALIZED VIEW yj_uo24 AS SELECT ie.subject_id,
    ie.hadm_id,
    ie.icustay_id,
    sum(
        CASE
            WHEN (oe.itemid = 227488) THEN (('-1'::integer)::double precision * oe.value)
            ELSE oe.value
        END) AS urineoutput
   FROM (mimiciii.yj_uolesscohort ie
     LEFT JOIN mimiciii.outputevents oe ON (((ie.subject_id = oe.subject_id) AND (ie.hadm_id = oe.hadm_id) AND (ie.icustay_id = oe.icustay_id) AND ((oe.charttime >= ie.starttime) AND (oe.charttime <= ie.endtime)))))
  WHERE (oe.itemid = ANY (ARRAY[40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057, 40056, 40405, 40428, 40086, 40096, 40651, 226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558, 227488, 227489]))
  GROUP BY ie.subject_id, ie.hadm_id, ie.icustay_id
  ORDER BY ie.subject_id, ie.hadm_id, ie.icustay_id;
