CREATE MATERIALIZED VIEW yj_eigen_24 AS WITH pvt AS (
         SELECT yu.subject_id,
            yu.hadm_id,
            yu.icustay_id,
            yu.starttime,
            yu.endtime,
            yu.classlabel,
            yu.age,
                CASE
                    WHEN (le.itemid = 50821) THEN 'pO2'::text
                    WHEN (le.itemid = 50818) THEN 'pCO2'::text
                    WHEN (le.itemid = 50820) THEN 'pH'::text
                    WHEN (le.itemid = 50822) THEN 'POTASSIUM'::text
                    WHEN (le.itemid = 50808) THEN 'CALCIUM'::text
                    WHEN (le.itemid = 50824) THEN 'SODIUM'::text
                    WHEN (le.itemid = 50806) THEN 'CHLORIDE'::text
                    WHEN (le.itemid = 50809) THEN 'GLUCOSE'::text
                    WHEN (le.itemid = 50803) THEN 'BICARBONATE'::text
                    WHEN (le.itemid = 50804) THEN 'BICARBONATE'::text
                    WHEN (le.itemid = 50802) THEN 'BE'::text
                    WHEN (le.itemid = 51300) THEN 'WBC'::text
                    WHEN (le.itemid = 51301) THEN 'WBC'::text
                    WHEN (le.itemid = 51279) THEN 'RBC'::text
                    WHEN (le.itemid = 51265) THEN 'PLATELET'::text
                    WHEN (le.itemid = 50889) THEN 'CRP'::text
                    WHEN (le.itemid = 50813) THEN 'LACTATE'::text
                    WHEN (le.itemid = 50861) THEN 'ALT'::text
                    WHEN (le.itemid = 50878) THEN 'AST'::text
                    WHEN (le.itemid = 50912) THEN 'CREATININE'::text
                    WHEN (le.itemid = 51006) THEN 'BUN'::text
                    WHEN (le.itemid = 50867) THEN 'AMYLASE'::text
                    WHEN (le.itemid = 50956) THEN 'LIPASE'::text
                    WHEN (le.itemid = 51274) THEN 'PT'::text
                    WHEN (le.itemid = 51237) THEN 'INR'::text
                    WHEN (le.itemid = 51196) THEN 'D-Dimer'::text
                    WHEN (le.itemid = 51214) THEN 'FIB'::text
                    WHEN (le.itemid = 51275) THEN 'PTT'::text
                    WHEN (le.itemid = 51498) THEN 'SG'::text
                    WHEN (le.itemid = 51084) THEN 'GLU-UC'::text
                    WHEN (le.itemid = 51466) THEN 'Blood-UR'::text
                    WHEN (le.itemid = 51094) THEN 'PH-UC'::text
                    WHEN (le.itemid = 51491) THEN 'PH-UH'::text
                    WHEN (le.itemid = 51102) THEN 'PRO-UC'::text
                    WHEN (le.itemid = 51492) THEN 'PRO-UH'::text
                    WHEN (le.itemid = 51487) THEN 'NIT-UH'::text
                    WHEN (le.itemid = 51484) THEN 'KET-UH'::text
                    WHEN (le.itemid = 51486) THEN 'LEU-UH'::text
                    WHEN (le.itemid = 51514) THEN 'UBG-UH'::text
                    ELSE NULL::text
                END AS label,
                CASE
                    WHEN ((le.itemid = 50803) AND (le.valuenum > (10000)::double precision)) THEN NULL::double precision
                    WHEN ((le.itemid = 50804) AND (le.valuenum > (10000)::double precision)) THEN NULL::double precision
                    WHEN ((le.itemid = 50806) AND (le.valuenum > (10000)::double precision)) THEN NULL::double precision
                    WHEN ((le.itemid = 50912) AND (le.valuenum > (150)::double precision)) THEN NULL::double precision
                    WHEN ((le.itemid = 50809) AND (le.valuenum > (10000)::double precision)) THEN NULL::double precision
                    WHEN ((le.itemid = 50813) AND (le.valuenum > (50)::double precision)) THEN NULL::double precision
                    WHEN ((le.itemid = 51265) AND (le.valuenum > (10000)::double precision)) THEN NULL::double precision
                    WHEN ((le.itemid = 50822) AND (le.valuenum > (30)::double precision)) THEN NULL::double precision
                    WHEN ((le.itemid = 51275) AND (le.valuenum > (150)::double precision)) THEN NULL::double precision
                    WHEN ((le.itemid = 51237) AND (le.valuenum > (50)::double precision)) THEN NULL::double precision
                    WHEN ((le.itemid = 51274) AND (le.valuenum > (150)::double precision)) THEN NULL::double precision
                    WHEN ((le.itemid = 50824) AND (le.valuenum > (200)::double precision)) THEN NULL::double precision
                    WHEN ((le.itemid = 51006) AND (le.valuenum > (300)::double precision)) THEN NULL::double precision
                    WHEN ((le.itemid = 51300) AND (le.valuenum > (1000)::double precision)) THEN NULL::double precision
                    WHEN ((le.itemid = 51301) AND (le.valuenum > (1000)::double precision)) THEN NULL::double precision
                    ELSE le.valuenum
                END AS valuenum
           FROM (mimiciii.yj_uolesscohort yu
             LEFT JOIN mimiciii.labevents le ON (((yu.subject_id = le.subject_id) AND (yu.hadm_id = le.hadm_id) AND ((le.charttime >= yu.starttime) AND (le.charttime <= yu.endtime)) AND (le.itemid = ANY (ARRAY[50821, 50818, 50820, 50822, 50808, 50824, 50806, 50809, 50803, 50804, 50802, 51300, 51301, 51279, 51265, 50889, 50813, 50861, 50878, 50912, 51006, 50867, 50956, 51274, 51237, 51196, 51214, 51275, 51498, 51084, 51466, 51094, 51491, 51102, 51492, 51487, 51484, 51486, 51514])) AND (le.valuenum IS NOT NULL) AND (le.valuenum > (0)::double precision))))
        ), labval AS (
         SELECT pvt.subject_id,
            pvt.hadm_id,
            pvt.icustay_id,
            pvt.starttime,
            pvt.age,
            pvt.classlabel,
            min(
                CASE
                    WHEN (pvt.label = 'pO2'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS po2_min,
            max(
                CASE
                    WHEN (pvt.label = 'pO2'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS po2_max,
            avg(
                CASE
                    WHEN (pvt.label = 'pO2'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS po2_avg,
            min(
                CASE
                    WHEN (pvt.label = 'pCO2'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS pco2_min,
            max(
                CASE
                    WHEN (pvt.label = 'pCO2'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS pco2_max,
            avg(
                CASE
                    WHEN (pvt.label = 'pCO2'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS pco2_avg,
            min(
                CASE
                    WHEN (pvt.label = 'pH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS ph_min,
            max(
                CASE
                    WHEN (pvt.label = 'pH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS ph_max,
            avg(
                CASE
                    WHEN (pvt.label = 'pH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS ph_avg,
            min(
                CASE
                    WHEN (pvt.label = 'POTASSIUM'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS potassium_min,
            max(
                CASE
                    WHEN (pvt.label = 'POTASSIUM'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS potassium_max,
            avg(
                CASE
                    WHEN (pvt.label = 'POTASSIUM'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS potassium_avg,
            min(
                CASE
                    WHEN (pvt.label = 'CALCIUM'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS calcium_min,
            max(
                CASE
                    WHEN (pvt.label = 'CALCIUM'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS calcium_max,
            avg(
                CASE
                    WHEN (pvt.label = 'CALCIUM'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS calcium_avg,
            min(
                CASE
                    WHEN (pvt.label = 'SODIUM'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS sodium_min,
            max(
                CASE
                    WHEN (pvt.label = 'SODIUM'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS sodium_max,
            avg(
                CASE
                    WHEN (pvt.label = 'SODIUM'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS sodium_avg,
            min(
                CASE
                    WHEN (pvt.label = 'CHLORIDE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS chloride_min,
            max(
                CASE
                    WHEN (pvt.label = 'CHLORIDE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS chloride_max,
            avg(
                CASE
                    WHEN (pvt.label = 'CHLORIDE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS chloride_avg,
            min(
                CASE
                    WHEN (pvt.label = 'GLUCOSE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS glucose_min,
            max(
                CASE
                    WHEN (pvt.label = 'GLUCOSE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS glucose_max,
            avg(
                CASE
                    WHEN (pvt.label = 'GLUCOSE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS glucose_avg,
            min(
                CASE
                    WHEN (pvt.label = 'BICARBONATE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS bicarbonate_min,
            max(
                CASE
                    WHEN (pvt.label = 'BICARBONATE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS bicarbonate_max,
            avg(
                CASE
                    WHEN (pvt.label = 'BICARBONATE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS bicarbonate_avg,
            min(
                CASE
                    WHEN (pvt.label = 'BE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS be_min,
            max(
                CASE
                    WHEN (pvt.label = 'BE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS be_max,
            avg(
                CASE
                    WHEN (pvt.label = 'BE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS be_avg,
            min(
                CASE
                    WHEN (pvt.label = 'WBC'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS wbc_min,
            max(
                CASE
                    WHEN (pvt.label = 'WBC'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS wbc_max,
            avg(
                CASE
                    WHEN (pvt.label = 'WBC'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS wbc_avg,
            min(
                CASE
                    WHEN (pvt.label = 'RBC'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS rbc_min,
            max(
                CASE
                    WHEN (pvt.label = 'RBC'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS rbc_max,
            avg(
                CASE
                    WHEN (pvt.label = 'RBC'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS rbc_avg,
            min(
                CASE
                    WHEN (pvt.label = 'PLATELET'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS platelet_min,
            max(
                CASE
                    WHEN (pvt.label = 'PLATELET'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS platelet_max,
            avg(
                CASE
                    WHEN (pvt.label = 'PLATELET'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS platelet_avg,
            min(
                CASE
                    WHEN (pvt.label = 'CRP'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS crp_min,
            max(
                CASE
                    WHEN (pvt.label = 'CRP'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS crp_max,
            avg(
                CASE
                    WHEN (pvt.label = 'CRP'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS crp_avg,
            min(
                CASE
                    WHEN (pvt.label = 'LACTATE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS lactate_min,
            max(
                CASE
                    WHEN (pvt.label = 'LACTATE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS lactate_max,
            avg(
                CASE
                    WHEN (pvt.label = 'LACTATE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS lactate_avg,
            min(
                CASE
                    WHEN (pvt.label = 'ALT'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS alt_min,
            max(
                CASE
                    WHEN (pvt.label = 'ALT'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS alt_max,
            avg(
                CASE
                    WHEN (pvt.label = 'ALT'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS alt_avg,
            min(
                CASE
                    WHEN (pvt.label = 'AST'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS ast_min,
            max(
                CASE
                    WHEN (pvt.label = 'AST'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS ast_max,
            avg(
                CASE
                    WHEN (pvt.label = 'AST'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS ast_avg,
            min(
                CASE
                    WHEN (pvt.label = 'CREATININE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS creatinine_min,
            max(
                CASE
                    WHEN (pvt.label = 'CREATININE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS creatinine_max,
            avg(
                CASE
                    WHEN (pvt.label = 'CREATININE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS creatinine_avg,
            min(
                CASE
                    WHEN (pvt.label = 'BUN'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS bun_min,
            max(
                CASE
                    WHEN (pvt.label = 'BUN'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS bun_max,
            avg(
                CASE
                    WHEN (pvt.label = 'BUN'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS bun_avg,
            min(
                CASE
                    WHEN (pvt.label = 'AMYLASE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS amylase_min,
            max(
                CASE
                    WHEN (pvt.label = 'AMYLASE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS amylase_max,
            avg(
                CASE
                    WHEN (pvt.label = 'AMYLASE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS amylase_avg,
            min(
                CASE
                    WHEN (pvt.label = 'LIPASE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS lipase_min,
            max(
                CASE
                    WHEN (pvt.label = 'LIPASE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS lipase_max,
            avg(
                CASE
                    WHEN (pvt.label = 'LIPASE'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS lipase_avg,
            min(
                CASE
                    WHEN (pvt.label = 'PT'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS pt_min,
            max(
                CASE
                    WHEN (pvt.label = 'PT'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS pt_max,
            avg(
                CASE
                    WHEN (pvt.label = 'PT'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS pt_avg,
            min(
                CASE
                    WHEN (pvt.label = 'INR'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS inr_min,
            max(
                CASE
                    WHEN (pvt.label = 'INR'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS inr_max,
            avg(
                CASE
                    WHEN (pvt.label = 'INR'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS inr_avg,
            min(
                CASE
                    WHEN (pvt.label = 'D-Dimer'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS ddimer_min,
            max(
                CASE
                    WHEN (pvt.label = 'D-Dimer'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS ddimer_max,
            avg(
                CASE
                    WHEN (pvt.label = 'D-Dimer'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS ddimer_avg,
            min(
                CASE
                    WHEN (pvt.label = 'FIB'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS fib_min,
            max(
                CASE
                    WHEN (pvt.label = 'FIB'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS fib_max,
            avg(
                CASE
                    WHEN (pvt.label = 'FIB'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS fib_avg,
            min(
                CASE
                    WHEN (pvt.label = 'PTT'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS ptt_min,
            max(
                CASE
                    WHEN (pvt.label = 'PTT'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS ptt_max,
            avg(
                CASE
                    WHEN (pvt.label = 'PTT'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS ptt_avg,
            min(
                CASE
                    WHEN (pvt.label = 'SG'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS sg_min,
            max(
                CASE
                    WHEN (pvt.label = 'SG'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS sg_max,
            avg(
                CASE
                    WHEN (pvt.label = 'SG'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS sg_avg,
            min(
                CASE
                    WHEN (pvt.label = 'GLU-UC'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS gluuc_min,
            max(
                CASE
                    WHEN (pvt.label = 'GLU-UC'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS gluuc_max,
            avg(
                CASE
                    WHEN (pvt.label = 'GLU-UC'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS gluuc_avg,
            min(
                CASE
                    WHEN (pvt.label = 'Blood-UR'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS bloodu_min,
            max(
                CASE
                    WHEN (pvt.label = 'Blood-UR'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS bloodu_max,
            avg(
                CASE
                    WHEN (pvt.label = 'Blood-UR'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS bloodu_avg,
            min(
                CASE
                    WHEN (pvt.label = 'PH-UC'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS phuc_min,
            max(
                CASE
                    WHEN (pvt.label = 'PH-UC'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS phuc_max,
            avg(
                CASE
                    WHEN (pvt.label = 'PH-UC'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS phuc_avg,
            min(
                CASE
                    WHEN (pvt.label = 'PH-UH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS phuh_min,
            max(
                CASE
                    WHEN (pvt.label = 'PH-UH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS phuh_max,
            avg(
                CASE
                    WHEN (pvt.label = 'PH-UH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS phuh_avg,
            min(
                CASE
                    WHEN (pvt.label = 'PRO-UC'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS prouc_min,
            max(
                CASE
                    WHEN (pvt.label = 'PRO-UC'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS prouc_max,
            avg(
                CASE
                    WHEN (pvt.label = 'PRO-UC'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS prouc_avg,
            min(
                CASE
                    WHEN (pvt.label = 'PRO-UH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS prouh_min,
            max(
                CASE
                    WHEN (pvt.label = 'PRO-UH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS prouh_max,
            avg(
                CASE
                    WHEN (pvt.label = 'PRO-UH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS prouh_avg,
            min(
                CASE
                    WHEN (pvt.label = 'NIT-UH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS nit_min,
            max(
                CASE
                    WHEN (pvt.label = 'NIT-UH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS nit_max,
            avg(
                CASE
                    WHEN (pvt.label = 'NIT-UH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS nit_avg,
            min(
                CASE
                    WHEN (pvt.label = 'KET-UH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS ket_min,
            max(
                CASE
                    WHEN (pvt.label = 'KET-UH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS ket_max,
            avg(
                CASE
                    WHEN (pvt.label = 'KET-UH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS ket_avg,
            min(
                CASE
                    WHEN (pvt.label = 'LEU-UH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS leu_min,
            max(
                CASE
                    WHEN (pvt.label = 'LEU-UH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS leu_max,
            avg(
                CASE
                    WHEN (pvt.label = 'LEU-UH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS leu_avg,
            min(
                CASE
                    WHEN (pvt.label = 'UBG-UH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS ubg_min,
            max(
                CASE
                    WHEN (pvt.label = 'UBG-UH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS ubg_max,
            avg(
                CASE
                    WHEN (pvt.label = 'UBG-UH'::text) THEN pvt.valuenum
                    ELSE NULL::double precision
                END) AS ubg_avg
           FROM pvt
          GROUP BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.starttime, pvt.endtime, pvt.age, pvt.classlabel
        ), addvital AS (
         SELECT lab.subject_id,
            lab.hadm_id,
            lab.icustay_id,
            lab.starttime,
            lab.age,
            lab.classlabel,
            lab.po2_min,
            lab.po2_max,
            lab.po2_avg,
            lab.pco2_min,
            lab.pco2_max,
            lab.pco2_avg,
            lab.ph_min,
            lab.ph_max,
            lab.ph_avg,
            lab.potassium_min,
            lab.potassium_max,
            lab.potassium_avg,
            lab.calcium_min,
            lab.calcium_max,
            lab.calcium_avg,
            lab.sodium_min,
            lab.sodium_max,
            lab.sodium_avg,
            lab.chloride_min,
            lab.chloride_max,
            lab.chloride_avg,
            lab.glucose_min,
            lab.glucose_max,
            lab.glucose_avg,
            lab.bicarbonate_min,
            lab.bicarbonate_max,
            lab.bicarbonate_avg,
            lab.be_min,
            lab.be_max,
            lab.be_avg,
            lab.wbc_min,
            lab.wbc_max,
            lab.wbc_avg,
            lab.rbc_min,
            lab.rbc_max,
            lab.rbc_avg,
            lab.platelet_min,
            lab.platelet_max,
            lab.platelet_avg,
            lab.crp_min,
            lab.crp_max,
            lab.crp_avg,
            lab.lactate_min,
            lab.lactate_max,
            lab.lactate_avg,
            lab.alt_min,
            lab.alt_max,
            lab.alt_avg,
            lab.ast_min,
            lab.ast_max,
            lab.ast_avg,
            lab.creatinine_min,
            lab.creatinine_max,
            lab.creatinine_avg,
            lab.bun_min,
            lab.bun_max,
            lab.bun_avg,
            lab.amylase_min,
            lab.amylase_max,
            lab.amylase_avg,
            lab.lipase_min,
            lab.lipase_max,
            lab.lipase_avg,
            lab.pt_min,
            lab.pt_max,
            lab.pt_avg,
            lab.inr_min,
            lab.inr_max,
            lab.inr_avg,
            lab.ddimer_min,
            lab.ddimer_max,
            lab.ddimer_avg,
            lab.fib_min,
            lab.fib_max,
            lab.fib_avg,
            lab.ptt_min,
            lab.ptt_max,
            lab.ptt_avg,
            lab.sg_min,
            lab.sg_max,
            lab.sg_avg,
            lab.gluuc_min,
            lab.gluuc_max,
            lab.gluuc_avg,
            lab.bloodu_min,
            lab.bloodu_max,
            lab.bloodu_avg,
            lab.phuc_min,
            lab.phuc_max,
            lab.phuc_avg,
            lab.phuh_min,
            lab.phuh_max,
            lab.phuh_avg,
            lab.prouc_min,
            lab.prouc_max,
            lab.prouc_avg,
            lab.prouh_min,
            lab.prouh_max,
            lab.prouh_avg,
            lab.nit_min,
            lab.nit_max,
            lab.nit_avg,
            lab.ket_min,
            lab.ket_max,
            lab.ket_avg,
            lab.leu_min,
            lab.leu_max,
            lab.leu_avg,
            lab.ubg_min,
            lab.ubg_max,
            lab.ubg_avg,
            vf.heartrate_max,
            vf.heartrate_mean,
            vf.heartrate_min,
            vf.sysbp_max,
            vf.sysbp_mean,
            vf.sysbp_min,
            vf.diasbp_max,
            vf.diasbp_mean,
            vf.diasbp_min,
            vf.meanbp_max,
            vf.meanbp_mean,
            vf.meanbp_min,
            vf.resprate_max,
            vf.resprate_mean,
            vf.resprate_min,
            vf.spo2_max,
            vf.spo2_mean,
            vf.spo2_min
           FROM (labval lab
             LEFT JOIN mimiciii.yj_vital24 vf ON (((vf.subject_id = lab.subject_id) AND (vf.hadm_id = lab.hadm_id) AND (vf.icustay_id = lab.icustay_id))))
        ), addheightweight AS (
         SELECT addvital.subject_id,
            addvital.hadm_id,
            addvital.icustay_id,
            addvital.starttime,
            addvital.age,
            addvital.classlabel,
            addvital.po2_min,
            addvital.po2_max,
            addvital.po2_avg,
            addvital.pco2_min,
            addvital.pco2_max,
            addvital.pco2_avg,
            addvital.ph_min,
            addvital.ph_max,
            addvital.ph_avg,
            addvital.potassium_min,
            addvital.potassium_max,
            addvital.potassium_avg,
            addvital.calcium_min,
            addvital.calcium_max,
            addvital.calcium_avg,
            addvital.sodium_min,
            addvital.sodium_max,
            addvital.sodium_avg,
            addvital.chloride_min,
            addvital.chloride_max,
            addvital.chloride_avg,
            addvital.glucose_min,
            addvital.glucose_max,
            addvital.glucose_avg,
            addvital.bicarbonate_min,
            addvital.bicarbonate_max,
            addvital.bicarbonate_avg,
            addvital.be_min,
            addvital.be_max,
            addvital.be_avg,
            addvital.wbc_min,
            addvital.wbc_max,
            addvital.wbc_avg,
            addvital.rbc_min,
            addvital.rbc_max,
            addvital.rbc_avg,
            addvital.platelet_min,
            addvital.platelet_max,
            addvital.platelet_avg,
            addvital.crp_min,
            addvital.crp_max,
            addvital.crp_avg,
            addvital.lactate_min,
            addvital.lactate_max,
            addvital.lactate_avg,
            addvital.alt_min,
            addvital.alt_max,
            addvital.alt_avg,
            addvital.ast_min,
            addvital.ast_max,
            addvital.ast_avg,
            addvital.creatinine_min,
            addvital.creatinine_max,
            addvital.creatinine_avg,
            addvital.bun_min,
            addvital.bun_max,
            addvital.bun_avg,
            addvital.amylase_min,
            addvital.amylase_max,
            addvital.amylase_avg,
            addvital.lipase_min,
            addvital.lipase_max,
            addvital.lipase_avg,
            addvital.pt_min,
            addvital.pt_max,
            addvital.pt_avg,
            addvital.inr_min,
            addvital.inr_max,
            addvital.inr_avg,
            addvital.ddimer_min,
            addvital.ddimer_max,
            addvital.ddimer_avg,
            addvital.fib_min,
            addvital.fib_max,
            addvital.fib_avg,
            addvital.ptt_min,
            addvital.ptt_max,
            addvital.ptt_avg,
            addvital.sg_min,
            addvital.sg_max,
            addvital.sg_avg,
            addvital.gluuc_min,
            addvital.gluuc_max,
            addvital.gluuc_avg,
            addvital.bloodu_min,
            addvital.bloodu_max,
            addvital.bloodu_avg,
            addvital.phuc_min,
            addvital.phuc_max,
            addvital.phuc_avg,
            addvital.phuh_min,
            addvital.phuh_max,
            addvital.phuh_avg,
            addvital.prouc_min,
            addvital.prouc_max,
            addvital.prouc_avg,
            addvital.prouh_min,
            addvital.prouh_max,
            addvital.prouh_avg,
            addvital.nit_min,
            addvital.nit_max,
            addvital.nit_avg,
            addvital.ket_min,
            addvital.ket_max,
            addvital.ket_avg,
            addvital.leu_min,
            addvital.leu_max,
            addvital.leu_avg,
            addvital.ubg_min,
            addvital.ubg_max,
            addvital.ubg_avg,
            addvital.heartrate_max,
            addvital.heartrate_mean,
            addvital.heartrate_min,
            addvital.sysbp_max,
            addvital.sysbp_mean,
            addvital.sysbp_min,
            addvital.diasbp_max,
            addvital.diasbp_mean,
            addvital.diasbp_min,
            addvital.meanbp_max,
            addvital.meanbp_mean,
            addvital.meanbp_min,
            addvital.resprate_max,
            addvital.resprate_mean,
            addvital.resprate_min,
            addvital.spo2_max,
            addvital.spo2_mean,
            addvital.spo2_min,
            hw.height_first,
            hw.height_max,
            hw.height_min,
            hw.weight_first,
            hw.weight_min,
            hw.weight_max
           FROM (addvital
             LEFT JOIN mimiciii.heightweight hw ON (((addvital.subject_id = hw.subject_id) AND (addvital.icustay_id = hw.icustay_id))))
        ), adduo AS (
         SELECT ahw.subject_id,
            ahw.hadm_id,
            ahw.icustay_id,
            ahw.starttime,
            ahw.age,
            ahw.classlabel,
            ahw.po2_min,
            ahw.po2_max,
            ahw.po2_avg,
            ahw.pco2_min,
            ahw.pco2_max,
            ahw.pco2_avg,
            ahw.ph_min,
            ahw.ph_max,
            ahw.ph_avg,
            ahw.potassium_min,
            ahw.potassium_max,
            ahw.potassium_avg,
            ahw.calcium_min,
            ahw.calcium_max,
            ahw.calcium_avg,
            ahw.sodium_min,
            ahw.sodium_max,
            ahw.sodium_avg,
            ahw.chloride_min,
            ahw.chloride_max,
            ahw.chloride_avg,
            ahw.glucose_min,
            ahw.glucose_max,
            ahw.glucose_avg,
            ahw.bicarbonate_min,
            ahw.bicarbonate_max,
            ahw.bicarbonate_avg,
            ahw.be_min,
            ahw.be_max,
            ahw.be_avg,
            ahw.wbc_min,
            ahw.wbc_max,
            ahw.wbc_avg,
            ahw.rbc_min,
            ahw.rbc_max,
            ahw.rbc_avg,
            ahw.platelet_min,
            ahw.platelet_max,
            ahw.platelet_avg,
            ahw.crp_min,
            ahw.crp_max,
            ahw.crp_avg,
            ahw.lactate_min,
            ahw.lactate_max,
            ahw.lactate_avg,
            ahw.alt_min,
            ahw.alt_max,
            ahw.alt_avg,
            ahw.ast_min,
            ahw.ast_max,
            ahw.ast_avg,
            ahw.creatinine_min,
            ahw.creatinine_max,
            ahw.creatinine_avg,
            ahw.bun_min,
            ahw.bun_max,
            ahw.bun_avg,
            ahw.amylase_min,
            ahw.amylase_max,
            ahw.amylase_avg,
            ahw.lipase_min,
            ahw.lipase_max,
            ahw.lipase_avg,
            ahw.pt_min,
            ahw.pt_max,
            ahw.pt_avg,
            ahw.inr_min,
            ahw.inr_max,
            ahw.inr_avg,
            ahw.ddimer_min,
            ahw.ddimer_max,
            ahw.ddimer_avg,
            ahw.fib_min,
            ahw.fib_max,
            ahw.fib_avg,
            ahw.ptt_min,
            ahw.ptt_max,
            ahw.ptt_avg,
            ahw.sg_min,
            ahw.sg_max,
            ahw.sg_avg,
            ahw.gluuc_min,
            ahw.gluuc_max,
            ahw.gluuc_avg,
            ahw.bloodu_min,
            ahw.bloodu_max,
            ahw.bloodu_avg,
            ahw.phuc_min,
            ahw.phuc_max,
            ahw.phuc_avg,
            ahw.phuh_min,
            ahw.phuh_max,
            ahw.phuh_avg,
            ahw.prouc_min,
            ahw.prouc_max,
            ahw.prouc_avg,
            ahw.prouh_min,
            ahw.prouh_max,
            ahw.prouh_avg,
            ahw.nit_min,
            ahw.nit_max,
            ahw.nit_avg,
            ahw.ket_min,
            ahw.ket_max,
            ahw.ket_avg,
            ahw.leu_min,
            ahw.leu_max,
            ahw.leu_avg,
            ahw.ubg_min,
            ahw.ubg_max,
            ahw.ubg_avg,
            ahw.heartrate_max,
            ahw.heartrate_mean,
            ahw.heartrate_min,
            ahw.sysbp_max,
            ahw.sysbp_mean,
            ahw.sysbp_min,
            ahw.diasbp_max,
            ahw.diasbp_mean,
            ahw.diasbp_min,
            ahw.meanbp_max,
            ahw.meanbp_mean,
            ahw.meanbp_min,
            ahw.resprate_max,
            ahw.resprate_mean,
            ahw.resprate_min,
            ahw.spo2_max,
            ahw.spo2_mean,
            ahw.spo2_min,
            ahw.height_first,
            ahw.height_max,
            ahw.height_min,
            ahw.weight_first,
            ahw.weight_min,
            ahw.weight_max
           FROM (addheightweight ahw
             LEFT JOIN mimiciii.yj_uo24 yu24 ON (((yu24.icustay_id = ahw.icustay_id) AND (yu24.subject_id = ahw.subject_id) AND (yu24.hadm_id = ahw.hadm_id))))
        )
 SELECT DISTINCT adduo.subject_id,
    adduo.hadm_id,
    adduo.icustay_id,
    adduo.starttime,
    adduo.age,
    adduo.classlabel,
    adduo.po2_min,
    adduo.po2_max,
    adduo.po2_avg,
    adduo.pco2_min,
    adduo.pco2_max,
    adduo.pco2_avg,
    adduo.ph_min,
    adduo.ph_max,
    adduo.ph_avg,
    adduo.potassium_min,
    adduo.potassium_max,
    adduo.potassium_avg,
    adduo.calcium_min,
    adduo.calcium_max,
    adduo.calcium_avg,
    adduo.sodium_min,
    adduo.sodium_max,
    adduo.sodium_avg,
    adduo.chloride_min,
    adduo.chloride_max,
    adduo.chloride_avg,
    adduo.glucose_min,
    adduo.glucose_max,
    adduo.glucose_avg,
    adduo.bicarbonate_min,
    adduo.bicarbonate_max,
    adduo.bicarbonate_avg,
    adduo.be_min,
    adduo.be_max,
    adduo.be_avg,
    adduo.wbc_min,
    adduo.wbc_max,
    adduo.wbc_avg,
    adduo.rbc_min,
    adduo.rbc_max,
    adduo.rbc_avg,
    adduo.platelet_min,
    adduo.platelet_max,
    adduo.platelet_avg,
    adduo.crp_min,
    adduo.crp_max,
    adduo.crp_avg,
    adduo.lactate_min,
    adduo.lactate_max,
    adduo.lactate_avg,
    adduo.alt_min,
    adduo.alt_max,
    adduo.alt_avg,
    adduo.ast_min,
    adduo.ast_max,
    adduo.ast_avg,
    adduo.creatinine_min,
    adduo.creatinine_max,
    adduo.creatinine_avg,
    adduo.bun_min,
    adduo.bun_max,
    adduo.bun_avg,
    adduo.amylase_min,
    adduo.amylase_max,
    adduo.amylase_avg,
    adduo.lipase_min,
    adduo.lipase_max,
    adduo.lipase_avg,
    adduo.pt_min,
    adduo.pt_max,
    adduo.pt_avg,
    adduo.inr_min,
    adduo.inr_max,
    adduo.inr_avg,
    adduo.ddimer_min,
    adduo.ddimer_max,
    adduo.ddimer_avg,
    adduo.fib_min,
    adduo.fib_max,
    adduo.fib_avg,
    adduo.ptt_min,
    adduo.ptt_max,
    adduo.ptt_avg,
    adduo.sg_min,
    adduo.sg_max,
    adduo.sg_avg,
    adduo.gluuc_min,
    adduo.gluuc_max,
    adduo.gluuc_avg,
    adduo.bloodu_min,
    adduo.bloodu_max,
    adduo.bloodu_avg,
    adduo.phuc_min,
    adduo.phuc_max,
    adduo.phuc_avg,
    adduo.phuh_min,
    adduo.phuh_max,
    adduo.phuh_avg,
    adduo.prouc_min,
    adduo.prouc_max,
    adduo.prouc_avg,
    adduo.prouh_min,
    adduo.prouh_max,
    adduo.prouh_avg,
    adduo.nit_min,
    adduo.nit_max,
    adduo.nit_avg,
    adduo.ket_min,
    adduo.ket_max,
    adduo.ket_avg,
    adduo.leu_min,
    adduo.leu_max,
    adduo.leu_avg,
    adduo.ubg_min,
    adduo.ubg_max,
    adduo.ubg_avg,
    adduo.heartrate_max,
    adduo.heartrate_mean,
    adduo.heartrate_min,
    adduo.sysbp_max,
    adduo.sysbp_mean,
    adduo.sysbp_min,
    adduo.diasbp_max,
    adduo.diasbp_mean,
    adduo.diasbp_min,
    adduo.meanbp_max,
    adduo.meanbp_mean,
    adduo.meanbp_min,
    adduo.resprate_max,
    adduo.resprate_mean,
    adduo.resprate_min,
    adduo.spo2_max,
    adduo.spo2_mean,
    adduo.spo2_min,
    adduo.height_first,
    adduo.height_max,
    adduo.height_min,
    adduo.weight_first,
    adduo.weight_min,
    adduo.weight_max
   FROM adduo;
