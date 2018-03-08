# Oliguria 少尿预测


# 1.少尿样本筛选
------------------
## 全部入组患者筛选及数据提取<br>
  ### 生成的一些视图说明
     yj_exclusion_flag_per6:判断样本是否满足各个入组条件，满足为0，不满足为1<br>
     yj_cohort_all_per6:满足所有入组标准的样本<br>
     yj_cohort_sec:对全部入组样本打标签，发生少尿的标签为1，未发生少尿的标签为0<br>
     yj_eigen_sec:提取全部入组样本进入ICU后24小时的暴露变量<br>
     yj_diureticdurations:利尿剂使用情况视图<br>
  
## 少尿组数据提取
  ### 主要视图说明
     yj_uolesscohort:保存所有发生少尿的样本<br>
      yj_uo24:发生少尿前24小时的总尿量<br>
      yj_vital24:发生少尿钱24小时的关键生理参数<br>
      yj_eigen_24:发生少尿前24小时暴露变量的提取结果（没有严重程度评分的）<br>
