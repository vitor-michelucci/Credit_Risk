# Credit_Risk
Regressão logística binária para classificação de risco de crédito.

1. Variáveis do dataset:
RangeIndex: 28501 entries, 0 to 28500
Data columns (total 12 columns):
 #   Column                      Non-Null Count  Dtype  
---  ------                      --------------  -----  
 0   person_age                  28501 non-null  int64  
 1   person_income               28501 non-null  int64  
 2   person_home_ownership       28501 non-null  object 
 3   person_emp_length           28501 non-null  float64
 4   loan_intent                 28501 non-null  object 
 5   loan_grade                  28501 non-null  object 
 6   loan_amnt                   28501 non-null  int64  
 7   loan_int_rate               28501 non-null  float64
 8   loan_percent_income         28501 non-null  float64
 9   cb_person_default_on_file   28501 non-null  object 
 10  cb_person_cred_hist_length  28501 non-null  int64  
 11  default                     28501 non-null  int64  
dtypes: float64(3), int64(5), object(4)

2. Proporção da variável target (default):

1 = default
0 = não default

   Proporção  Contagem
0   0.782885     22313
1   0.217115      6188

3. Dataset após get dummies:

RangeIndex: 28501 entries, 0 to 28500
Data columns (total 23 columns):
 #   Column                       Non-Null Count  Dtype  
---  ------                       --------------  -----  
 0   person_age                   28501 non-null  int64  
 1   person_income                28501 non-null  int64  
 2   person_emp_length            28501 non-null  float64
 3   loan_amnt                    28501 non-null  int64  
 4   loan_int_rate                28501 non-null  float64
 5   loan_percent_income          28501 non-null  float64
 6   cb_person_cred_hist_length   28501 non-null  int64  
 7   default                      28501 non-null  int64  
 8   person_home_ownership_OTHER  28501 non-null  uint8  
 9   person_home_ownership_OWN    28501 non-null  uint8  
 10  person_home_ownership_RENT   28501 non-null  uint8  
 11  loan_intent_EDUCATION        28501 non-null  uint8  
 12  loan_intent_HOMEIMPROVEMENT  28501 non-null  uint8  
 13  loan_intent_MEDICAL          28501 non-null  uint8  
 14  loan_intent_PERSONAL         28501 non-null  uint8  
 15  loan_intent_VENTURE          28501 non-null  uint8  
 16  loan_grade_B                 28501 non-null  uint8  
 17  loan_grade_C                 28501 non-null  uint8  
 18  loan_grade_D                 28501 non-null  uint8  
 19  loan_grade_E                 28501 non-null  uint8  
 20  loan_grade_F                 28501 non-null  uint8  
 21  loan_grade_G                 28501 non-null  uint8  
 22  cb_person_default_on_file_Y  28501 non-null  uint8  
dtypes: float64(3), int64(5), uint8(15)

4. Output do primeiro modelo:

                           Logit Regression Results                           
==============================================================================
Dep. Variable:                default   No. Observations:                28501
Model:                          Logit   Df Residuals:                    28478
Method:                           MLE   Df Model:                           22
Date:                Sun, 19 Mar 2023   Pseudo R-squ.:                  0.3584
Time:                        15:25:11   Log-Likelihood:                -9567.9
converged:                       True   LL-Null:                       -14913.
Covariance Type:            nonrobust   LLR p-value:                     0.000
===============================================================================================
                                  coef    std err          z      P>|z|      [0.025      0.975]
-----------------------------------------------------------------------------------------------
Intercept                      -4.1288      0.200    -20.695      0.000      -4.520      -3.738
person_age                     -0.0125      0.006     -1.995      0.046      -0.025      -0.000
person_income                8.763e-07   3.22e-07      2.723      0.006    2.46e-07    1.51e-06
person_emp_length              -0.0142      0.005     -2.831      0.005      -0.024      -0.004
loan_amnt                      -0.0001   4.29e-06    -23.511      0.000      -0.000   -9.24e-05
loan_int_rate                   0.0892      0.018      4.910      0.000       0.054       0.125
loan_percent_income            13.1512      0.252     52.208      0.000      12.657      13.645
cb_person_cred_hist_length      0.0105      0.009      1.113      0.266      -0.008       0.029
person_home_ownership_OTHER     0.4388      0.302      1.451      0.147      -0.154       1.031
person_home_ownership_OWN      -1.7874      0.113    -15.811      0.000      -2.009      -1.566
person_home_ownership_RENT      0.8258      0.043     19.212      0.000       0.742       0.910
loan_intent_EDUCATION          -0.8755      0.061    -14.310      0.000      -0.995      -0.756
loan_intent_HOMEIMPROVEMENT     0.0485      0.068      0.714      0.475      -0.085       0.182
loan_intent_MEDICAL            -0.1569      0.058     -2.715      0.007      -0.270      -0.044
loan_intent_PERSONAL           -0.6437      0.062    -10.300      0.000      -0.766      -0.521
loan_intent_VENTURE            -1.1430      0.067    -17.146      0.000      -1.274      -1.012
loan_grade_B                    0.1085      0.083      1.308      0.191      -0.054       0.271
loan_grade_C                    0.2372      0.125      1.901      0.057      -0.007       0.482
loan_grade_D                    2.3284      0.157     14.856      0.000       2.021       2.636
loan_grade_E                    2.4932      0.198     12.611      0.000       2.106       2.881
loan_grade_F                    2.7882      0.273     10.208      0.000       2.253       3.324
loan_grade_G                    6.3306      1.056      5.996      0.000       4.261       8.400
cb_person_default_on_file_Y     0.0216      0.053      0.407      0.684      -0.083       0.126
===============================================================================================

5. Output do modelo, após Stepwise:

                           Logit Regression Results                           
==============================================================================
Dep. Variable:                default   No. Observations:                28501
Model:                          Logit   Df Residuals:                    28484
Method:                           MLE   Df Model:                           16
Date:                Sun, 19 Mar 2023   Pseudo R-squ.:                  0.3581
Time:                        15:25:29   Log-Likelihood:                -9572.2
converged:                       True   LL-Null:                       -14913.
Covariance Type:            nonrobust   LLR p-value:                     0.000
==============================================================================================
                                 coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------------
Intercept                     -4.4405      0.136    -32.738      0.000      -4.706      -4.175
person_age                    -0.0063      0.003     -2.043      0.041      -0.012      -0.000
person_income               8.205e-07   3.15e-07      2.606      0.009    2.03e-07    1.44e-06
person_emp_length             -0.0148      0.005     -2.951      0.003      -0.025      -0.005
loan_amnt                     -0.0001   4.26e-06    -23.642      0.000      -0.000   -9.24e-05
loan_int_rate                  0.1218      0.008     14.642      0.000       0.106       0.138
loan_percent_income           13.1430      0.251     52.405      0.000      12.651      13.635
person_home_ownership_OWN     -1.7914      0.113    -15.883      0.000      -2.013      -1.570
person_home_ownership_RENT     0.8188      0.043     19.176      0.000       0.735       0.902
loan_intent_EDUCATION         -0.8915      0.056    -16.050      0.000      -1.000      -0.783
loan_intent_MEDICAL           -0.1751      0.052     -3.383      0.001      -0.277      -0.074
loan_intent_PERSONAL          -0.6614      0.057    -11.628      0.000      -0.773      -0.550
loan_intent_VENTURE           -1.1595      0.061    -18.862      0.000      -1.280      -1.039
loan_grade_D                   2.0715      0.062     33.315      0.000       1.950       2.193
loan_grade_E                   2.1822      0.101     21.539      0.000       1.984       2.381
loan_grade_F                   2.4284      0.193     12.556      0.000       2.049       2.807
loan_grade_G                   5.9077      1.030      5.735      0.000       3.889       7.927
==============================================================================================

6. Comparação dos modelos:

==========================================================
                            MODELO INICIAL MODELO STEPWISE
----------------------------------------------------------
Intercept                   -4.1288***     -4.4405***     
                            (0.1995)       (0.1356)       
cb_person_cred_hist_length  0.0105                        
                            (0.0095)                      
cb_person_default_on_file_Y 0.0216                        
                            (0.0532)                      
loan_amnt                   -0.0001***     -0.0001***     
                            (0.0000)       (0.0000)       
loan_grade_B                0.1085                        
                            (0.0830)                      
loan_grade_C                0.2372*                       
                            (0.1248)                      
loan_grade_D                2.3284***      2.0715***      
                            (0.1567)       (0.0622)       
loan_grade_E                2.4932***      2.1822***      
                            (0.1977)       (0.1013)       
loan_grade_F                2.7882***      2.4284***      
                            (0.2731)       (0.1934)       
loan_grade_G                6.3306***      5.9077***      
                            (1.0558)       (1.0302)       
loan_int_rate               0.0892***      0.1218***      
                            (0.0182)       (0.0083)       
loan_intent_EDUCATION       -0.8755***     -0.8915***     
                            (0.0612)       (0.0555)       
loan_intent_HOMEIMPROVEMENT 0.0485                        
                            (0.0679)                      
loan_intent_MEDICAL         -0.1569***     -0.1751***     
                            (0.0578)       (0.0518)       
loan_intent_PERSONAL        -0.6437***     -0.6614***     
                            (0.0625)       (0.0569)       
loan_intent_VENTURE         -1.1430***     -1.1595***     
                            (0.0667)       (0.0615)       
loan_percent_income         13.1512***     13.1430***     
                            (0.2519)       (0.2508)       
person_age                  -0.0125**      -0.0063**      
                            (0.0062)       (0.0031)       
person_emp_length           -0.0142***     -0.0148***     
                            (0.0050)       (0.0050)       
person_home_ownership_OTHER 0.4388                        
                            (0.3024)                      
person_home_ownership_OWN   -1.7874***     -1.7914***     
                            (0.1130)       (0.1128)       
person_home_ownership_RENT  0.8258***      0.8188***      
                            (0.0430)       (0.0427)       
person_income               0.0000***      0.0000***      
                            (0.0000)       (0.0000)       
N                           28501          28501          
Log-lik                     -9567.85       -9572.21       
==========================================================
Standard errors in parentheses.
* p<.1, ** p<.05, ***p<.01

7. Performance do modelo:

Cutoff: 30%
   Sensitividade  Especificidade  Acurácia
0       0.724144        0.882893  0.848426

![image](https://user-images.githubusercontent.com/94931093/226200009-f780568a-ab32-456d-8ed8-16d016e5b3fc.png)

Cutoff: 50%
   Sensitividade  Especificidade  Acurácia
0       0.561571        0.952853  0.867899

![image](https://user-images.githubusercontent.com/94931093/226200044-d2acb585-2f2f-47b0-ba7d-6fbe6b79f2e7.png)

Cutoff: 70%
   Sensitividade  Especificidade  Acurácia
0       0.355365         0.98409  0.847584

![image](https://user-images.githubusercontent.com/94931093/226200053-6a7bd5db-d1ba-4df3-98ba-3607754ec7cd.png)

8. Curva ROC:

![image](https://user-images.githubusercontent.com/94931093/226200083-7ed1e079-51ad-4d26-9f87-b11498af4a2b.png)

9. Análise Sensitividade x Especificidade:

![image](https://user-images.githubusercontent.com/94931093/226200116-33cf0f59-6d21-41e3-af95-4b29bd085b65.png)

