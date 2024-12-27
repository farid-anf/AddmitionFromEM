
## Anticipating Hospital Admissions from the Emergency Department
![My Image](wall.png)


### 1. Problem Identification

#### 1.1. Problem Statement

Most visits to the emergency department (ED) conclude with the patient being discharged, yet these departments stand out as the primary gateway for hospital admissions. When patients arrive at the ED, they undergo a critical sorting process known as "triage," which determines the urgency of their medical needs. This essential task is usually carried out by a skilled member of the nursing staff who assesses various factors, including the patient's demographic information, their principal complaint, and vital signs. After this initial evaluation, the patient is seen by a medical provider who formulates an initial care plan tailored to their specific situation. Based on this assessment, the provider ultimately makes a recommendation regarding the patient’s next steps, which in this study focuses on whether the patient should be admitted to the hospital or discharged to continue their recovery elsewhere.

This process is important because accurate triage and subsequent decisions significantly impact patient outcomes and hospital resource management. Proper triage ensures that patients with the most urgent needs receive timely care, which can be crucial for their recovery and survival. Additionally, effective triage and decision-making help optimize the use of hospital resources, reducing overcrowding in the emergency department and ensuring that hospital beds are available for those who truly need them.

If we cannot accurately predict the need for hospital admission versus discharge, several negative consequences can arise:

1. **Patient Safety Risks**: Patients who require urgent care might not receive timely treatment, leading to worsened health outcomes or even fatalities.

2. **Overcrowding**: Misjudging the severity of conditions can lead to overcrowded emergency departments, causing delays in treatment for all patients and increasing the risk of medical errors.

3. **Resource Misallocation**: Inefficient use of hospital resources, such as admitting patients who could safely be discharged or discharging patients who need inpatient care, can strain the healthcare system and increase costs.

4. **Increased Healthcare Costs**: Unnecessary admissions can lead to higher healthcare costs for both hospitals and patients, while inadequate care can result in complications requiring more extensive treatment later.

5. **Patient Dissatisfaction**: Delays and inefficiencies in care can lead to patient dissatisfaction, potentially reducing trust in the healthcare system.

Predictive models and accurate assessments are thus critical for maintaining the efficiency and effectiveness of emergency care, ensuring that patients receive appropriate treatment and resources are used optimally.

The main goal for this project is to predict hospital admissions at the time of ED triage by using patient demographics and information gathered during triage.

#### 1.2. Dataset Description

The dataset includes all adult emergency department visits from March 2014 to July 2017 at one academic and two community emergency rooms that resulted in either admission or discharge. A total of 972 variables were collected for each patient visit. In this study, we only consider the demographic information and information gathered during the triage:

**Demographics:**
- ``age``
- ``gender``
- ``ethnicity``
- ``race``
- ``language``
- ``religion``
- ``marital status``
- ``employment status``
- ``insurance status``

**Triage and Hospital Usage:**
- ``dep_name``: presenting hospital (recoded into A, B, C)
- ``esi``: ESI level determined by triage nurse
- ``disposition``: admission or discharge
- ``arrivalmode``: ambulance, walk-in, car, etc.
- ``arrivalmonth``: month of arrival (Jan-Dec)
- ``arrivalday``: day of arrival (Mon-Sun)
- ``arrivalhour_bin``: hour of arrival, binned to 4-hour timeframes (23-02, 03-06, etc.)
- ``triage_vital_hr``: heart rate recorded at triage
- ``triage_vital_sbp``: systolic blood pressure recorded at triage
- ``triage_vital_dbp``: diastolic blood pressure recorded at triage
- ``triage_vital_rr``: respiratory rate recorded at triage
- ``triage_vital_o2``: O2 saturation recorded at triage
- ``triage_vital_o2_device``: presence of supplementary O2 device at triage
- ``triage_vital_temp``: temperature recorded at triage
- ``n_edvisits``: number of ED visits within the past year
- ``n_admissions``: number of in-patient admissions within the past year
- ``previousdispo``: disposition of the patient's last visit to the ED
- ``n_surgeries``: number of surgeries and procedures within the past year

The dataset is available at: [Admission Prediction Dataset](https://github.com/yaleemmlc/admissionprediction/tree/master/Results)

