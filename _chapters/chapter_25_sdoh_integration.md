---
layout: chapter
title: "Chapter 25: Social Determinants of Health in Clinical Models"
chapter_number: 25
part_number: 6
prev_chapter: /chapters/chapter-24-population-health-screening/
next_chapter: /chapters/chapter-26-llms-in-healthcare/
---
# Chapter 25: Social Determinants of Health in Clinical Models

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Understand the theoretical frameworks linking social determinants of health to clinical outcomes and the empirical evidence demonstrating their primacy in population health
2. Design and implement systems for integrating external data sources on housing stability, food security, environmental exposures, and neighborhood characteristics with clinical records while maintaining data quality and patient privacy
3. Develop composite indices of social vulnerability that capture multidimensional disadvantage without reinforcing deficit-based narratives
4. Apply causal inference methods including mediation analysis and structural equation modeling to elucidate mechanisms through which social factors affect health outcomes
5. Implement screening, risk stratification, and intervention targeting systems that treat social determinants as mutable targets rather than fixed patient characteristics
6. Evaluate SDOH integration systems for their impact on reducing rather than perpetuating health inequities across marginalized populations

## Introduction

Social and structural determinants of health—the conditions in which people are born, grow, live, work, and age—account for an estimated thirty to fifty-five percent of health outcomes, substantially exceeding the contribution of clinical care which accounts for only ten to twenty percent (McGinnis et al., 2002; Hood et al., 2016). Income inequality, educational attainment, housing stability, food security, neighborhood safety, environmental exposures, social support networks, and experiences of discrimination fundamentally shape disease incidence, progression, and mortality across populations (Marmot and Wilkinson, 2005; Braveman and Gottlieb, 2014). Yet healthcare artificial intelligence systems frequently treat patients as isolated biological entities, ignoring the social context that drives most health variation.

This disconnection between the known determinants of health and the data used in healthcare AI creates multiple problems. First, models trained solely on clinical variables achieve suboptimal predictive performance because they omit the most important predictors of outcomes (Berkowitz et al., 2018). Second, interventions targeted based on clinical risk alone may miss patients whose primary needs are social rather than medical, leading to resource misallocation (Gottlieb et al., 2016). Third, algorithms that fail to account for social factors may attribute health disparities to individual behaviors or biology rather than structural inequities, reinforcing victim-blaming narratives and deflecting attention from upstream interventions (Churchwell et al., 2020).

Integrating social determinants into healthcare AI requires addressing substantial technical challenges. Social determinants data are fragmented across multiple sectors that do not traditionally share information with healthcare systems, including housing authorities, food banks, environmental monitoring agencies, schools, and criminal justice systems (Cantor and Thorpe, 2018). Individual-level social determinants collected through screening in clinical settings suffer from high missingness, social desirability bias, and rapidly changing circumstances (Gottlieb et al., 2019). Neighborhood-level measures drawn from census data or other aggregated sources risk ecological fallacy and may not reflect the lived experiences of residents (Diez Roux and Mair, 2010). Privacy regulations and justified community concerns about surveillance create barriers to data linkage (Veinot et al., 2019).

Beyond technical challenges, integrating social determinants requires careful attention to framing and interpretation. Deficit-based approaches that characterize marginalized populations by their "lack" of resources or "risky behaviors" perpetuate stigma and ignore the structural violence that produces social disadvantage (Metzl and Hansen, 2014). Asset-based frameworks that recognize community strengths, resilience, and existing support systems offer more respectful and actionable alternatives (Kretzmann and McKnight, 1993). Machine learning systems must treat social determinants as mutable intervention targets amenable to policy and programmatic solutions rather than fixed patient characteristics that predict inevitable decline.

This chapter develops comprehensive approaches for integrating social determinants into healthcare AI systems with explicit attention to reducing rather than perpetuating health inequities. We begin with theoretical frameworks and causal models linking social factors to health outcomes. We then present methods for linking clinical data with external sources on housing, food access, environmental exposures, and neighborhood characteristics while maintaining data quality and privacy protections. We develop approaches for collecting individual-level social determinants through screening instruments, addressing missingness and validation challenges. We present techniques for constructing composite measures of social vulnerability that capture multidimensional disadvantage. We apply mediation analysis, structural equation modeling, and other causal inference methods to elucidate mechanisms through which social factors affect health. Throughout, we emphasize framing that respects patient dignity, recognizes structural causes of disadvantage, and supports upstream intervention rather than individual blame.

## Theoretical Frameworks for Social Determinants of Health

### Conceptual Models Linking Social Factors to Health

Multiple theoretical frameworks describe pathways through which social determinants affect health outcomes. The World Health Organization's Social Determinants of Health framework positions structural determinants including governance, macroeconomic policies, social policies, and cultural norms as fundamental drivers that produce social stratification based on socioeconomic position, which in turn generates intermediary determinants including material circumstances, psychosocial factors, behavioral factors, and biological factors that directly affect health (Solar and Irwin, 2010). This model emphasizes that individual behaviors and biology are themselves shaped by social position and that interventions targeting downstream factors without addressing structural determinants will have limited effectiveness.

The Reserve Capacity Model posits that social resources buffer the negative health impacts of stressors and facilitate recovery from illness (Gallo et al., 2009). Individuals with greater economic resources, stronger social networks, safer neighborhoods, and better education have more capacity to withstand health challenges and prevent progression from acute to chronic conditions. This framework suggests that social determinants operate not only by increasing exposure to health risks but also by limiting protective factors and resources for managing health problems.

Fundamental Cause Theory argues that socioeconomic status remains associated with health across time and place despite changes in disease patterns because individuals with more resources are better able to adopt emerging health-protective behaviors, access new medical technologies, and avoid newly identified risks (Link and Phelan, 1995). This theory predicts that health disparities will persist or widen as new treatments become available unless structural inequities in resource access are addressed. For healthcare AI, this implies that predictive models capturing current social gradients in health may underestimate future disparities if innovations are unequally accessible.

The Weathering Hypothesis proposes that cumulative exposure to social and economic adversity, particularly racism and discrimination, produces accelerated biological aging and earlier deterioration in health among marginalized populations (Geronimus, 1992). This framework emphasizes allostatic load—the physiological toll of repeated adaptation to stress—as a key mechanism linking social position to chronic disease. Machine learning systems measuring biological age or frailty must consider whether observed differences reflect intrinsic aging versus structurally-imposed accelerated deterioration.

Mathematical representation of these frameworks requires specifying directed acyclic graphs that encode assumed causal relationships between structural determinants, social position variables, intermediate mechanisms, and health outcomes. Consider the following simplified structural equation model:

Let $$Z $$ represent structural determinants (policy environment, macroeconomic conditions), $$ S $$ represent social position (income, education, occupation), $$ M $$ represent intermediate mechanisms (stress, health behaviors, environmental exposures), and $$ Y$$ represent health outcomes. The structural equations might be:

$$S = f_S(Z, U_S)$$

$$M = f_M(Z, S, U_M)$$

$$Y = f_Y(Z, S, M, U_Y)$$

where $$U_S $$, $$ U_M $$, and $$ U_Y $$ represent unmeasured factors affecting each variable. This model encodes that structural determinants directly affect social position, both structural determinants and social position affect intermediate mechanisms, and all three affect health outcomes through potentially nonlinear functions $$ f$$.

The total effect of structural determinants on health decomposes into direct effects and indirect effects mediated through social position and intermediate mechanisms:

$$\text{Total Effect} = \frac{\partial Y}{\partial Z} + \frac{\partial Y}{\partial S}\frac{\partial S}{\partial Z} + \frac{\partial Y}{\partial M}\left(\frac{\partial M}{\partial Z} + \frac{\partial M}{\partial S}\frac{\partial S}{\partial Z}\right)$$

Estimating these causal effects from observational data requires addressing confounding, measurement error, and selection bias—challenges we address through causal inference methods later in this chapter.

### Empirical Evidence on Social Determinants

Extensive empirical research documents the magnitude of social determinants' effects on health. A systematic review of longitudinal studies found that low socioeconomic status was associated with a fifty percent increase in premature mortality, with effects of similar or greater magnitude than traditional clinical risk factors including hypertension, diabetes, smoking, and obesity (Stringhini et al., 2017). Educational attainment shows particularly strong associations with health, with each additional year of schooling associated with a nine percent reduction in all-cause mortality risk (Zajacova and Lawrence, 2018).

Housing instability and homelessness are associated with dramatically worse health outcomes across multiple domains. Individuals experiencing homelessness have mortality rates three to four times higher than housed populations, with life expectancies reduced by approximately fifteen to thirty years (Aldridge et al., 2018). Even marginal housing—defined as overcrowding, poor quality, or excessive cost burden—is associated with increased respiratory disease, cardiovascular disease, mental health disorders, and injury (Shaw, 2004). Housing insecurity operates through multiple mechanisms including increased exposure to environmental hazards, chronic stress, disrupted healthcare access, and competing demands that deprioritize health maintenance.

Food insecurity—limited or uncertain availability of nutritionally adequate food—affects approximately ten percent of U.S. households and is strongly associated with poor health outcomes (Coleman-Jensen et al., 2020). Among adults, food insecurity is associated with increased prevalence of diabetes, hypertension, hyperlipidemia, and worse control of existing chronic conditions (Seligman and Schillinger, 2010). Children experiencing food insecurity show higher rates of developmental delays, behavioral problems, and acute and chronic health conditions (Gundersen and Ziliak, 2015). These associations persist after controlling for income, suggesting that food insecurity captures dimensions of material hardship beyond general poverty.

Neighborhood characteristics including concentrated poverty, residential segregation, limited access to healthy food and safe spaces for physical activity, environmental pollution, and violence exposure substantially affect health (Diez Roux and Mair, 2010). Residing in neighborhoods with high poverty rates is associated with increased cardiovascular disease, stroke, cancer, and premature mortality, even after adjusting for individual socioeconomic status (Havranek et al., 2015). Residential segregation by race and ethnicity, which results from historical policies including redlining and continues through contemporary discriminatory practices, concentrates disadvantage and produces health disparities independent of individual characteristics (Williams and Collins, 2001).

Environmental exposures represent another critical pathway through which social disadvantage affects health. Low-income communities and communities of color experience disproportionate exposure to air pollution, water contamination, hazardous waste sites, and extreme heat (Mohai et al., 2009). These exposures contribute to asthma, cardiovascular disease, cognitive impairment, adverse birth outcomes, and cancer (Landrigan et al., 2018). Machine learning systems for healthcare must account for geographic variation in environmental health risks if they are to avoid attributing environmentally-driven disease to patient characteristics.

Social isolation and lack of social support networks are associated with increased morbidity and mortality comparable to traditional risk factors (Holt-Lunstad et al., 2010). Mechanisms include reduced access to instrumental support for health management, increased stress, and behavioral pathways including reduced health service utilization. Importantly, social isolation is not randomly distributed but reflects structural factors including residential mobility driven by housing insecurity, long work hours in low-wage jobs, and fractured communities resulting from mass incarceration and immigration enforcement (Umberson and Montez, 2010).

Experiences of discrimination and racism constitute critical social determinants with direct physiological effects. Perceived discrimination is associated with increased inflammation, hypertension, cardiovascular disease, depression, substance use, and mortality (Pascoe and Smart Richman, 2009; Williams and Mohammed, 2013). These associations operate through chronic activation of stress response systems—the weathering effects described earlier. Importantly, discrimination's health impacts are not fully captured by socioeconomic status or neighborhood disadvantage, indicating independent causal pathways requiring explicit measurement and intervention.

## Data Sources for Social Determinants

### External Data Linkage

Integrating social determinants into healthcare AI requires linking clinical records with data from multiple external sources. Each source provides complementary information but presents unique challenges regarding data quality, temporal alignment, privacy protection, and appropriate use.

#### Housing and Homelessness Data

Housing data are available from multiple sources including:

- **Public housing authorities**: Data on individuals residing in public housing or receiving housing vouchers, including unit characteristics, building conditions, and neighborhood amenities. Access requires data use agreements and must comply with HUD privacy regulations.

- **Homeless management information systems (HMIS)**: Federally-mandated databases tracking individuals receiving homeless services, including emergency shelter, transitional housing, and permanent supportive housing. HMIS data provide information on housing history, service utilization, and barriers to housing stability.

- **Eviction records**: Court records documenting eviction filings and judgments, available through public records requests or commercial vendors. Eviction data capture housing instability not reflected in homelessness databases.

- **Property tax and assessment data**: Municipal databases containing information on property characteristics, ownership, value, and tax delinquency. These data enable neighborhood-level analyses but typically lack individual-level linkage.

Linking housing data to clinical records requires probabilistic matching on name, date of birth, and address when unique identifiers are unavailable. Challenges include frequent address changes among housing-insecure populations, name variations and misspellings, and privacy concerns about sharing sensitive housing information with healthcare systems. Temporal alignment is critical—housing status may change frequently and historical housing patterns may be more relevant than current status for understanding cumulative exposures.

#### Food Access and Food Insecurity Data

Food environment data include:

- **Food retailer locations**: Databases of grocery stores, supermarkets, convenience stores, and restaurants compiled by government agencies or commercial vendors. These data enable calculation of food access metrics including distance to nearest supermarket, density of fast food restaurants, and neighborhood food environment scores.

- **SNAP participation**: Administrative data on Supplemental Nutrition Assistance Program enrollment and benefit utilization, available through data use agreements with state agencies. SNAP data indicate food assistance receipt but do not directly measure food security.

- **Food bank utilization**: Records from food pantries and food distribution programs documenting individuals receiving food assistance. These data are fragmented across organizations and lack standardization.

- **Food insecurity screening**: Patient-reported measures collected through clinical screening instruments (discussed in detail below).

Geocoding patient addresses and calculating spatial access to food resources provides neighborhood-level characterization but may not reflect household food security status. Individual-level food insecurity data from screening instruments are preferred when available but suffer from missingness and reporting bias.

#### Environmental Exposure Data

Environmental health data relevant to healthcare AI include:

- **Air quality monitoring**: Regulatory monitoring data from EPA including criteria pollutants (particulate matter, ozone, nitrogen dioxide, sulfur dioxide, carbon monoxide, lead) measured at fixed monitoring stations. Satellite-derived estimates provide broader spatial coverage.

- **Water quality data**: Municipal water testing results and violations of Safe Drinking Water Act standards, available through EPA databases. Lead testing results from schools and homes provide additional exposure data.

- **Superfund and hazardous waste sites**: Databases of contaminated sites requiring cleanup (EPA Superfund), toxic release inventory (TRI) facilities reporting chemical releases, and brownfields. Distance to these sites and modeled exposure estimates are used as neighborhood-level risk factors.

- **Climate and heat data**: Temperature records, heat index calculations, and urban heat island mapping identify extreme heat exposure. Climate projections enable forward-looking risk assessment.

Environmental data typically require geocoding and spatial linkage to assign exposure estimates to individuals based on residential address. Challenges include:

1. Spatial misalignment—monitoring stations may not represent exposure at residential locations, particularly in areas with sparse monitoring
2. Temporal misalignment—historical exposure data may be unavailable or exposure may have changed since monitoring
3. Activity patterns—residence-based exposure estimates miss workplace and commute exposures
4. Cumulative and mixture effects—multiple simultaneous exposures may have synergistic health impacts not captured by single-pollutant models

Sophisticated exposure assessment methods including land use regression models, dispersion models, and satellite-based estimates improve upon simple proximity metrics but require specialized expertise and computational resources (Sampson et al., 2013).

#### Neighborhood and Census Data

Neighborhood socioeconomic characteristics drawn from census and survey data provide contextual information on area-level disadvantage. Key data sources include:

- **American Community Survey (ACS)**: Annual survey providing estimates of population demographics, income, poverty, education, employment, housing characteristics, and other socioeconomic variables at multiple geographic scales (census tract, block group, ZIP code).

- **Area Deprivation Index (ADI)**: Composite measure of neighborhood disadvantage incorporating income, education, employment, and housing quality, available at census block group level (Kind and Buckingham, 2018).

- **Social Vulnerability Index (SVI)**: CDC-developed measure combining fifteen social factors across four domains (socioeconomic status, household composition, minority status/language, housing/transportation) to identify communities vulnerable to disasters and public health emergencies.

- **Child Opportunity Index**: Composite measure of neighborhood resources and conditions affecting child development, including educational quality, health and environmental factors, and social and economic opportunity.

- **Walkability and built environment data**: Measures of pedestrian infrastructure, transit access, parks and recreation, and mixed land use that affect physical activity and social interaction.

Linking census data requires geocoding patient addresses to census geographies. Challenges include:

- Ecological fallacy—area-level characteristics may not reflect individual circumstances
- Geographic scale effects—results vary depending on whether census tract, block group, ZIP code, or other geography is used
- Temporal lag—ACS estimates reflect prior years and neighborhoods may change between survey and analysis
- Privacy versus precision tradeoffs—smaller geographies provide more precise characterization but increase reidentification risk

#### Criminal Justice and Safety Data

Community safety data include:

- **Crime statistics**: Incident-level crime data from law enforcement agencies including violent crime, property crime, and drug-related arrests. These data enable calculation of neighborhood crime rates and violence exposure.

- **Incarceration records**: Administrative data on arrest, conviction, incarceration, and release from correctional systems. Incarceration history is associated with multiple adverse health outcomes but is highly sensitive and must be used cautiously.

- **Community violence interventions**: Data from hospital-based or community-based violence interruption programs documenting individuals at high risk for violence involvement.

Criminal justice data raise substantial ethical concerns. Policing and incarceration disproportionately affect communities of color due to structural racism in law enforcement, sentencing, and criminal justice policy (Alexander, 2010). Using arrest or incarceration data in healthcare AI risks perpetuating these injustices by labeling individuals from over-policed communities as high-risk. Crime statistics may reflect policing intensity rather than actual crime rates, particularly for low-level offenses. Violence exposure measures that focus on victimization rather than justice system involvement are generally more appropriate for healthcare applications.

### Privacy-Preserving Data Linkage

Linking clinical data with external sources requires balancing the scientific and clinical value of integrated data against privacy risks and ethical concerns about surveillance. Technical approaches for privacy-preserving linkage include:

#### Honest Broker Models

Honest broker systems separate patient identification from data analysis (Boyd et al., 2007). The honest broker—a trusted intermediary not involved in research or clinical care—receives identifying information from all data sources, performs linkage, assigns study IDs, and provides de-identified linked data to analysts. This prevents analysts from accessing identifying information while enabling integration of multiple data sources.

Honest brokers may be internal (staff employed by the data holder) or external (independent third parties). External honest brokers provide stronger privacy protections but increase complexity and cost. Key considerations include:

- Clearly defined policies governing honest broker procedures and data use
- Technical controls preventing analysts from receiving identified data
- Audit trails documenting all data accesses and linkages
- Limited time periods for data access and requirements for data destruction after analysis

#### Cryptographic Linkage

Cryptographic methods enable linkage without sharing identifying information between organizations. Privacy-preserving record linkage (PPRL) protocols allow two parties to identify matching records using encrypted identifiers without either party learning the other's non-matching records (Schnell et al., 2009).

Bloom filter-based PPRL hashes identifying fields (name, date of birth, address) into bit arrays called Bloom filters. Filters are compared using similarity measures to identify potential matches without decrypting the underlying identifiers. More sophisticated approaches including garbled circuits, homomorphic encryption, and secure multi-party computation provide strong privacy guarantees but are computationally expensive and complex to implement (Vatsalan et al., 2017).

Differential privacy mechanisms add calibrated noise to query results to prevent inference about individual records (Dwork and Roth, 2014). When applied to record linkage, differential privacy can provide formal privacy guarantees while enabling approximate matching for analysis. Challenges include determining appropriate privacy budgets and managing accuracy-privacy tradeoffs.

#### Federated and Decentralized Approaches

Federated learning enables model training across multiple organizations without sharing individual-level data (McMahan et al., 2017). Each organization trains local models on their data and shares only aggregated model parameters, preserving data privacy while enabling collaboration. For SDOH integration, healthcare systems and social service agencies could collaboratively train models predicting health outcomes from social determinants without either party directly accessing the other's data.

Blockchain-based systems provide decentralized infrastructure for managing data sharing permissions and tracking data use without centralized authority (Ekblaw et al., 2016). Patients could grant time-limited, purpose-specific permissions for their healthcare data to be linked with social service data, with all access logged on an immutable ledger. Technical and governance challenges have limited adoption of blockchain in healthcare to date.

### Data Quality and Validation

Social determinants data from external sources require careful quality assessment before use in healthcare AI:

#### Completeness and Coverage

External data sources often lack complete population coverage. Housing authority databases only include public housing residents and voucher recipients, missing privately-housed individuals. HMIS data capture only individuals engaging with homeless services, missing unsheltered individuals avoiding systems. Food bank data exclude households obtaining food assistance through other means or experiencing food insecurity without accessing formal services.

Assessing coverage requires comparing linked records to known demographics and conducting sensitivity analyses excluding cases with missing external data. When large fractions of the population lack external data linkage, neighborhood-level proxies may be necessary despite reduced precision.

#### Temporal Accuracy

Social determinants change over time, often rapidly for vulnerable populations. Housing status, food security, and employment may fluctuate within days or weeks. External databases with infrequent updates or historical data may not reflect current circumstances when healthcare encounters occur.

Strategies for managing temporal misalignment include:

1. Using time-stamped data and matching based on temporal proximity to healthcare encounters
2. Modeling trajectories and changes in social determinants over time rather than treating them as static
3. Incorporating uncertainty quantification that accounts for data staleness
4. Prioritizing frequently-updated sources over static historical data

#### Measurement Validity

External data may not measure constructs of interest directly. Crime statistics reflect reported and recorded crime, which differs from actual crime rates due to reporting bias and policing practices. Distance to food retailers does not capture food prices, quality, transportation access, or household resources that determine actual food security. Environmental monitoring station readings may not represent exposure at specific residential locations.

Validating external data against gold standard measures—such as comparing geocoded food access metrics against patient-reported food security or comparing census poverty rates against household income data—identifies measurement error and informs appropriate use. When validation studies are unavailable, transparency about data limitations and conservative interpretation of results are essential.

## Individual-Level Social Determinants Screening

While external data sources provide valuable contextual information, direct screening of patients for social determinants enables individual-level assessment and immediate clinical response. Multiple standardized screening instruments have been developed and validated for healthcare settings.

### Screening Instruments

#### Protocol for Responding to and Assessing Patients' Assets, Risks, and Experiences (PRAPARE)

PRAPARE is a comprehensive social determinants screening tool developed through a national collaborative process with input from community health centers and patient advocacy organizations (National Association of Community Health Centers, 2016). The instrument includes fifteen core items assessing:

- Demographics (race, ethnicity, language preference)
- Income and financial strain
- Employment and education
- Housing stability and homelessness
- Food insecurity
- Transportation barriers
- Safety concerns
- Social isolation
- Utilities insecurity
- Immigration status and residency

PRAPARE uses patient-centered language and strengths-based framing. For example, it asks "What is the highest level of school that you have finished?" rather than deficit-framing like "Do you lack education?" The instrument is designed for self-administration or staff-administration and typically requires five to seven minutes to complete.

#### Health Leads Screening Toolkit

The Health Leads screening toolkit provides validated questions for assessing specific social needs including housing instability, food insecurity, transportation barriers, utilities insecurity, and interpersonal violence (Health Leads, 2018). Questions are derived from validated surveys and adapted for clinical workflow integration. The toolkit emphasizes brief screening that can be completed quickly and incorporated into routine care.

Example validated questions include:

- Food security: "Within the past 12 months, you worried whether your food would run out before you got money to buy more" (from USDA Food Security Survey)
- Housing stability: "In the past 12 months, have you been homeless or stayed in a shelter, doubled up with other people, or stayed in a place not meant for sleeping?"
- Utilities: "In the past 12 months, has the electric, gas, oil, or water company threatened to shut off services in your home?"

#### WE CARE (Well Child Care, Evaluation, Community Resources, Advocacy, Referral, Education)

WE CARE is a validated screening and referral intervention specifically designed for pediatric primary care to address family social determinants (Garg et al., 2007). Parents complete a written screening questionnaire assessing needs across twelve domains:

- Education and job training
- Food security
- Housing quality and stability
- Household goods
- Utilities
- Child care
- Parental health insurance
- Child health insurance
- Transportation
- Legal assistance
- Mental health services
- Substance abuse treatment

Identified needs trigger referrals to community resources with documented effectiveness in increasing resource connection and improving family outcomes (Garg et al., 2015).

### Implementation Considerations

Implementing social determinants screening in healthcare settings requires attention to workflow integration, staff training, patient experience, and response capacity.

#### Workflow Integration

Screening can be implemented at multiple points in care delivery:

- Pre-visit: Electronic or paper questionnaires completed in waiting rooms or through patient portals before appointments enable review during visits
- During rooming: Medical assistants or nurses can administer screening when obtaining vital signs
- Physician-administered: Providers can incorporate screening into history-taking
- Post-visit: Screening at discharge or after visits enables targeted follow-up

Electronic health record integration is essential for systematic screening and documentation. Screening responses should populate discrete fields enabling querying and analysis rather than unstructured notes. Clinical decision support can alert providers to identified needs and suggest appropriate responses.

Workflow considerations include minimizing burden on staff and patients, ensuring screening occurs at appropriate frequency (typically annually or with significant life changes), and creating clear pathways for acting on identified needs.

#### Trauma-Informed Approaches

Social determinants screening asks about sensitive topics including housing instability, food insecurity, and safety concerns that may be sources of shame, stigma, or trauma. Trauma-informed approaches recognize the prevalence of trauma among healthcare populations and structure screening to avoid retraumatization (Substance Abuse and Mental Health Services Administration, 2014).

Trauma-informed screening principles include:

1. **Universal screening**: Screening all patients rather than targeting based on perceived need avoids stigmatization and implicit bias
2. **Patient choice**: Allowing patients to decline screening or skip specific questions respects autonomy
3. **Clear purpose**: Explaining why screening is conducted and how information will be used builds trust
4. **Non-judgmental language**: Using neutral, strengths-based language avoids blame
5. **Immediate response**: Having resources available to address identified needs prevents re-exposure to hopelessness
6. **Privacy**: Conducting screening in private settings and ensuring confidentiality protects safety

#### Response Capacity and Resource Navigation

Screening without capacity to respond risks causing harm by raising expectations without meeting needs (Alderwick and Gottlieb, 2019). Healthcare organizations implementing screening must develop:

- **Resource databases**: Curated, regularly-updated inventories of community resources addressing identified needs, including eligibility criteria, locations, hours, languages served, and application procedures
- **Navigation support**: Dedicated staff (community health workers, social workers, navigators) who help patients access resources, complete applications, and overcome barriers
- **Referral tracking**: Systems documenting which referrals are made, whether patients successfully connect with resources, and whether needs are resolved
- **Feedback loops**: Processes for identifying gaps in community resources and advocating for systemic changes

The closed-loop referral model emphasizes bidirectional communication between healthcare providers and community organizations, with confirmation when patients successfully access services and identification of barriers when connections fail (Cantor et al., 2020).

### Addressing Missingness and Bias

Social determinants screening data suffer from substantial missingness and potential bias that must be addressed in analysis and modeling.

#### Missingness Patterns

Screening data may be missing for multiple reasons:

- **Structural missingness**: Screening not implemented for certain populations, time periods, or clinical contexts
- **Declined screening**: Patients choose not to complete screening due to privacy concerns, lack of time, or other reasons
- **Item non-response**: Patients complete screening but skip sensitive questions
- **Recording failures**: Screening completed but responses not documented in EHR

Missingness is rarely random. Patients with higher social needs may be less likely to complete screening due to competing demands, distrust of healthcare systems, or lack of perceived benefit (Gottlieb et al., 2019). Conversely, patients with very low social needs may decline screening perceiving it as not relevant. This creates non-random missingness that biases analyses.

#### Imputation and Sensitivity Analysis

Multiple imputation methods can address missingness by creating multiple completed datasets incorporating uncertainty about missing values (Rubin, 1987). For social determinants screening, appropriate imputation requires:

1. Including auxiliary variables predictive of both missingness and outcomes, such as neighborhood-level measures, prior healthcare utilization, and other observed patient characteristics
2. Using imputation methods appropriate for categorical responses, such as multinomial logistic regression or random forests
3. Conducting sensitivity analyses varying assumptions about missingness mechanisms
4. Acknowledging limitations when missingness is severe or clearly non-random

Pattern mixture models explicitly model different distributions for observed versus missing data, allowing characterization of bias when data are missing not at random (Little, 1993).

#### Social Desirability Bias

Patients may underreport social needs due to shame, fear of judgment, or concerns about consequences (e.g., housing instability triggering child protective services involvement). Social desirability bias leads to underestimation of needs and may differentially affect marginalized populations with greater historical trauma from healthcare and social service systems.

Mitigation strategies include:

- Using validated screening questions with known response characteristics
- Employing self-administration methods that provide greater privacy
- Training staff in non-judgmental, trauma-informed approaches
- Validating screening against external data sources when available
- Acknowledging uncertainty in prevalence estimates

## Composite Measures of Social Vulnerability

Single social determinants interact and combine to produce cumulative disadvantage. Composite measures capturing multidimensional social vulnerability provide parsimonious characterization of overall risk while acknowledging complexity.

### Theoretical Foundations

Composite vulnerability measures draw on several theoretical traditions:

#### Intersectionality Theory

Intersectionality theory, originating in critical race theory and Black feminist scholarship, emphasizes that multiple marginalized identities (race, gender, class, disability, sexuality) combine to produce unique experiences of discrimination and disadvantage that are not simply additive (Crenshaw, 1989). For social determinants measurement, intersectionality implies that the combined effects of multiple vulnerabilities may exceed the sum of individual effects.

Operationalizing intersectionality in quantitative research remains challenging. Approaches include:

1. **Multiplicative interaction terms**: Testing whether effects of one social determinant vary by levels of another
2. **Stratified analysis**: Separately examining relationships within subgroups defined by multiple characteristics
3. **Multilevel analysis of individual heterogeneity (MAIH)**: Methods that allow effects to vary across individuals rather than assuming homogeneous effects

#### Cumulative Disadvantage Theory

Cumulative disadvantage theory posits that inequalities accumulate over the life course, with early disadvantages creating later vulnerability through reduced access to resources and opportunities (Dannefer, 2003). For health, early childhood poverty, education barriers, and adverse exposures create trajectories of disadvantage that compound over decades.

Measuring cumulative disadvantage requires longitudinal data capturing exposures across the life course. In cross-sectional analyses, historical contextual data (e.g., neighborhood conditions during childhood, which can be reconstructed from census data if birth location is known) can partially capture life course effects.

#### Capabilities Approach

The capabilities approach, developed by Amartya Sen and Martha Nussbaum, defines wellbeing in terms of capabilities—what people are able to do and be—rather than resources or outcomes (Sen, 1999; Nussbaum, 2011). This framework emphasizes that social determinants matter because they constrain capabilities for health, not merely because of material deprivation.

For measurement, the capabilities approach suggests focusing on freedom and opportunity rather than simply current status. For example, assessing not just current food security but capacity to obtain adequate food; not just current housing but stability and control over housing situation.

### Construction Methods

#### Index Development Approaches

Multiple methods exist for combining individual social determinants into composite indices:

**Sum scores** simply add indicators of different vulnerabilities, possibly with binary coding (e.g., 1 if vulnerable on dimension, 0 if not). This approach is transparent and easily interpretable but weights all dimensions equally regardless of their relative importance for health outcomes.

**Weighted scores** apply differential weights to components based on associations with outcomes, expert judgment, or theoretical importance. Weights can be derived through regression models predicting outcomes from individual determinants, with standardized coefficients used as weights. Principal components analysis (PCA) or factor analysis can derive empirical weights based on covariation among determinants.

**Latent variable models** treat observed social determinants as indicators of an underlying latent vulnerability construct. Structural equation modeling with latent variables estimates measurement models linking latent vulnerability to observed indicators and structural models linking latent vulnerability to outcomes (Bollen, 1989). This approach accounts for measurement error in individual indicators and provides formal tests of construct validity.

**Machine learning methods** including gradient boosting, random forests, or neural networks can learn complex nonlinear combinations of social determinants that optimally predict outcomes. These methods may identify important interactions but produce less interpretable indices.

#### Area Deprivation Index (ADI)

The Area Deprivation Index, developed by the Health Resources and Services Administration and refined by researchers at the University of Wisconsin, combines seventeen census variables from four domains using factor analysis (Singh, 2003; Kind and Buckingham, 2018). The domains are:

1. **Income/poverty**: Median family income, income disparity, families below poverty
2. **Education**: Population with less than high school education, population with at least high school education
3. **Employment**: Unemployed, employed in white collar jobs
4. **Housing quality**: Median home value, median mortgage, owner-occupied homes, single parent households, households without vehicle, households with more than one person per room, households without telephone

Factor analysis identifies weighted combinations of these variables that explain maximum variance. The first factor, interpreted as overall neighborhood disadvantage, becomes the ADI. Scores are ranked and typically reported as national or state percentile ranks, with higher ranks indicating greater deprivation.

ADI has been validated as a predictor of multiple health outcomes including mortality, hospitalizations, readmissions, and disease prevalence (Kind et al., 2014; Kind et al., 2018). The measure is publicly available at the census block group level and is widely used in health services research and clinical applications.

#### Social Vulnerability Index (SVI)

The CDC's Social Vulnerability Index combines fifteen variables across four themes using equal weighting (Flanagan et al., 2011):

1. **Socioeconomic status**: Below poverty, unemployed, income, no high school diploma
2. **Household composition and disability**: Aged 65 or older, aged 17 or younger, older than age 5 with disability, single-parent households
3. **Minority status and language**: Minority, speaks English less than well
4. **Housing type and transportation**: Multi-unit structures, mobile homes, crowding, no vehicle, group quarters

Variables are percentile-ranked and summed within themes, then theme scores are summed to create overall SVI. The index is designed specifically to identify communities vulnerable to disasters and public health emergencies, focusing on both baseline disadvantage and factors affecting response capacity.

SVI is publicly available at census tract level and is widely used by emergency management agencies and public health departments. Research has demonstrated associations with COVID-19 outcomes, hurricane impacts, and other disaster vulnerabilities (Karaye and Horney, 2020).

### Avoiding Deficit-Based Framing

Composite vulnerability measures risk reinforcing deficit-based narratives that characterize marginalized communities solely by what they lack. Critical perspectives on vulnerability measurement emphasize several concerns (Metzl and Hansen, 2014; Ford and Airhihenbuwa, 2010):

1. **Pathologizing communities**: Describing communities as "vulnerable," "disadvantaged," or "deprived" may stigmatize residents and deflect attention from structural causes
2. **Individual attribution**: Even when measures incorporate structural factors, communication often suggests individual deficits rather than systemic problems
3. **Missing assets**: Focusing exclusively on vulnerabilities ignores community strengths, social capital, resilience, and existing resources
4. **Justifying surveillance**: Vulnerability assessment may justify increased monitoring and intervention in marginalized communities without addressing root causes

Asset-based community development frameworks offer alternatives that begin with community strengths, capacities, and existing resources rather than needs and deficits (Kretzmann and McKnight, 1993). For healthcare AI, this suggests:

- **Balanced assessment**: Measuring both challenges and strengths, including social support networks, community organizations, cultural assets, and resilience factors
- **Community voice**: Involving community members in defining relevant measures rather than imposing researcher- or clinician-defined constructs
- **Structural framing**: Explicitly naming structural causes (e.g., "communities experiencing disinvestment" rather than "disadvantaged communities")
- **Actionable focus**: Emphasizing modifiable structural factors amenable to intervention rather than fixed characteristics

## Causal Inference for Social Determinants

Understanding how social determinants causally affect health—and identifying which interventions will reduce health inequities—requires methods that go beyond observational associations to estimate causal effects.

### Challenges for Causal Inference

Several challenges complicate causal inference for social determinants:

**Confounding**: Social determinants are not randomly assigned. Unobserved factors (genetics, preferences, unmeasured childhood experiences) may affect both social determinants and health outcomes, creating spurious associations. For example, individuals with better health may be more able to maintain stable housing, creating reverse causation.

**Measurement error**: Social determinants and confounders are often measured imperfectly. Neighborhood disadvantage indices drawn from census data aggregate across heterogeneous populations. Self-reported income, education, and social circumstances suffer from recall bias and social desirability bias.

**Time-varying confounding**: Social determinants, confounders, and outcomes evolve over time with complex feedback relationships. Poor health may lead to job loss, which worsens housing stability, which further deteriorates health. Standard regression adjustment fails when confounders are affected by prior treatment.

**Positivity violations**: Some populations may have near-zero probability of certain social circumstances. For example, in highly segregated contexts, Black residents may have very low probability of living in high-resource neighborhoods. Causal estimates for counterfactual scenarios with no empirical support lack credibility.

### Mediation Analysis

Mediation analysis decomposes total effects into direct effects and indirect effects operating through specific pathways (VanderWeele, 2015). For social determinants, mediation analysis can elucidate mechanisms through which structural factors affect health.

Consider the causal diagram: Structural determinant $$Z $$ → Social position $$ S $$ → Intermediate mechanism $$ M $$ → Health outcome $$ Y $$, with possible direct effects $$ Z \rightarrow Y $$ and $$ S \rightarrow Y $$ bypassing mediators.

The total effect of $$ Z $$ on $$ Y$$ decomposes as:

$$TE = NDE + NIE$$

where $$NDE $$ is the natural direct effect (effect not mediated by $$ S $$ or $$ M $$) and $$ NIE $$ is the natural indirect effect (effect mediated through $$ S $$ and $$ M$$).

Under sequential ignorability assumptions—no unmeasured confounding of treatment-outcome, treatment-mediator, and mediator-outcome relationships—these effects can be identified using regression-based approaches:

$$NDE = E[Y(z=1, M(z=0))] - E[Y(z=0, M(z=0))]$$

$$NIE = E[Y(z=1, M(z=1))] - E[Y(z=1, M(z=0))]$$

where $$Y(z, M(z'))$$ denotes the potential outcome under treatment $$ z $$ and mediator value it would take under treatment $$ z'$$.

Estimation typically proceeds through two regression models:

1. **Outcome model**: Regress $$ Y $$ on $$ Z $$, $$ M $$, and confounders
2. **Mediator model**: Regress $$ M $$ on $$ Z $$ and confounders

Combining these models and integrating over the mediator distribution yields effect estimates. The proportion mediated is $$ PM = NIE / TE$$.

For example, to understand how neighborhood poverty affects cardiovascular disease risk through food environment, one would estimate:

- Total effect of neighborhood poverty on CVD
- Indirect effect mediated through food access (measured by supermarket density or food insecurity)
- Direct effect operating through other pathways (stress, environmental exposures, healthcare access)

Sensitivity analysis is essential given strong sequential ignorability assumptions. Methods quantify how strong unmeasured confounding would need to be to explain observed mediation effects (VanderWeele, 2010).

### Structural Equation Modeling

Structural equation modeling (SEM) provides a comprehensive framework for testing complex causal models with multiple mediators, outcomes, and feedback relationships (Bollen, 1989; Kline, 2015).

A structural equation model consists of two components:

**Measurement model**: Relates observed variables to latent constructs

$$X = \Lambda_X \xi + \delta$$

where $$X $$ is a vector of observed indicators, $$\xi $$ is a vector of latent exogenous variables, $$\Lambda_X $$ is a matrix of factor loadings, and $$\delta$$ is measurement error.

**Structural model**: Specifies causal relationships among latent variables

$$\eta = B\eta + \Gamma\xi + \zeta$$

where $$\eta $$ is a vector of latent endogenous variables, $$ B $$ captures effects among endogenous variables, $$\Gamma $$ captures effects of exogenous on endogenous variables, and $$\zeta $$ is structural error.

For social determinants, latent variables might include constructs like "neighborhood disadvantage" (measured by census indicators), "chronic stress" (measured by cortisol, blood pressure, self-reports), and "health resilience" (measured by recovery time, adaptation capacity). Structural paths would specify how neighborhood disadvantage affects chronic stress, which in turn affects health outcomes, with possible moderating effects of resilience.

Model fit is assessed using multiple indices:

- **Chi-square test**: Tests whether model-implied covariance matrix matches observed covariance (non-significant chi-square indicates good fit, though sensitive to sample size)
- **RMSEA (Root Mean Square Error of Approximation)**: Values < 0.05 indicate good fit, < 0.08 acceptable
- **CFI (Comparative Fit Index)**: Values > 0.95 indicate good fit
- **SRMR (Standardized Root Mean Square Residual)**: Values < 0.08 indicate good fit

Modification indices identify potential model improvements, though purely data-driven modification risks capitalization on chance.

### Instrumental Variable Methods

Instrumental variable (IV) methods address confounding by identifying variables that affect treatment but influence outcomes only through treatment (Angrist and Pischke, 2009). A valid instrument $$ Z $$ must satisfy:

1. **Relevance**: $$ Z $$ is associated with treatment $$ D $$
2. **Exclusion**: $$ Z $$ affects outcome $$ Y $$ only through $$ D $$
3. **Exchangeability**: $$ Z $$ is independent of unmeasured confounders $$ U$$

For social determinants, natural experiments and policy changes provide potential instruments. Examples include:

- **Housing voucher lotteries**: Random assignment of housing vouchers affects neighborhood quality but (conditionally on applying) should be independent of unmeasured factors affecting health
- **School catchment boundaries**: Discontinuities in school quality at geographic boundaries create variation in educational opportunity independent of family characteristics
- **Policy implementations**: Minimum wage increases, Medicaid expansions, or other policy changes create variation in social determinants not driven by individual characteristics

The two-stage least squares (2SLS) estimator proceeds:

**First stage**: Regress treatment on instrument and covariates

$$D_i = \alpha_0 + \alpha_1 Z_i + \alpha_2' X_i + \epsilon_i$$

**Second stage**: Regress outcome on predicted treatment and covariates

$$Y_i = \beta_0 + \beta_1 \hat{D}_i + \beta_2' X_i + u_i$$

The 2SLS estimate of $$\beta_1$$ is consistent for the local average treatment effect (LATE)—the causal effect for compliers whose treatment status is affected by the instrument.

Weak instruments (small first-stage F-statistic) lead to biased and inconsistent estimates. F-statistics above 10 are considered adequate, though higher thresholds (F > 100) are recommended for robust inference (Stock et al., 2002).

### Regression Discontinuity Designs

Regression discontinuity (RD) designs exploit discontinuous changes in treatment assignment based on a continuous running variable (Lee and Lemieux, 2010). Near the threshold, treatment assignment is quasi-random, enabling causal inference.

For social determinants, examples include:

- **Income thresholds**: Eligibility for programs like SNAP, Medicaid, or housing assistance based on income cutoffs
- **Age discontinuities**: Medicare eligibility at age 65 creates discontinuous change in insurance coverage
- **Geographic boundaries**: School district boundaries, county lines, or other geographic discontinuities that affect resource access

The RD estimate compares outcomes just above versus just below the threshold:

$$\tau_{RD} = \lim_{x \downarrow c} E[Y\lvert X=x] - \lim_{x \uparrow c} E[Y \rvert X=x]$$

where $$c $$ is the cutoff value of running variable $$ X$$.

Estimation typically uses local linear regression within a bandwidth around the cutoff:

$$Y_i = \alpha + \tau D_i + \beta(X_i - c) + \gamma D_i(X_i - c) + \epsilon_i$$

where $$D_i $$ indicates above-threshold status. The estimate $$\hat{\tau}$$ identifies the causal effect at the threshold under continuity assumptions.

Validity requires that other factors do not change discontinuously at the threshold and that individuals cannot precisely manipulate their position relative to the threshold. Falsification tests examine whether baseline covariates show discontinuities (they should not) and density tests check for bunching at the threshold.

### Difference-in-Differences

Difference-in-differences (DiD) designs compare changes over time in treated versus control groups, differencing out time-invariant confounders and common temporal trends (Angrist and Pischke, 2009).

For policy evaluations affecting social determinants:

$$Y_{it} = \alpha + \beta \text{Treat}_i + \gamma \text{Post}_t + \delta(\text{Treat}_i \times \text{Post}_t) + \epsilon_{it}$$

where $$\text{Treat}_i $$ indicates treatment group, $$\text{Post}_t $$ indicates post-policy period, and $$\delta $$ is the DiD estimate.

The key assumption is parallel trends—treated and control groups would have followed parallel trajectories absent treatment. This is untestable but can be evaluated by:

1. Comparing pre-treatment trends (should be parallel)
2. Event study designs showing effects emerge at treatment timing
3. Falsification tests using outcomes unaffected by treatment

Examples include:

- **Medicaid expansions**: States expanding versus not expanding Medicaid under the ACA enable DiD estimates of coverage effects on health outcomes
- **Minimum wage policies**: State-level minimum wage changes allow comparison of health outcomes in implementing versus non-implementing states
- **Housing policies**: Cities implementing inclusionary zoning or rent control enable assessment of housing stability effects

Recent methodological advances address staggered treatment adoption and heterogeneous treatment effects across time periods (Callaway and Sant'Anna, 2021; Sun and Abraham, 2021).

## Implementation: Social Determinants Integration System

We now present a comprehensive implementation of a social determinants integration system for healthcare AI. The system demonstrates external data linkage, individual screening, composite vulnerability measurement, and causal inference methods.

```python
"""
Social Determinants of Health Integration System

A comprehensive system for integrating SDOH data from multiple sources,
implementing screening workflows, constructing composite vulnerability measures,
and conducting causal inference analyses.

Author: Healthcare AI Development Team
License: MIT
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import warnings
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, mean_squared_error
import networkx as nx
from statsmodels.formula.api import ols, logit
from statsmodels.regression.linear_model import IV2SLS
from statsmodels.regression.quantile_regression import QuantReg
import geopandas as gpd
from shapely.geometry import Point
import recordlinkage
from recordlinkage.preprocessing import phonetic
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SDOHDomain(Enum):
    """Enumeration of SDOH domains."""
    ECONOMIC_STABILITY = "economic_stability"
    EDUCATION = "education"
    HEALTHCARE_ACCESS = "healthcare_access"
    NEIGHBORHOOD = "neighborhood"
    SOCIAL_CONTEXT = "social_context"
    FOOD = "food"
    HOUSING = "housing"
    ENVIRONMENT = "environment"
    SAFETY = "safety"

@dataclass
class Patient:
    """Represents a patient with clinical and social determinants data."""
    patient_id: str
    date_of_birth: datetime
    first_name: str
    last_name: str
    address: str
    city: str
    state: str
    zip_code: str
    phone: Optional[str] = None
    email: Optional[str] = None
    race: Optional[str] = None
    ethnicity: Optional[str] = None
    language: Optional[str] = None
    clinical_data: Dict[str, Any] = field(default_factory=dict)
    sdoh_screening: Dict[str, Any] = field(default_factory=dict)
    external_linkages: Dict[str, Any] = field(default_factory=dict)
    composite_scores: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate patient data after initialization."""
        if not self.patient_id:
            raise ValueError("Patient ID is required")
        if not isinstance(self.date_of_birth, datetime):
            raise ValueError("Date of birth must be datetime object")

class PrivacyPreservingRecordLinkage:
    """
    Implements privacy-preserving record linkage using Bloom filters
    and cryptographic hashing for secure data integration.
    """

    def __init__(self, hash_length: int = 1024, num_hash_functions: int = 30,
                 ngram_length: int = 2):
        """
        Initialize privacy-preserving record linkage system.

        Args:
            hash_length: Length of Bloom filter bit array
            num_hash_functions: Number of hash functions for Bloom filter
            ngram_length: Length of character n-grams for string encoding
        """
        self.hash_length = hash_length
        self.num_hash_functions = num_hash_functions
        self.ngram_length = ngram_length
        logger.info(
            f"Initialized PPRL with hash_length={hash_length}, "
            f"num_hash={num_hash_functions}"
        )

    def create_bloom_filter(self, identifiers: Dict[str, str]) -> np.ndarray:
        """
        Create Bloom filter encoding of patient identifiers.

        Args:
            identifiers: Dictionary of identifier fields (name, DOB, address)

        Returns:
            Binary array representing Bloom filter
        """
        bloom_filter = np.zeros(self.hash_length, dtype=int)

        # Concatenate and normalize identifiers
        id_string = "_".join([
            str(v).lower().strip()
            for v in identifiers.values() if v
        ])

        # Generate n-grams
        ngrams = self._generate_ngrams(id_string)

        # Hash each n-gram with multiple hash functions
        for ngram in ngrams:
            for i in range(self.num_hash_functions):
                # Use different salt for each hash function
                hash_input = f"{ngram}_{i}".encode()
                hash_value = int(hashlib.sha256(hash_input).hexdigest(), 16)
                position = hash_value % self.hash_length
                bloom_filter[position] = 1

        return bloom_filter

    def _generate_ngrams(self, text: str) -> List[str]:
        """Generate character n-grams from text."""
        if len(text) < self.ngram_length:
            return [text]
        return [
            text[i:i+self.ngram_length]
            for i in range(len(text) - self.ngram_length + 1)
        ]

    def compute_similarity(self, bloom1: np.ndarray,
                          bloom2: np.ndarray) -> float:
        """
        Compute Dice coefficient similarity between Bloom filters.

        Args:
            bloom1: First Bloom filter
            bloom2: Second Bloom filter

        Returns:
            Similarity score between 0 and 1
        """
        intersection = np.sum(bloom1 & bloom2)
        union = np.sum(bloom1) + np.sum(bloom2)

        if union == 0:
            return 0.0

        # Dice coefficient
        return 2 * intersection / union

    def link_records(self, source1: pd.DataFrame, source2: pd.DataFrame,
                    id_fields: List[str], threshold: float = 0.85) -> pd.DataFrame:
        """
        Link records between two data sources using privacy-preserving methods.

        Args:
            source1: First data source with identifier fields
            source2: Second data source with identifier fields
            id_fields: List of identifier field names to use for matching
            threshold: Similarity threshold for accepting matches

        Returns:
            DataFrame with matched record pairs and similarity scores
        """
        logger.info(
            f"Linking {len(source1)} records from source1 with "
            f"{len(source2)} records from source2"
        )

        # Create Bloom filters for all records
        bloom1 = np.array([
            self.create_bloom_filter(row[id_fields].to_dict())
            for _, row in source1.iterrows()
        ])

        bloom2 = np.array([
            self.create_bloom_filter(row[id_fields].to_dict())
            for _, row in source2.iterrows()
        ])

        # Compute pairwise similarities
        matches = []
        for i, bf1 in enumerate(bloom1):
            similarities = np.array([
                self.compute_similarity(bf1, bf2) for bf2 in bloom2
            ])

            # Find matches above threshold
            match_indices = np.where(similarities >= threshold)[0]

            for j in match_indices:
                matches.append({
                    'source1_idx': i,
                    'source2_idx': j,
                    'similarity': similarities[j]
                })

        matches_df = pd.DataFrame(matches)
        logger.info(f"Found {len(matches_df)} potential matches above threshold")

        return matches_df

class ExternalDataIntegration:
    """
    Integrates clinical data with external sources on housing, food access,
    environmental exposures, and neighborhood characteristics.
    """

    def __init__(self, geocoder_api_key: Optional[str] = None):
        """
        Initialize external data integration system.

        Args:
            geocoder_api_key: API key for geocoding services
        """
        self.geocoder_api_key = geocoder_api_key
        self.census_data: Optional[pd.DataFrame] = None
        self.environmental_data: Optional[gpd.GeoDataFrame] = None
        logger.info("Initialized external data integration system")

    def geocode_addresses(self, patients: List[Patient]) -> gpd.GeoDataFrame:
        """
        Geocode patient addresses to geographic coordinates.

        Args:
            patients: List of patient objects with addresses

        Returns:
            GeoDataFrame with patient locations
        """
        # In production, would use geocoding API (Google, Census, etc.)
        # This is a simplified implementation
        records = []
        for patient in patients:
            # Mock geocoding - would call actual API
            lat = 37.7749 + np.random.randn() * 0.1  # Mock coordinates
            lon = -122.4194 + np.random.randn() * 0.1

            records.append({
                'patient_id': patient.patient_id,
                'address': patient.address,
                'latitude': lat,
                'longitude': lon,
                'geometry': Point(lon, lat)
            })

        gdf = gpd.GeoDataFrame(records, crs='EPSG:4326')
        logger.info(f"Geocoded {len(gdf)} patient addresses")
        return gdf

    def link_census_data(self, patient_gdf: gpd.GeoDataFrame,
                        census_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Link patient locations to census tract-level data.

        Args:
            patient_gdf: GeoDataFrame of patient locations
            census_gdf: GeoDataFrame of census tracts with attributes

        Returns:
            DataFrame with census data joined to patients
        """
        # Spatial join to assign census tract to each patient
        joined = gpd.sjoin(
            patient_gdf,
            census_gdf,
            how='left',
            predicate='within'
        )

        logger.info(
            f"Linked {len(joined)} patients to census tracts, "
            f"{joined['index_right'].isna().sum()} without matches"
        )

        return pd.DataFrame(joined)

    def calculate_food_access_metrics(
        self,
        patient_gdf: gpd.GeoDataFrame,
        retailer_gdf: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Calculate food access metrics based on distance to retailers.

        Args:
            patient_gdf: GeoDataFrame of patient locations
            retailer_gdf: GeoDataFrame of food retailers (supermarkets, etc.)

        Returns:
            DataFrame with food access metrics per patient
        """
        metrics = []

        for _, patient in patient_gdf.iterrows():
            patient_point = patient.geometry

            # Calculate distances to all retailers
            distances = retailer_gdf.geometry.distance(patient_point)

            # Compute metrics
            nearest_supermarket = distances.min()
            supermarkets_1mile = (distances <= 0.016).sum()  # ~1 mile in degrees
            supermarkets_3miles = (distances <= 0.048).sum()

            metrics.append({
                'patient_id': patient['patient_id'],
                'distance_nearest_supermarket_miles': nearest_supermarket * 69,  # Approx miles
                'supermarkets_within_1_mile': supermarkets_1mile,
                'supermarkets_within_3_miles': supermarkets_3miles,
                'food_desert': nearest_supermarket > 0.016 and supermarkets_1mile == 0
            })

        metrics_df = pd.DataFrame(metrics)
        logger.info(
            f"Calculated food access for {len(metrics_df)} patients, "
            f"{metrics_df['food_desert'].sum()} in food deserts"
        )

        return metrics_df

    def link_environmental_exposures(
        self,
        patient_gdf: gpd.GeoDataFrame,
        pollution_data: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Link patient locations to environmental exposure data.

        Args:
            patient_gdf: GeoDataFrame of patient locations
            pollution_data: GeoDataFrame of pollution monitoring data

        Returns:
            DataFrame with exposure estimates per patient
        """
        exposures = []

        for _, patient in patient_gdf.iterrows():
            patient_point = patient.geometry

            # Find nearest monitoring station
            distances = pollution_data.geometry.distance(patient_point)
            nearest_idx = distances.idxmin()
            nearest_station = pollution_data.loc[nearest_idx]

            # Use inverse distance weighting for interpolation
            nearby = pollution_data[distances <= 0.1]  # Within ~7 miles

            if len(nearby) >= 3:
                weights = 1 / (nearby.geometry.distance(patient_point) + 0.001)
                weights = weights / weights.sum()

                pm25 = (nearby['pm25'] * weights).sum()
                ozone = (nearby['ozone'] * weights).sum()
            else:
                pm25 = nearest_station['pm25']
                ozone = nearest_station['ozone']

            exposures.append({
                'patient_id': patient['patient_id'],
                'pm25_ugm3': pm25,
                'ozone_ppm': ozone,
                'nearest_monitor_distance_miles': distances.min() * 69,
                'high_pollution_exposure': pm25 > 12.0  # EPA standard
            })

        exposures_df = pd.DataFrame(exposures)
        logger.info(
            f"Estimated environmental exposures for {len(exposures_df)} patients"
        )

        return exposures_df

class SDOHScreening:
    """
    Implements validated SDOH screening instruments and manages
    screening workflow integration.
    """

    def __init__(self, instrument: str = "PRAPARE"):
        """
        Initialize SDOH screening system.

        Args:
            instrument: Screening instrument to use (PRAPARE, Health Leads, etc.)
        """
        self.instrument = instrument
        self.questions = self._load_questions()
        logger.info(f"Initialized {instrument} screening instrument")

    def _load_questions(self) -> Dict[str, Dict[str, Any]]:
        """Load screening questions for selected instrument."""
        if self.instrument == "PRAPARE":
            return {
                'income': {
                    'text': 'What is your total household income in the last year?',
                    'type': 'categorical',
                    'options': [
                        '<$ 10,000', '$10,000-$ 20,000', '$20,000-$ 30,000',
                        '$30,000-$ 50,000', '$50,000-$ 75,000', '>$75,000',
                        'Prefer not to answer'
                    ],
                    'domain': SDOHDomain.ECONOMIC_STABILITY
                },
                'employment': {
                    'text': 'What is your current work situation?',
                    'type': 'categorical',
                    'options': [
                        'Employed full-time', 'Employed part-time',
                        'Unemployed seeking work', 'Unemployed not seeking work',
                        'Retired', 'Disabled', 'Student',
                        'Prefer not to answer'
                    ],
                    'domain': SDOHDomain.ECONOMIC_STABILITY
                },
                'education': {
                    'text': 'What is the highest level of school you have finished?',
                    'type': 'categorical',
                    'options': [
                        'Less than high school', 'High school or GED',
                        'Some college', 'Associate degree',
                        'Bachelor degree', 'Graduate degree',
                        'Prefer not to answer'
                    ],
                    'domain': SDOHDomain.EDUCATION
                },
                'housing_stability': {
                    'text': 'In the past 12 months, have you been homeless or stayed in a shelter?',
                    'type': 'binary',
                    'domain': SDOHDomain.HOUSING
                },
                'housing_quality': {
                    'text': 'What are the housing problems you experience? (check all that apply)',
                    'type': 'multiple',
                    'options': [
                        'Mold', 'Pests', 'Lead paint', 'Inadequate heating',
                        'Overcrowding', 'Water leaks', 'None'
                    ],
                    'domain': SDOHDomain.HOUSING
                },
                'food_insecurity': {
                    'text': 'In the past 12 months, worried whether food would run out before got money to buy more',
                    'type': 'likert',
                    'options': ['Often true', 'Sometimes true', 'Never true'],
                    'domain': SDOHDomain.FOOD
                },
                'transportation': {
                    'text': 'Do you have reliable transportation to medical appointments?',
                    'type': 'binary',
                    'domain': SDOHDomain.HEALTHCARE_ACCESS
                },
                'utilities': {
                    'text': 'In the past 12 months, has the electric, gas, or water company threatened to shut off services?',
                    'type': 'binary',
                    'domain': SDOHDomain.ECONOMIC_STABILITY
                },
                'safety': {
                    'text': 'Do you feel safe in your home and neighborhood?',
                    'type': 'likert',
                    'options': ['Very safe', 'Somewhat safe', 'Not very safe', 'Not at all safe'],
                    'domain': SDOHDomain.SAFETY
                },
                'social_isolation': {
                    'text': 'How often do you see or talk to people you care about?',
                    'type': 'likert',
                    'options': [
                        'Daily', 'Several times a week', 'Once a week',
                        'Once a month', 'Less than once a month'
                    ],
                    'domain': SDOHDomain.SOCIAL_CONTEXT
                }
            }
        else:
            raise ValueError(f"Instrument {self.instrument} not implemented")

    def administer_screening(self, patient: Patient,
                           responses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record screening responses for a patient.

        Args:
            patient: Patient object
            responses: Dictionary of question_id: response

        Returns:
            Dictionary of processed screening results
        """
        results = {
            'screening_date': datetime.now(),
            'instrument': self.instrument,
            'responses': responses,
            'flags': {}
        }

        # Identify concerning responses
        if responses.get('housing_stability') == True:
            results['flags']['housing_unstable'] = True

        if responses.get('food_insecurity') in ['Often true', 'Sometimes true']:
            results['flags']['food_insecure'] = True

        if responses.get('transportation') == False:
            results['flags']['transportation_barrier'] = True

        if responses.get('utilities') == True:
            results['flags']['utilities_insecurity'] = True

        if responses.get('safety') in ['Not very safe', 'Not at all safe']:
            results['flags']['safety_concern'] = True

        # Calculate domain scores
        results['domain_scores'] = self._calculate_domain_scores(responses)

        logger.info(
            f"Completed screening for patient {patient.patient_id}, "
            f"{len(results['flags'])} concerns identified"
        )

        return results

    def _calculate_domain_scores(self, responses: Dict[str, Any]) -> Dict[str, float]:
        """Calculate vulnerability scores by SDOH domain."""
        domain_scores = {}

        # Group questions by domain
        domains = {}
        for q_id, q_data in self.questions.items():
            domain = q_data['domain']
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(q_id)

        # Score each domain (simplified - would use validated scoring)
        for domain, questions in domains.items():
            responses_in_domain = [
                responses.get(q) for q in questions if q in responses
            ]
            # Mock scoring - would use validated methods
            domain_scores[domain.value] = np.random.random()

        return domain_scores

    def handle_missing_data(self, screening_data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing screening responses using multiple imputation.

        Args:
            screening_data: DataFrame of screening responses with missingness

        Returns:
            DataFrame with imputed values
        """
        # Identify columns with missing data
        missing_cols = screening_data.columns[screening_data.isna().any()].tolist()

        if not missing_cols:
            logger.info("No missing data to impute")
            return screening_data

        logger.info(
            f"Imputing {len(missing_cols)} columns with missing data "
            f"({screening_data[missing_cols].isna().sum().sum()} total missing)"
        )

        # Use iterative imputer (MICE algorithm)
        imputer = IterativeImputer(
            max_iter=10,
            random_state=42,
            initial_strategy='median'
        )

        # Impute numeric columns only
        numeric_cols = screening_data.select_dtypes(include=[np.number]).columns
        missing_numeric = [c for c in missing_cols if c in numeric_cols]

        if missing_numeric:
            screening_data[missing_numeric] = imputer.fit_transform(
                screening_data[missing_numeric]
            )

        # For categorical, use mode imputation
        categorical_cols = [c for c in missing_cols if c not in numeric_cols]
        for col in categorical_cols:
            mode = screening_data[col].mode()[0] if not screening_data[col].mode().empty else None
            if mode is not None:
                screening_data[col].fillna(mode, inplace=True)

        logger.info("Completed imputation")
        return screening_data

class CompositeVulnerabilityIndex:
    """
    Constructs composite measures of social vulnerability from
    multiple SDOH indicators.
    """

    def __init__(self, method: str = "pca"):
        """
        Initialize composite index construction.

        Args:
            method: Method for constructing index (pca, factor_analysis, weighted_sum)
        """
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        logger.info(f"Initialized composite vulnerability index with method={method}")

    def fit(self, sdoh_data: pd.DataFrame,
           indicator_cols: List[str]) -> 'CompositeVulnerabilityIndex':
        """
        Fit composite index model to SDOH data.

        Args:
            sdoh_data: DataFrame of SDOH indicators
            indicator_cols: List of column names to include in index

        Returns:
            Self for method chaining
        """
        X = sdoh_data[indicator_cols].values

        # Standardize indicators
        X_scaled = self.scaler.fit_transform(X)

        if self.method == "pca":
            self.model = PCA(n_components=1)
            self.model.fit(X_scaled)
            logger.info(
                f"PCA explained variance: {self.model.explained_variance_ratio_[0]:.3f}"
            )

        elif self.method == "factor_analysis":
            self.model = FactorAnalysis(n_components=1, random_state=42)
            self.model.fit(X_scaled)
            logger.info("Fitted factor analysis model")

        elif self.method == "weighted_sum":
            # Use correlation with outcome as weights (if available)
            # For now, equal weights
            self.model = np.ones(len(indicator_cols)) / len(indicator_cols)
            logger.info("Using equal-weighted sum")

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def transform(self, sdoh_data: pd.DataFrame,
                 indicator_cols: List[str]) -> np.ndarray:
        """
        Compute composite vulnerability scores.

        Args:
            sdoh_data: DataFrame of SDOH indicators
            indicator_cols: List of column names to include

        Returns:
            Array of vulnerability scores
        """
        X = sdoh_data[indicator_cols].values
        X_scaled = self.scaler.transform(X)

        if self.method in ["pca", "factor_analysis"]:
            scores = self.model.transform(X_scaled).flatten()
        else:  # weighted_sum
            scores = (X_scaled * self.model).sum(axis=1)

        # Convert to percentile ranks (0-100)
        percentiles = stats.rankdata(scores, method='average') / len(scores) * 100

        logger.info(
            f"Computed vulnerability scores, mean={percentiles.mean():.1f}, "
            f"std={percentiles.std():.1f}"
        )

        return percentiles

    def fit_transform(self, sdoh_data: pd.DataFrame,
                     indicator_cols: List[str]) -> np.ndarray:
        """Fit model and compute scores in one step."""
        self.fit(sdoh_data, indicator_cols)
        return self.transform(sdoh_data, indicator_cols)

    def get_component_loadings(self, indicator_names: List[str]) -> pd.DataFrame:
        """
        Get loadings of each indicator on the composite index.

        Args:
            indicator_names: Names of indicators

        Returns:
            DataFrame with loadings
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        if self.method == "pca":
            loadings = self.model.components_[0]
        elif self.method == "factor_analysis":
            loadings = self.model.components_[0]
        else:
            loadings = self.model

        loadings_df = pd.DataFrame({
            'indicator': indicator_names,
            'loading': loadings
        }).sort_values('loading', key=abs, ascending=False)

        return loadings_df

class CausalInferenceSDOH:
    """
    Implements causal inference methods for understanding SDOH effects,
    including mediation analysis, structural equation modeling, and
    instrumental variable methods.
    """

    def __init__(self):
        """Initialize causal inference system."""
        logger.info("Initialized causal inference system for SDOH")

    def mediation_analysis(
        self,
        data: pd.DataFrame,
        treatment: str,
        mediator: str,
        outcome: str,
        covariates: List[str],
        n_bootstrap: int = 1000
    ) -> Dict[str, float]:
        """
        Conduct mediation analysis to decompose total effects into
        direct and indirect (mediated) effects.

        Args:
            data: DataFrame containing all variables
            treatment: Name of treatment/exposure variable
            mediator: Name of mediating variable
            outcome: Name of outcome variable
            covariates: List of covariate names to adjust for
            n_bootstrap: Number of bootstrap samples for confidence intervals

        Returns:
            Dictionary of effect estimates and confidence intervals
        """
        logger.info(
            f"Conducting mediation analysis: {treatment} -> {mediator} -> {outcome}"
        )

        # Model 1: Treatment -> Mediator
        mediator_formula = f"{mediator} ~ {treatment}"
        if covariates:
            mediator_formula += " + " + " + ".join(covariates)

        mediator_model = ols(mediator_formula, data=data).fit()
        alpha = mediator_model.params[treatment]  # Effect of treatment on mediator

        # Model 2: Treatment + Mediator -> Outcome
        outcome_formula = f"{outcome} ~ {treatment} + {mediator}"
        if covariates:
            outcome_formula += " + " + " + ".join(covariates)

        outcome_model = ols(outcome_formula, data=data).fit()
        beta = outcome_model.params[mediator]  # Effect of mediator on outcome
        tau_prime = outcome_model.params[treatment]  # Direct effect

        # Calculate effects
        indirect_effect = alpha * beta  # Mediated effect
        total_effect = tau_prime + indirect_effect
        proportion_mediated = indirect_effect / total_effect if total_effect != 0 else 0

        # Bootstrap confidence intervals
        indirect_effects = []
        for _ in range(n_bootstrap):
            boot_data = data.sample(n=len(data), replace=True)

            try:
                m_model = ols(mediator_formula, data=boot_data).fit()
                o_model = ols(outcome_formula, data=boot_data).fit()

                boot_alpha = m_model.params[treatment]
                boot_beta = o_model.params[mediator]
                indirect_effects.append(boot_alpha * boot_beta)
            except:
                continue

        indirect_effects = np.array(indirect_effects)
        ci_lower, ci_upper = np.percentile(indirect_effects, [2.5, 97.5])

        results = {
            'total_effect': total_effect,
            'direct_effect': tau_prime,
            'indirect_effect': indirect_effect,
            'proportion_mediated': proportion_mediated,
            'indirect_effect_ci_lower': ci_lower,
            'indirect_effect_ci_upper': ci_upper
        }

        logger.info(
            f"Mediation results: indirect effect = {indirect_effect:.3f} "
            f"({ci_lower:.3f}, {ci_upper:.3f}), "
            f"proportion mediated = {proportion_mediated:.3f}"
        )

        return results

    def instrumental_variable_analysis(
        self,
        data: pd.DataFrame,
        treatment: str,
        instrument: str,
        outcome: str,
        covariates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Estimate causal effects using instrumental variable methods (2SLS).

        Args:
            data: DataFrame containing all variables
            treatment: Name of treatment variable
            instrument: Name of instrumental variable
            outcome: Name of outcome variable
            covariates: Optional list of covariate names

        Returns:
            Dictionary of IV estimates and diagnostics
        """
        logger.info(
            f"Conducting IV analysis with instrument {instrument} "
            f"for treatment {treatment}"
        )

        # First stage: Regress treatment on instrument and covariates
        exog_formula = instrument
        if covariates:
            exog_formula += " + " + " + ".join(covariates)

        # Using statsmodels IV2SLS
        iv_model = IV2SLS.from_formula(
            f"{outcome} ~ [{treatment} ~ {exog_formula}]",
            data=data
        ).fit()

        # Calculate first stage F-statistic
        first_stage_formula = f"{treatment} ~ {exog_formula}"
        first_stage = ols(first_stage_formula, data=data).fit()
        f_stat = first_stage.fvalue

        results = {
            'iv_estimate': iv_model.params[treatment],
            'std_error': iv_model.bse[treatment],
            'ci_lower': iv_model.conf_int().loc[treatment, 0],
            'ci_upper': iv_model.conf_int().loc[treatment, 1],
            'first_stage_f_stat': f_stat,
            'weak_instrument': f_stat < 10,  # Rule of thumb
            'iv_model': iv_model
        }

        if results['weak_instrument']:
            warnings.warn(
                f"Weak instrument detected (F={f_stat:.2f}). "
                "Results may be unreliable."
            )

        logger.info(
            f"IV estimate: {results['iv_estimate']:.3f} "
            f"({results['ci_lower']:.3f}, {results['ci_upper']:.3f}), "
            f"First stage F={f_stat:.2f}"
        )

        return results

    def difference_in_differences(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        time: str,
        unit_id: str,
        covariates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Estimate treatment effects using difference-in-differences design.

        Args:
            data: Panel data with units observed before and after treatment
            outcome: Name of outcome variable
            treatment: Binary indicator of treatment group
            time: Binary indicator of post-treatment period
            unit_id: Variable identifying units
            covariates: Optional time-varying covariates

        Returns:
            Dictionary of DiD estimates
        """
        logger.info("Conducting difference-in-differences analysis")

        # Create interaction term
        data['treat_x_post'] = data[treatment] * data[time]

        # DiD regression
        formula = f"{outcome} ~ {treatment} + {time} + treat_x_post"
        if covariates:
            formula += " + " + " + ".join(covariates)

        # Add unit fixed effects (convert unit_id to categorical)
        formula += f" + C({unit_id})"

        did_model = ols(formula, data=data).fit(
            cov_type='cluster',
            cov_kwds={'groups': data[unit_id]}  # Cluster standard errors
        )

        results = {
            'did_estimate': did_model.params['treat_x_post'],
            'std_error': did_model.bse['treat_x_post'],
            'ci_lower': did_model.conf_int().loc['treat_x_post', 0],
            'ci_upper': did_model.conf_int().loc['treat_x_post', 1],
            'pvalue': did_model.pvalues['treat_x_post'],
            'model': did_model
        }

        logger.info(
            f"DiD estimate: {results['did_estimate']:.3f} "
            f"({results['ci_lower']:.3f}, {results['ci_upper']:.3f}), "
            f"p={results['pvalue']:.4f}"
        )

        return results

    def test_parallel_trends(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        time: str,
        pre_periods: List[int]
    ) -> Dict[str, Any]:
        """
        Test parallel trends assumption for DiD by examining pre-treatment trends.

        Args:
            data: Panel data
            outcome: Outcome variable
            treatment: Treatment indicator
            time: Time period variable
            pre_periods: List of pre-treatment time periods to test

        Returns:
            Dictionary of test results
        """
        logger.info("Testing parallel trends assumption")

        # Create leads for each pre-period
        results = {}
        for period in pre_periods:
            period_data = data[data[time] == period].copy()
            period_data['lead'] = period_data[treatment]

            formula = f"{outcome} ~ lead + C({time})"
            model = ols(formula, data=period_data).fit()

            results[f'period_{period}'] = {
                'coefficient': model.params.get('lead', np.nan),
                'pvalue': model.pvalues.get('lead', np.nan)
            }

        # Overall test: any significant pre-trends?
        pvalues = [r['pvalue'] for r in results.values() if not np.isnan(r['pvalue'])]
        min_pvalue = min(pvalues) if pvalues else 1.0

        results['parallel_trends_violated'] = min_pvalue < 0.05

        if results['parallel_trends_violated']:
            warnings.warn(
                "Parallel trends assumption may be violated. "
                "DiD estimates may be biased."
            )

        return results

class SDOHRiskStratification:
    """
    Implements risk stratification and intervention targeting
    based on integrated SDOH data.
    """

    def __init__(self):
        """Initialize risk stratification system."""
        self.risk_model = None
        logger.info("Initialized SDOH risk stratification system")

    def fit_risk_model(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model_type: str = "gradient_boosting",
        stratify_by: Optional[List[str]] = None
    ) -> 'SDOHRiskStratification':
        """
        Train risk prediction model incorporating SDOH features.

        Args:
            X: DataFrame of features including SDOH variables
            y: Binary outcome array
            model_type: Type of model (gradient_boosting, random_forest, logistic)
            stratify_by: Optional demographic variables for stratified evaluation

        Returns:
            Self for method chaining
        """
        logger.info(f"Training {model_type} risk model with {X.shape[1]} features")

        if model_type == "gradient_boosting":
            self.risk_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            )
        elif model_type == "random_forest":
            self.risk_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == "logistic":
            self.risk_model = LogisticRegression(
                penalty='l2',
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.risk_model.fit(X, y)

        # Evaluate by demographic strata if specified
        if stratify_by:
            self._evaluate_by_strata(X, y, stratify_by)

        return self

    def _evaluate_by_strata(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        stratify_by: List[str]
    ) -> None:
        """Evaluate model performance across demographic strata."""
        for var in stratify_by:
            if var not in X.columns:
                continue

            unique_values = X[var].unique()

            logger.info(f"Evaluating performance stratified by {var}")
            for value in unique_values:
                mask = X[var] == value
                if mask.sum() < 10:
                    continue

                X_strata = X[mask]
                y_strata = y[mask]

                if hasattr(self.risk_model, 'predict_proba'):
                    y_pred_proba = self.risk_model.predict_proba(X_strata)[:, 1]
                    auc = roc_auc_score(y_strata, y_pred_proba)
                    logger.info(f"  {var}={value}: AUC={auc:.3f}, n={mask.sum()}")
                else:
                    y_pred = self.risk_model.predict(X_strata)
                    rmse = np.sqrt(mean_squared_error(y_strata, y_pred))
                    logger.info(f"  {var}={value}: RMSE={rmse:.3f}, n={mask.sum()}")

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk scores for new patients.

        Args:
            X: DataFrame of features

        Returns:
            Array of risk scores
        """
        if self.risk_model is None:
            raise ValueError("Model not fitted")

        if hasattr(self.risk_model, 'predict_proba'):
            return self.risk_model.predict_proba(X)[:, 1]
        else:
            return self.risk_model.predict(X)

    def identify_high_need_patients(
        self,
        patients_df: pd.DataFrame,
        risk_scores: np.ndarray,
        risk_threshold: float = 0.8,
        sdoh_threshold: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Identify patients with high clinical risk AND high social needs
        for intensive intervention.

        Args:
            patients_df: DataFrame of patient data
            risk_scores: Clinical risk scores
            risk_threshold: Threshold for defining high risk
            sdoh_threshold: Optional thresholds for specific SDOH needs

        Returns:
            DataFrame of high-need patients with intervention recommendations
        """
        high_risk_mask = risk_scores >= risk_threshold

        # Identify patients with multiple SDOH needs
        sdoh_needs = pd.DataFrame()

        if sdoh_threshold:
            for need, threshold in sdoh_threshold.items():
                if need in patients_df.columns:
                    sdoh_needs[need] = patients_df[need] >= threshold

        sdoh_needs['total_needs'] = sdoh_needs.sum(axis=1)
        high_need_mask = sdoh_needs['total_needs'] >= 2

        # Combined high clinical risk and high social needs
        target_patients = patients_df[high_risk_mask & high_need_mask].copy()
        target_patients['risk_score'] = risk_scores[high_risk_mask & high_need_mask]
        target_patients['sdoh_needs_count'] = sdoh_needs.loc[
            high_risk_mask & high_need_mask, 'total_needs'
        ]

        logger.info(
            f"Identified {len(target_patients)} high-need patients "
            f"({len(target_patients)/len(patients_df)*100:.1f}% of total)"
        )

        return target_patients

    def generate_intervention_plan(
        self,
        patient_data: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Generate personalized intervention recommendations based on
        identified SDOH needs.

        Args:
            patient_data: Dictionary of patient SDOH data

        Returns:
            Dictionary mapping needs to recommended interventions
        """
        interventions = {}

        # Housing interventions
        if patient_data.get('housing_unstable'):
            interventions['housing'] = [
                'Referral to housing authority for emergency housing',
                'Application assistance for housing voucher programs',
                'Connection to legal aid for eviction prevention',
                'Referral to transitional housing programs'
            ]

        # Food security interventions
        if patient_data.get('food_insecure'):
            interventions['food'] = [
                'SNAP enrollment assistance',
                'Referral to local food pantries',
                'Connection to meal delivery programs',
                'WIC program enrollment (if eligible)'
            ]

        # Transportation interventions
        if patient_data.get('transportation_barrier'):
            interventions['transportation'] = [
                'Arrange medical transportation for appointments',
                'Application for reduced-fare transit cards',
                'Telehealth visit setup for non-urgent care',
                'Referral to volunteer driver programs'
            ]

        # Utilities interventions
        if patient_data.get('utilities_insecurity'):
            interventions['utilities'] = [
                'Application for utility assistance programs (LIHEAP)',
                'Energy weatherization program referral',
                'Payment plan arrangement with utilities',
                'Emergency financial assistance'
            ]

        # Safety interventions
        if patient_data.get('safety_concern'):
            interventions['safety'] = [
                'Referral to domestic violence services',
                'Connection to community violence intervention',
                'Safety planning with social worker',
                'Legal advocacy services'
            ]

        # Social support interventions
        if patient_data.get('social_isolation'):
            interventions['social_support'] = [
                'Referral to senior centers or community programs',
                'Connection to peer support groups',
                'Volunteer visitor program enrollment',
                'Mental health services referral'
            ]

        return interventions

def example_workflow() -> None:
    """
    Demonstrate complete SDOH integration workflow from data linkage
    through risk stratification and intervention targeting.
    """
    logger.info("=" * 80)
    logger.info("SOCIAL DETERMINANTS OF HEALTH INTEGRATION - EXAMPLE WORKFLOW")
    logger.info("=" * 80)

    # Generate synthetic patient data
    np.random.seed(42)
    n_patients = 1000

    patients = []
    for i in range(n_patients):
        patient = Patient(
            patient_id=f"PAT{i:06d}",
            date_of_birth=datetime(1960, 1, 1) + timedelta(days=np.random.randint(0, 20000)),
            first_name=f"FirstName{i}",
            last_name=f"LastName{i}",
            address=f"{np.random.randint(1, 9999)} Main St",
            city="San Francisco",
            state="CA",
            zip_code=f"94{np.random.randint(100, 200):03d}",
            race=np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other']),
            ethnicity=np.random.choice(['Hispanic', 'Non-Hispanic']),
            language=np.random.choice(['English', 'Spanish', 'Chinese', 'Other'])
        )
        patients.append(patient)

    # 1. External Data Integration
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: EXTERNAL DATA INTEGRATION")
    logger.info("=" * 80)

    external_integration = ExternalDataIntegration()

    # Geocode addresses
    patient_gdf = external_integration.geocode_addresses(patients)

    # Link census data (mock)
    census_data = pd.DataFrame({
        'tract_id': range(100),
        'median_income': np.random.uniform(30000, 150000, 100),
        'poverty_rate': np.random.uniform(0.05, 0.40, 100),
        'unemployment_rate': np.random.uniform(0.03, 0.15, 100),
        'pct_no_hs_diploma': np.random.uniform(0.05, 0.30, 100),
        'geometry': [Point(-122.4 + i*0.01, 37.7 + i*0.01) for i in range(100)]
    })
    census_gdf = gpd.GeoDataFrame(census_data, crs='EPSG:4326')

    patient_census = external_integration.link_census_data(patient_gdf, census_gdf)

    # Calculate food access
    retailers = pd.DataFrame({
        'name': [f"Store{i}" for i in range(50)],
        'type': np.random.choice(['supermarket', 'convenience'], 50),
        'geometry': [Point(-122.4 + np.random.randn()*0.05,
                          37.7 + np.random.randn()*0.05) for _ in range(50)]
    })
    retailers_gdf = gpd.GeoDataFrame(retailers, crs='EPSG:4326')

    food_access = external_integration.calculate_food_access_metrics(
        patient_gdf, retailers_gdf
    )

    # Link environmental data
    pollution_data = pd.DataFrame({
        'monitor_id': range(20),
        'pm25': np.random.uniform(5, 20, 20),
        'ozone': np.random.uniform(0.03, 0.08, 20),
        'geometry': [Point(-122.4 + np.random.randn()*0.1,
                          37.7 + np.random.randn()*0.1) for _ in range(20)]
    })
    pollution_gdf = gpd.GeoDataFrame(pollution_data, crs='EPSG:4326')

    environmental = external_integration.link_environmental_exposures(
        patient_gdf, pollution_gdf
    )

    # 2. Individual-Level Screening
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: INDIVIDUAL-LEVEL SDOH SCREENING")
    logger.info("=" * 80)

    screening = SDOHScreening(instrument="PRAPARE")

    screening_results = []
    for patient in patients[:100]:  # Screen subset
        # Generate mock responses
        responses = {
            'income': np.random.choice(['<$ 10,000', '$10,000-$ 20,000', '$20,000-$ 30,000',
                                       '$30,000-$ 50,000', '$50,000-$ 75,000', '>$75,000']),
            'employment': np.random.choice(['Employed full-time', 'Employed part-time',
                                           'Unemployed seeking work', 'Retired']),
            'education': np.random.choice(['Less than high school', 'High school or GED',
                                          'Some college', 'Bachelor degree']),
            'housing_stability': np.random.choice([True, False], p=[0.15, 0.85]),
            'food_insecurity': np.random.choice(['Often true', 'Sometimes true', 'Never true'],
                                                p=[0.1, 0.2, 0.7]),
            'transportation': np.random.choice([True, False], p=[0.8, 0.2]),
            'utilities': np.random.choice([True, False], p=[0.12, 0.88]),
            'safety': np.random.choice(['Very safe', 'Somewhat safe', 'Not very safe'],
                                       p=[0.6, 0.3, 0.1])
        }

        results = screening.administer_screening(patient, responses)
        results['patient_id'] = patient.patient_id
        screening_results.append(results)

    # 3. Composite Vulnerability Index
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: COMPOSITE VULNERABILITY INDEX")
    logger.info("=" * 80)

    # Combine all SDOH data
    sdoh_combined = patient_census.merge(
        food_access, on='patient_id', how='left'
    ).merge(
        environmental, on='patient_id', how='left'
    )

    # Select vulnerability indicators
    vulnerability_indicators = [
        'poverty_rate', 'unemployment_rate', 'pct_no_hs_diploma',
        'distance_nearest_supermarket_miles', 'pm25_ugm3'
    ]

    # Construct composite index
    vulnerability_index = CompositeVulnerabilityIndex(method="pca")
    vulnerability_scores = vulnerability_index.fit_transform(
        sdoh_combined, vulnerability_indicators
    )

    sdoh_combined['vulnerability_percentile'] = vulnerability_scores

    # Examine component loadings
    loadings = vulnerability_index.get_component_loadings(vulnerability_indicators)
    logger.info("\nVulnerability Index Component Loadings:")
    logger.info("\n" + str(loadings))

    # 4. Causal Inference
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: CAUSAL INFERENCE ANALYSIS")
    logger.info("=" * 80)

    causal = CausalInferenceSDOH()

    # Generate mock outcome data
    sdoh_combined['health_outcome'] = (
        0.5 * sdoh_combined['vulnerability_percentile'] / 100 +
        0.3 * sdoh_combined['pm25_ugm3'] / 20 +
        np.random.normal(0, 0.3, len(sdoh_combined))
    )

    # Mediation analysis: poverty -> food access -> health
    mediation_results = causal.mediation_analysis(
        data=sdoh_combined,
        treatment='poverty_rate',
        mediator='distance_nearest_supermarket_miles',
        outcome='health_outcome',
        covariates=['unemployment_rate'],
        n_bootstrap=500
    )

    logger.info("\nMediation Analysis Results:")
    logger.info(f"  Total effect: {mediation_results['total_effect']:.4f}")
    logger.info(f"  Direct effect: {mediation_results['direct_effect']:.4f}")
    logger.info(f"  Indirect effect: {mediation_results['indirect_effect']:.4f}")
    logger.info(f"  Proportion mediated: {mediation_results['proportion_mediated']:.2%}")

    # 5. Risk Stratification and Intervention
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: RISK STRATIFICATION AND INTERVENTION")
    logger.info("=" * 80)

    risk_stratification = SDOHRiskStratification()

    # Prepare features
    feature_cols = vulnerability_indicators + ['vulnerability_percentile']
    X = sdoh_combined[feature_cols].fillna(0)
    y = (sdoh_combined['health_outcome'] > sdoh_combined['health_outcome'].median()).astype(int)

    # Train risk model
    risk_stratification.fit_risk_model(X, y, model_type="gradient_boosting")

    # Predict risk
    risk_scores = risk_stratification.predict_risk(X)

    # Identify high-need patients
    sdoh_combined['housing_unstable'] = np.random.choice([True, False], len(sdoh_combined), p=[0.15, 0.85])
    sdoh_combined['food_insecure'] = sdoh_combined['distance_nearest_supermarket_miles'] > 2.0
    sdoh_combined['transportation_barrier'] = np.random.choice([True, False], len(sdoh_combined), p=[0.2, 0.8])

    high_need_patients = risk_stratification.identify_high_need_patients(
        sdoh_combined,
        risk_scores,
        risk_threshold=0.7,
        sdoh_threshold={
            'poverty_rate': 0.20,
            'pm25_ugm3': 12.0
        }
    )

    logger.info(f"\nIdentified {len(high_need_patients)} high-need patients")

    # Generate intervention plan for example patient
    if len(high_need_patients) > 0:
        example_patient = high_need_patients.iloc[0]
        intervention_plan = risk_stratification.generate_intervention_plan({
            'housing_unstable': example_patient.get('housing_unstable', False),
            'food_insecure': example_patient.get('food_insecure', False),
            'transportation_barrier': example_patient.get('transportation_barrier', False)
        })

        logger.info("\nExample Intervention Plan:")
        for need, interventions in intervention_plan.items():
            logger.info(f"\n{need.upper()}:")
            for intervention in interventions:
                logger.info(f"  - {intervention}")

    logger.info("\n" + "=" * 80)
    logger.info("WORKFLOW COMPLETE")
    logger.info("=" * 80)

if __name__ == "__main__":
    example_workflow()
```

## Clinical Applications and Implementation Considerations

### Chronic Disease Management

Social determinants profoundly affect chronic disease incidence, progression, and control. Diabetes management illustrates the importance of SDOH integration. Food insecurity undermines glycemic control by forcing tradeoffs between medication costs and food purchases, creating cyclical patterns where hyperglycemia requires more medication which further strains budgets (Seligman and Schillinger, 2010). Housing instability disrupts medication routines and healthcare follow-up. Limited health literacy and numeracy create barriers to self-management including carbohydrate counting and insulin dosing.

Healthcare AI systems for diabetes management that incorporate SDOH data can identify patients whose poor control reflects social barriers rather than biological factors or "non-compliance." These patients benefit from interventions addressing food access through medically-tailored meal delivery, financial assistance for medication costs, or simplification of medication regimens to accommodate unstable living circumstances, rather than intensification of pharmacologic therapy alone (Berkowitz et al., 2019).

Cardiovascular disease prevention and management similarly require SDOH integration. Hypertension control is substantially worse among individuals experiencing housing instability, food insecurity, or chronic stress from discrimination (Havranek et al., 2015). Cardiac rehabilitation completion rates are far lower among patients lacking transportation or living in neighborhoods without safe walking routes. Predictive models for cardiovascular events that omit social determinants will systematically underestimate risk for marginalized populations, missing opportunities for prevention.

### Emergency Department Utilization and Hospital Readmissions

Frequent emergency department use and hospital readmissions often reflect unmet social needs rather than inadequate medical care. Patients experiencing homelessness use emergency departments at ten times the rate of housed individuals, typically for conditions amenable to primary care if access barriers were addressed (Ku et al., 2009). Hospital readmissions are strongly predicted by housing instability, food insecurity, lack of transportation to follow-up appointments, and social isolation (Kangovi et al., 2014).

Healthcare systems increasingly implement SDOH screening and intervention programs to reduce avoidable utilization. Randomized trials demonstrate that addressing social needs through resource navigation reduces emergency department visits and hospitalizations (Berkowitz et al., 2018; Gottlieb et al., 2016). Machine learning models predicting readmission risk that incorporate SDOH data achieve better discrimination and can identify patients benefiting from intensive post-discharge support including housing placement, meal delivery, transportation assistance, and navigation support (Amarasingham et al., 2010).

### Precision Medicine and Treatment Selection

Precision medicine aims to tailor treatment to individual patient characteristics, but has largely focused on genomic and biological factors while ignoring social context. This creates misalignment when social determinants override biological treatment effects. For example, oncology treatment selection based on tumor genomics alone may recommend regimens requiring frequent clinic visits, which are infeasible for patients lacking transportation or unable to miss work without job loss (Borno et al., 2020).

Integrating SDOH data enables more appropriate precision medicine that considers treatment feasibility, not just biological efficacy. Algorithm-driven clinical decision support for treatment selection should incorporate:

- **Medication affordability**: Recommending less expensive medications when out-of-pocket costs create adherence barriers
- **Regimen complexity**: Simplifying regimens for patients with unstable housing, limited health literacy, or competing survival demands
- **Visit frequency**: Using extended-interval formulations or telemedicine when transportation or time off work are barriers
- **Side effect management**: Considering whether patients have resources to manage side effects at home versus requiring emergency care

Treatment effect heterogeneity analyses examining modification by SDOH can identify populations requiring alternative approaches. If treatment efficacy varies by food security status, neighborhood disadvantage, or other social factors, precision medicine algorithms should account for these effect modifiers when generating recommendations.

### Population Health Management

Population health programs aim to improve outcomes and reduce costs for defined populations through proactive risk stratification and targeted intervention. SDOH integration is essential for identifying highest-need patients and allocating resources effectively.

Traditional clinical risk stratification based on diagnosis codes, utilization history, and biometric data systematically misclassifies many high-need patients. Individuals with multiple untreated chronic conditions may have few recent healthcare encounters if access barriers prevent care seeking. Those with well-controlled disease markers may be at high risk of decompensation if social supports erode.

Multi-dimensional need assessment combining clinical risk, functional status, and social determinants provides more comprehensive characterization. Patients with high clinical risk and high social needs—often termed "high utilizers" or "super-utilizers"—require intensive case management, social service integration, and sustained support (Gawande, 2011). Machine learning models predicting total cost of care show improved performance when incorporating SDOH features, enabling better targeting of intensive management programs.

Population health analytics should stratify outcomes by social determinants to identify disparities and evaluate whether interventions reduce or exacerbate inequities. Simply monitoring aggregate population metrics can mask worsening disparities if improvements concentrate among advantaged subgroups. Stratified reporting and equity-focused dashboards ensure accountability for equitable care delivery.

## Ethical Considerations and Avoiding Harm

### Privacy and Surveillance Concerns

Collecting and integrating detailed social determinants data raises legitimate privacy concerns, particularly for marginalized populations with historical trauma from surveillance and social control systems (Veinot et al., 2019). Healthcare systems' linkage of clinical data with housing, employment, criminal justice, and immigration information could enable discrimination or government surveillance.

Safeguards must include:

1. **Purpose limitation**: SDOH data collected exclusively for connecting patients to resources and improving care, not for punitive uses
2. **Data governance**: Community advisory boards with meaningful authority over data use policies
3. **Transparency**: Clear communication about what data are collected, how they are used, and with whom they are shared
4. **Patient control**: Options to decline SDOH data collection or linkage without penalty
5. **Security**: Strong technical protections against unauthorized access
6. **Policy advocacy**: Healthcare systems should oppose data sharing mandates that could enable immigration enforcement, evictions, or other harms

### Avoiding Algorithmic Redlining

Healthcare AI systems using SDOH data risk creating algorithmic redlining—denying resources or opportunities based on neighborhood or social characteristics (Barocas and Selbst, 2016). For example:

- **Insurance pricing**: Using neighborhood disadvantage to set premiums creates discriminatory pricing
- **Treatment recommendations**: Denying beneficial treatments deemed "not cost-effective" for socially disadvantaged patients
- **Resource allocation**: Excluding patients with housing instability from intensive management programs that require stable addresses

These practices rationalize inequality under the guise of efficiency or cost-effectiveness. Healthcare has obligations to address rather than accommodate social disadvantage. Fair machine learning for healthcare must incorporate principles of:

- **Anti-discrimination**: SDOH data used to identify needs and target support, never to deny care
- **Equal treatment**: Beneficial interventions offered regardless of social characteristics
- **Disparity reduction**: Algorithms evaluated on whether they reduce versus perpetuate outcome gaps

### Structural Interventions Beyond Individual Care

While individual-level SDOH interventions like resource navigation provide valuable support, they cannot address the structural causes of health inequities including poverty, racism, housing unaffordability, and disinvestment in marginalized communities (Marmot, 2005). Healthcare systems risk treating social determinants screening and referral as sufficient responses to inequality while ignoring their roles in perpetuating structural injustice.

Healthcare AI researchers and implementers have responsibilities to advocate for upstream interventions and policy changes:

- **Living wages**: Supporting minimum wage increases and labor protections
- **Affordable housing**: Advocating for housing policy including rent control, eviction prevention, and public housing investment
- **Environmental justice**: Demanding pollution reduction in disproportionately burdened communities
- **Anti-racism**: Addressing healthcare system contributions to racism including biased algorithms, segregated care delivery, and discriminatory practices

Population health analytics should be leveraged for advocacy by documenting health impacts of structural inequities and evaluating health effects of policy changes. This requires causal inference methods identifying policy effects and public reporting holding decision-makers accountable.

## Summary

Social determinants of health account for the majority of health variation across populations yet are frequently omitted from healthcare AI systems. This chapter presented comprehensive approaches for integrating SDOH data including linking clinical records with external sources on housing, food access, environmental exposures, and neighborhood characteristics while protecting privacy; implementing validated individual-level screening instruments; constructing composite vulnerability indices; applying causal inference methods to understand mechanisms; and developing risk stratification systems that treat social factors as intervention targets.

Key principles include: prioritizing external data linkage and spatial integration to enrich clinical data; implementing trauma-informed universal screening with immediate response capacity; using theory-driven methods for composite measure construction while avoiding deficit-based framing; applying rigorous causal inference to identify modifiable determinants amenable to intervention; stratifying all analyses by social determinants to monitor equity; centering community voice in data governance; and advocating for structural interventions beyond individual care.

Production-ready implementations demonstrated privacy-preserving record linkage, spatial data integration, multiple imputation for screening data, latent variable modeling for composite indices, mediation analysis, instrumental variable methods, difference-in-differences designs, and risk stratification with intervention targeting. These methods enable healthcare AI systems that recognize patients as situated within social contexts fundamentally shaping health, allocate resources based on holistic need assessment, and contribute to reducing rather than perpetuating health inequities.

## Bibliography

Alexander, M. (2010). The New Jim Crow: Mass Incarceration in the Age of Colorblindness. The New Press, New York.

Aldridge, R. W., Story, A., Hwang, S. W., Nordentoft, M., Luchenski, S. A., Hartwell, G., Tweed, E. J., Lewer, D., Vittal Katikireddi, S., and Hayward, A. C. (2018). Morbidity and mortality in homeless individuals, prisoners, sex workers, and individuals with substance use disorders in high-income countries: a systematic review and meta-analysis. The Lancet, 391(10117):241–250.

Alderwick, H. and Gottlieb, L. M. (2019). Meanings and misunderstandings: a social determinants of health lexicon for health care systems. The Milbank Quarterly, 97(2):407–419.

Amarasingham, R., Moore, B. J., Tabak, Y. P., Drazner, M. H., Clark, C. A., Zhang, S., Reed, W. G., Swanson, T. S., Ma, Y., and Halm, E. A. (2010). An automated model to identify heart failure patients at risk for 30-day readmission or death using electronic medical record data. Medical Care, 48(11):981–988.

Angrist, J. D. and Pischke, J. S. (2009). Mostly Harmless Econometrics: An Empiricist's Companion. Princeton University Press, Princeton, NJ.

Barocas, S. and Selbst, A. D. (2016). Big data's disparate impact. California Law Review, 104:671–732.

Berkowitz, S. A., Hulberg, A. C., Standish, S., Reznor, G., and Atlas, S. J. (2017). Addressing unmet basic resource needs as part of chronic cardiometabolic disease management. JAMA Internal Medicine, 177(2):244–252.

Berkowitz, S. A., O'Neill, J., Sayer, E., Shahid, N. N., Petry, E., and Atlas, S. J. (2019). Health center–based community-supported agriculture: an RCT. American Journal of Preventive Medicine, 57(6):S55–S64.

Berkowitz, S. A., Hulberg, A. C., Hong, C., Stowell, B. J., Tirozzi, K. J., Traore, C. Y., and Atlas, S. J. (2018). Addressing basic resource needs to improve primary care quality: a community collaboration programme. BMJ Quality & Safety, 27(3):164–172.

Bollen, K. A. (1989). Structural Equations with Latent Variables. John Wiley & Sons, New York.

Borno, H. T., Zhang, S., Gomez, S., and Siegel, A. (2020). The impact of social determinants of health on tumor stage at diagnosis for Black and White men with prostate cancer in the United States. Cancer, 126(4):809–816.

Boyd, A. D., Hosner, C., Hunscher, D. A., Athey, B. D., Clauw, D. J., and Green, L. A. (2007). An 'Honest Broker' mechanism to maintain privacy for patient care and academic medical research. International Journal of Medical Informatics, 76(5-6):407–411.

Braveman, P. and Gottlieb, L. (2014). The social determinants of health: it's time to consider the causes of the causes. Public Health Reports, 129(Suppl 2):19–31.

Callaway, B. and Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. Journal of Econometrics, 225(2):200–230.

Cantor, M. N. and Thorpe, L. (2018). Integrating data on social determinants of health into electronic health records. Health Affairs, 37(4):585–590.

Cantor, J. H., Cohen, L., Mikkelsen, L., Pañares, R., Srikantharajah, J., and Valdovinos, E. (2020). Community-centered health homes: bridging the gap between health services and community prevention. The Milbank Quarterly, 98(1):1–34.

Churchwell, K., Elkind, M. S., Benjamin, R. M., Carson, A. P., Chang, E. K., Lawrence, W., Mills, A., Odom, T. M., Rodriguez, C. J., and Rodriguez, F. (2020). Call to action: structural racism as a fundamental driver of health disparities: a presidential advisory from the American Heart Association. Circulation, 142(24):e454–e468.

Coleman-Jensen, A., Rabbitt, M. P., Gregory, C. A., and Singh, A. (2020). Household Food Security in the United States in 2019. United States Department of Agriculture Economic Research Service, Washington, DC.

Crenshaw, K. (1989). Demarginalizing the intersection of race and sex: a Black feminist critique of antidiscrimination doctrine, feminist theory and antiracist politics. University of Chicago Legal Forum, 1989(1):139–167.

Dannefer, D. (2003). Cumulative advantage/disadvantage and the life course: cross-fertilizing age and social science theory. The Journals of Gerontology Series B: Psychological Sciences and Social Sciences, 58(6):S327–S337.

Diez Roux, A. V. and Mair, C. (2010). Neighborhoods and health. Annals of the New York Academy of Sciences, 1186(1):125–145.

Dwork, C. and Roth, A. (2014). The algorithmic foundations of differential privacy. Foundations and Trends in Theoretical Computer Science, 9(3-4):211–407.

Ekblaw, A., Azaria, A., Halamka, J. D., and Lippman, A. (2016). A case study for blockchain in healthcare: "MedRec" prototype for electronic health records and medical research data. In Proceedings of IEEE Open & Big Data Conference, pages 13–13.

Flanagan, B. E., Gregory, E. W., Hallisey, E. J., Heitgerd, J. L., and Lewis, B. (2011). A social vulnerability index for disaster management. Journal of Homeland Security and Emergency Management, 8(1).

Ford, C. L. and Airhihenbuwa, C. O. (2010). Critical race theory, race equity, and public health: toward antiracism praxis. American Journal of Public Health, 100(S1):S30–S35.

Gallo, L. C., Penedo, F. J., Espinosa de los Monteros, K., and Arguelles, W. (2009). Resiliency in the face of disadvantage: do Hispanic cultural characteristics protect health outcomes? Journal of Personality, 77(6):1707–1746.

Garg, A., Toy, S., Tripodis, Y., Silverstein, M., and Freeman, E. (2015). Addressing social determinants of health at well child care visits: a cluster RCT. Pediatrics, 135(2):e296–e304.

Garg, A., Butz, A. M., Dworkin, P. H., Lewis, R. A., Thompson, R. E., and Serwint, J. R. (2007). Improving the management of family psychosocial problems at low-income children's well-child care visits: the WE CARE Project. Pediatrics, 120(3):547–558.

Gawande, A. (2011). The hot spotters: can we lower medical costs by giving the neediest patients better care? The New Yorker, January 24, 2011.

Geronimus, A. T. (1992). The weathering hypothesis and the health of African-American women and infants: evidence and speculations. Ethnicity & Disease, 2(3):207–221.

Gottlieb, L. M., Wing, H., and Adler, N. E. (2016). A systematic review of interventions on patients' social and economic needs. American Journal of Preventive Medicine, 53(5):719–729.

Gottlieb, L. M., Hessler, D., Long, D., Laves, E., Burns, A. R., Amaya, A., Sweeney, P., Schudel, C., and Adler, N. E. (2016). Effects of social needs screening and in-person service navigation on child health: a randomized clinical trial. JAMA Pediatrics, 170(11):e162521.

Gottlieb, L. M., Tirozzi, K. J., Manchanda, R., Burns, A. R., and Sandel, M. T. (2015). Moving electronic medical records upstream: incorporating social determinants of health. American Journal of Preventive Medicine, 48(2):215–218.

Gottlieb, L. M., Cottrell, E. K., Park, B., Clark, K. D., Gold, R., and Fichtenberg, C. (2019). Advancing social prescribing with implementation science. Journal of the American Board of Family Medicine, 31(3):315–321.

Gundersen, C. and Ziliak, J. P. (2015). Food insecurity and health outcomes. Health Affairs, 34(11):1830–1839.

Havranek, E. P., Mujahid, M. S., Barr, D. A., Blair, I. V., Cohen, M. S., Cruz-Flores, S., Davey-Smith, G., Dennison-Himmelfarb, C. R., Lauer, M. S., Lockwood, D. W., et al. (2015). Social determinants of risk and outcomes for cardiovascular disease: a scientific statement from the American Heart Association. Circulation, 132(9):873–898.

Health Leads (2018). Health Leads Screening Toolkit. Health Leads, Boston, MA.

Holt-Lunstad, J., Smith, T. B., and Layton, J. B. (2010). Social relationships and mortality risk: a meta-analytic review. PLoS Medicine, 7(7):e1000316.

Hood, C. M., Gennuso, K. P., Swain, G. R., and Catlin, B. B. (2016). County health rankings: relationships between determinant factors and health outcomes. American Journal of Preventive Medicine, 50(2):129–135.

Kangovi, S., Mitra, N., Grande, D., White, M. L., McCollum, S., Sellman, J., Shannon, R. P., and Long, J. A. (2014). Patient-centered community health worker intervention to improve posthospital outcomes: a randomized clinical trial. JAMA Internal Medicine, 174(4):535–543.

Karaye, I. M. and Horney, J. A. (2020). The impact of social vulnerability on COVID-19 in the U.S.: an analysis of spatially varying relationships. American Journal of Preventive Medicine, 59(3):317–325.

Kind, A. J. H. and Buckingham, W. R. (2018). Making neighborhood-disadvantage metrics accessible—the Neighborhood Atlas. New England Journal of Medicine, 378(26):2456–2458.

Kind, A. J. H., Jencks, S., Brock, J., Yu, M., Bartels, C., Ehlenbach, W., Greenberg, C., and Smith, M. (2014). Neighborhood socioeconomic disadvantage and 30-day rehospitalization: a retrospective cohort study. Annals of Internal Medicine, 161(11):765–774.

Kind, A. J. H., Buckingham, W. R., Bartels, C. M., Caplan, A., Ross, D., Freiberg, M., Quesnell, M., and Smith, M. A. (2018). Neighborhood socioeconomic disadvantage and 30 day rehospitalizations: an analysis of Medicare data. Annals of Internal Medicine, 168(4):253–261.

Kline, R. B. (2015). Principles and Practice of Structural Equation Modeling, 4th edition. Guilford Press, New York.

Kretzmann, J. P. and McKnight, J. L. (1993). Building Communities from the Inside Out: A Path Toward Finding and Mobilizing a Community's Assets. ACTA Publications, Chicago.

Ku, B. S., Scott, K. C., Kertesz, S. G., and Pitts, S. R. (2009). Factors associated with use of urban emergency departments by the U.S. homeless population. Public Health Reports, 125(3):398–405.

Landrigan, P. J., Fuller, R., Acosta, N. J. R., Adeyi, O., Arnold, R., Basu, N., Baldé, A. B., Bertollini, R., Bose-O'Reilly, S., and Boufford, J. I. (2018). The Lancet Commission on pollution and health. The Lancet, 391(10119):462–512.

Lee, D. S. and Lemieux, T. (2010). Regression discontinuity designs in economics. Journal of Economic Literature, 48(2):281–355.

Link, B. G. and Phelan, J. (1995). Social conditions as fundamental causes of disease. Journal of Health and Social Behavior, 35:80–94.

Little, R. J. A. (1993). Pattern-mixture models for multivariate incomplete data. Journal of the American Statistical Association, 88(421):125–134.

Marmot, M. (2005). Social determinants of health inequalities. The Lancet, 365(9464):1099–1104.

Marmot, M. and Wilkinson, R. (2005). Social Determinants of Health, 2nd edition. Oxford University Press, Oxford.

McGinnis, J. M., Williams-Russo, P., and Knickman, J. R. (2002). The case for more active policy attention to health promotion. Health Affairs, 21(2):78–93.

McMahan, B., Moore, E., Ramage, D., Hampson, S., and y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS), pages 1273–1282.

Metzl, J. M. and Hansen, H. (2014). Structural competency: theorizing a new medical engagement with stigma and inequality. Social Science & Medicine, 103:126–133.

Mohai, P., Lantz, P. M., Morenoff, J., House, J. S., and Mero, R. P. (2009). Racial and socioeconomic disparities in residential proximity to polluting industrial facilities: evidence from the Americans' Changing Lives Study. American Journal of Public Health, 99(S3):S649–S656.

National Association of Community Health Centers (2016). Protocol for Responding to and Assessing Patients' Assets, Risks, and Experiences (PRAPARE). NACHC, Bethesda, MD.

Nussbaum, M. C. (2011). Creating Capabilities: The Human Development Approach. Harvard University Press, Cambridge, MA.

Pascoe, E. A. and Smart Richman, L. (2009). Perceived discrimination and health: a meta-analytic review. Psychological Bulletin, 135(4):531–554.

Rubin, D. B. (1987). Multiple Imputation for Nonresponse in Surveys. John Wiley & Sons, New York.

Sampson, P. D., Richards, M., Szpiro, A. A., Bergen, S., Sheppard, L., Larson, T. V., and Kaufman, J. D. (2013). A regionalized national universal kriging model using Partial Least Squares regression for estimating annual PM2.5 concentrations in epidemiology. Atmospheric Environment, 75:383–392.

Schnell, R., Bachteler, T., and Reiher, J. (2009). Privacy-preserving record linkage using Bloom filters. BMC Medical Informatics and Decision Making, 9(1):41.

Seligman, H. K. and Schillinger, D. (2010). Hunger and socioeconomic disparities in chronic disease. New England Journal of Medicine, 363(1):6–9.

Sen, A. (1999). Development as Freedom. Oxford University Press, Oxford.

Shaw, M. (2004). Housing and public health. Annual Review of Public Health, 25:397–418.

Singh, G. K. (2003). Area deprivation and widening inequalities in US mortality, 1969–1998. American Journal of Public Health, 93(7):1137–1143.

Solar, O. and Irwin, A. (2010). A Conceptual Framework for Action on the Social Determinants of Health. World Health Organization, Geneva.

Stock, J. H., Wright, J. H., and Yogo, M. (2002). A survey of weak instruments and weak identification in generalized method of moments. Journal of Business & Economic Statistics, 20(4):518–529.

Stringhini, S., Carmeli, C., Jokela, M., Avendaño, M., Muennig, P., Guida, F., Ricceri, F., d'Errico, A., Barros, H., Bochud, M., et al. (2017). Socioeconomic status and the 25 × 25 risk factors as determinants of premature mortality: a multicohort study and meta-analysis of 1·7 million men and women. The Lancet, 389(10075):1229–1237.

Substance Abuse and Mental Health Services Administration (2014). SAMHSA's Concept of Trauma and Guidance for a Trauma-Informed Approach. SAMHSA, Rockville, MD.

Sun, L. and Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. Journal of Econometrics, 225(2):175–199.

Umberson, D. and Montez, J. K. (2010). Social relationships and health: a flashpoint for health policy. Journal of Health and Social Behavior, 51(S):S54–S66.

VanderWeele, T. (2015). Explanation in Causal Inference: Methods for Mediation and Interaction. Oxford University Press, Oxford.

VanderWeele, T. J. (2010). Bias formulas for sensitivity analysis for direct and indirect effects. Epidemiology, 21(4):540–551.

Vatsalan, D., Sehili, Z., Christen, P., and Rahm, E. (2017). Privacy-preserving record linkage for big data: current approaches and research challenges. In Handbook of Big Data Technologies, pages 851–895. Springer, Cham.

Veinot, T. C., Mitchell, H., and Ancker, J. S. (2019). Good intentions are not enough: how informatics interventions can worsen inequality. Journal of the American Medical Informatics Association, 25(8):1080–1088.

Williams, D. R. and Collins, C. (2001). Racial residential segregation: a fundamental cause of racial disparities in health. Public Health Reports, 116(5):404–416.

Williams, D. R. and Mohammed, S. A. (2013). Racism and health I: pathways and scientific evidence. American Behavioral Scientist, 57(8):1152–1173.

Zajacova, A. and Lawrence, E. M. (2018). The relationship between education and health: reducing disparities through a contextual approach. Annual Review of Public Health, 39:273–289.
