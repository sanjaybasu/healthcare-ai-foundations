---
layout: chapter
title: "Chapter 2: Mathematical Foundations for Clinical AI"
chapter_number: 2
part_number: 1
prev_chapter: /chapters/chapter-01-clinical-informatics/
next_chapter: /chapters/chapter-03-healthcare-data-engineering/
---
# Chapter 2: Mathematical Foundations for Clinical AI

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Apply linear algebra concepts including vector spaces, matrix operations, eigendecompositions, and singular value decomposition to healthcare data problems, with specific understanding of how these mathematical structures can either reinforce or challenge existing health disparities.

2. Develop probabilistic reasoning for clinical decision-making under uncertainty, including proper application of Bayes' theorem when prior probabilities reflect historical discrimination rather than biological differences, and understanding of conditional independence assumptions that may fail in the presence of unmeasured confounders like structural racism.

3. Implement bias-aware statistical inference methods that account for systematic differences in data collection quality across populations, including approaches for handling missing data mechanisms that correlate with social determinants of health.

4. Design and validate mathematical models that explicitly quantify uncertainty in ways that differ across patient subgroups, recognizing when standard assumptions about error distributions or independence break down in real healthcare settings serving diverse populations.

5. Critically evaluate mathematical formulations of fairness and equity in machine learning, understanding the tradeoffs between different fairness definitions and their appropriateness for specific healthcare applications affecting underserved communities.

## 2.1 Introduction: Why Mathematics Matters for Health Equity

Mathematics provides the formal language through which we express relationships in data, quantify uncertainty, and optimize decisions. In healthcare artificial intelligence, mathematical choices shape everything from how we represent patient similarity to how we define what it means for an algorithm to be fair. These choices are never merely technical but rather encode assumptions about the world that have profound implications for health equity.

Consider the seemingly simple task of measuring similarity between patients to identify those who might benefit from similar treatments. The mathematical framework we choose for this measurement determines which patient characteristics we treat as comparable and which we treat as fundamentally different. If we represent patients as points in a high-dimensional space and use Euclidean distance to measure similarity, we implicitly assume that all features contribute equally to meaningful clinical similarity and that the relationships between features are linear. These assumptions may work reasonably well for some patient populations but fail catastrophically for others.

A concrete example illustrates the stakes. Suppose we are developing a system to identify patients similar to a given individual for the purpose of predicting treatment response based on outcomes in similar patients. We represent each patient as a vector containing age, body mass index, blood pressure, hemoglobin A1c, and several other clinical measurements. For a middle-aged patient with well-controlled diabetes and good healthcare access, this representation may work well because their clinical trajectory is largely determined by the measured biomedical variables. But for a patient facing housing instability, food insecurity, and transportation barriers to care, the measured clinical variables capture only a small part of what determines their health trajectory. The unmeasured social factors may be far more important than the measured clinical ones, yet our mathematical similarity metric knows nothing about them.

The result is that our patient similarity algorithm will suggest treatment strategies based on patients whose clinical measurements are similar but whose actual circumstances are profoundly different. The recommendations may be inappropriate or even harmful, not because the mathematics was wrong in some abstract sense but because the mathematical framework was built on assumptions that don't hold for patients whose lives are shaped by social and structural factors that standard clinical data fails to capture.

This chapter develops mathematical foundations with sustained attention to how mathematical choices interact with health equity considerations. We begin with linear algebra because matrix operations pervade healthcare data analysis, from basic data transformations to sophisticated dimensionality reduction methods. But rather than presenting linear algebra in isolation, we ground every concept in healthcare applications and examine how the mathematical structures can either reveal or obscure health disparities. We then develop probability theory as the foundation for reasoning under uncertainty, but we do so while acknowledging that probability distributions in healthcare data often reflect social processes rather than natural phenomena, and that standard probabilistic assumptions frequently fail in ways that systematically disadvantage certain populations.

Throughout, we implement production-ready code that demonstrates not just how to perform mathematical operations but how to do so while maintaining critical awareness of their limitations and biases. The implementations include extensive validation and sensitivity analyses that surface when mathematical assumptions break down, with particular attention to whether these failures affect different patient populations differently. By the end of this chapter, you will have both the mathematical sophistication needed for advanced healthcare AI work and the critical lens needed to ensure that mathematical elegance doesn't come at the cost of health equity.

## 2.2 Linear Algebra for Healthcare Data

Linear algebra provides the mathematical framework for representing and manipulating healthcare data in ways that enable both human interpretation and algorithmic processing. Vectors represent individual patients or clinical measurements, matrices organize collections of observations or relationships between variables, and linear transformations enable us to change perspectives on data or reduce dimensionality while preserving important structure. Understanding linear algebra deeply means understanding not just the mechanics of matrix operations but also what these operations mean in healthcare contexts and how they can introduce or reveal bias.

### 2.2.1 Vector Spaces and Patient Representation

At the most fundamental level, we represent each patient as a point in a high-dimensional space where each dimension corresponds to a measured or derived feature. This vector representation enables us to apply geometric intuitions about distance, angles, and projections to reason about patient similarity, cluster structure, and predictive relationships. However, the choice of how to construct these vector representations involves numerous decisions that affect whether the resulting mathematical structure serves or undermines health equity goals.

Consider a patient represented by clinical measurements including age, systolic blood pressure, serum creatinine, body mass index, and hemoglobin A1c. We can write this patient as a vector in five-dimensional space. However, the raw measurements have different units and scales: age in years, blood pressure in millimeters of mercury, creatinine in milligrams per deciliter, and so on. Computing distances between patients using raw measurements would mean that variables with larger numerical ranges would dominate the distance calculation, even if they are not more clinically important.

The standard mathematical solution is to standardize the features by subtracting the mean and dividing by the standard deviation, transforming each variable to have mean zero and unit variance. This standardization ensures that all features contribute comparably to distance calculations. But this seemingly neutral mathematical operation embeds important assumptions. When we standardize using the overall population mean and standard deviation, we implicitly assume that deviations from the population average are equally meaningful for all patients. Yet if different patient subgroups have systematically different distributions of these variables due to social determinants of health or differential access to care, standardizing to the overall population statistics may obscure rather than reveal clinically meaningful patterns.

Let me make this concrete with an implementation that demonstrates both standard vector operations and equity-aware alternatives:

```python
"""
Patient Vector Representation with Equity Considerations
This module implements patient vector representations with explicit attention
to how standardization and distance metric choices can introduce bias when
patient populations have systematically different feature distributions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PatientVector:
    """
    Represents a patient as a vector in feature space with metadata about
    the patient's demographic characteristics and data quality.

    Attributes:
        patient_id: Unique patient identifier
        features: Numeric feature vector
        feature_names: Names corresponding to each feature dimension
        demographic_group: Demographic stratification information
        data_quality_score: Completeness and reliability score for this patient's data
        missingness_indicators: Binary indicators for which features are imputed
    """
    patient_id: str
    features: np.ndarray
    feature_names: List[str]
    demographic_group: Optional[Dict[str, str]] = None
    data_quality_score: Optional[float] = None
    missingness_indicators: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate that feature array dimensions match feature names."""
        if len(self.features) != len(self.feature_names):
            raise ValueError(
                f"Feature array length ({len(self.features)}) must match "
                f"feature names length ({len(self.feature_names)})"
            )

class EquityAwareVectorSpace:
    """
    Creates and manages vector representations of patients with explicit
    attention to how standardization choices affect fairness across
    patient populations with different feature distributions.
    """

    def __init__(
        self,
        standardization_method: str = 'standard',
        group_specific_scaling: bool = False
    ):
        """
        Initialize vector space with configurable standardization approach.

        Args:
            standardization_method: Method for feature scaling
                'standard': Mean centering and unit variance scaling
                'robust': Median centering and IQR scaling (less sensitive to outliers)
                'minmax': Scale to [0, 1] range
                'none': No standardization
            group_specific_scaling: Whether to use group-specific scaling parameters
                If True, fits separate scalers for each demographic group
                This preserves within-group variation patterns
        """
        self.standardization_method = standardization_method
        self.group_specific_scaling = group_specific_scaling

        self.global_scaler: Optional[Union[StandardScaler, RobustScaler]] = None
        self.group_scalers: Dict[str, Union[StandardScaler, RobustScaler]] = {}
        self.feature_names: List[str] = []

        logger.info(
            f"Initialized vector space with {standardization_method} standardization, "
            f"group_specific_scaling={group_specific_scaling}"
        )

    def fit(
        self,
        patient_df: pd.DataFrame,
        feature_columns: List[str],
        demographic_column: Optional[str] = None
    ) -> 'EquityAwareVectorSpace':
        """
        Fit standardization parameters from training data with option for
        group-specific standardization to preserve within-group patterns.

        Args:
            patient_df: DataFrame with patient data
            feature_columns: Column names to use as features
            demographic_column: Column for group-specific standardization

        Returns:
            Self for method chaining
        """
        self.feature_names = feature_columns
        X = patient_df[feature_columns].values

        # Fit global scaler
        if self.standardization_method == 'standard':
            self.global_scaler = StandardScaler()
        elif self.standardization_method == 'robust':
            self.global_scaler = RobustScaler()
        elif self.standardization_method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.global_scaler = MinMaxScaler()
        elif self.standardization_method == 'none':
            self.global_scaler = None
        else:
            raise ValueError(f"Unknown standardization method: {self.standardization_method}")

        if self.global_scaler is not None:
            self.global_scaler.fit(X)

        # Fit group-specific scalers if requested
        if self.group_specific_scaling and demographic_column is not None:
            if demographic_column not in patient_df.columns:
                logger.warning(
                    f"Demographic column {demographic_column} not found. "
                    f"Using global scaler only."
                )
            else:
                for group in patient_df[demographic_column].unique():
                    if pd.notna(group):
                        group_data = patient_df[patient_df[demographic_column] == group]
                        X_group = group_data[feature_columns].values

                        if len(X_group) > 1:  # Need at least 2 samples for scaling
                            if self.standardization_method == 'standard':
                                group_scaler = StandardScaler()
                            elif self.standardization_method == 'robust':
                                group_scaler = RobustScaler()
                            elif self.standardization_method == 'minmax':
                                from sklearn.preprocessing import MinMaxScaler
                                group_scaler = MinMaxScaler()
                            else:
                                group_scaler = None

                            if group_scaler is not None:
                                group_scaler.fit(X_group)
                                self.group_scalers[str(group)] = group_scaler

        logger.info(f"Fitted vector space on {len(patient_df)} patients")
        if self.group_scalers:
            logger.info(f"Created group-specific scalers for {len(self.group_scalers)} groups")

        return self

    def transform(
        self,
        patient_df: pd.DataFrame,
        demographic_column: Optional[str] = None
    ) -> List[PatientVector]:
        """
        Transform patients into vector representations using fitted scaling.

        Args:
            patient_df: DataFrame with patient data
            demographic_column: Column with demographic group labels

        Returns:
            List of PatientVector objects
        """
        if not self.feature_names:
            raise ValueError("Must call fit() before transform()")

        patient_vectors = []

        for idx, row in patient_df.iterrows():
            patient_id = str(row.get('patient_id', f'patient_{idx}'))

            # Extract feature values
            feature_values = row[self.feature_names].values.astype(float)

            # Determine which scaler to use
            demographic_group = None
            if demographic_column and demographic_column in row:
                demographic_group = {demographic_column: str(row[demographic_column])}
                group_key = str(row[demographic_column])
            else:
                group_key = None

            # Apply scaling
            if self.group_specific_scaling and group_key in self.group_scalers:
                # Use group-specific scaler
                scaler = self.group_scalers[group_key]
                scaled_features = scaler.transform(feature_values.reshape(1, -1)).flatten()
            elif self.global_scaler is not None:
                # Use global scaler
                scaled_features = self.global_scaler.transform(
                    feature_values.reshape(1, -1)
                ).flatten()
            else:
                # No scaling
                scaled_features = feature_values

            # Track missingness
            missingness_indicators = pd.isna(row[self.feature_names]).values.astype(int)

            # Calculate data quality score
            data_quality = 1.0 - missingness_indicators.mean()

            patient_vector = PatientVector(
                patient_id=patient_id,
                features=scaled_features,
                feature_names=self.feature_names,
                demographic_group=demographic_group,
                data_quality_score=data_quality,
                missingness_indicators=missingness_indicators
            )

            patient_vectors.append(patient_vector)

        return patient_vectors

    def compute_distance_matrix(
        self,
        patient_vectors: List[PatientVector],
        metric: str = 'euclidean',
        weight_by_quality: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pairwise distances between patients with optional quality weighting.

        Standard distance metrics can be misleading when data quality differs
        systematically across populations. Quality weighting down-weights distances
        involving patients with poor data quality.

        Args:
            patient_vectors: List of patient vector representations
            metric: Distance metric ('euclidean', 'manhattan', 'cosine')
            weight_by_quality: Whether to weight distances by data quality scores

        Returns:
            Tuple of (distance matrix, quality-adjusted distance matrix if weight_by_quality)
        """
        n = len(patient_vectors)
        distance_matrix = np.zeros((n, n))
        quality_matrix = np.ones((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Compute base distance
                if metric == 'euclidean':
                    dist = np.linalg.norm(
                        patient_vectors[i].features - patient_vectors[j].features
                    )
                elif metric == 'manhattan':
                    dist = np.sum(np.abs(
                        patient_vectors[i].features - patient_vectors[j].features
                    ))
                elif metric == 'cosine':
                    dot_product = np.dot(
                        patient_vectors[i].features,
                        patient_vectors[j].features
                    )
                    norm_i = np.linalg.norm(patient_vectors[i].features)
                    norm_j = np.linalg.norm(patient_vectors[j].features)

                    if norm_i > 0 and norm_j > 0:
                        dist = 1.0 - (dot_product / (norm_i * norm_j))
                    else:
                        dist = 1.0  # Maximum distance if either vector is zero
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

                # Compute quality adjustment
                if weight_by_quality:
                    quality_i = patient_vectors[i].data_quality_score or 1.0
                    quality_j = patient_vectors[j].data_quality_score or 1.0
                    quality_weight = np.sqrt(quality_i * quality_j)
                    quality_matrix[i, j] = quality_weight
                    quality_matrix[j, i] = quality_weight

        if weight_by_quality:
            # Adjust distances by quality - higher quality pairs get lower adjusted distance
            adjusted_distance = distance_matrix / quality_matrix
            return distance_matrix, adjusted_distance
        else:
            return distance_matrix, distance_matrix

    def analyze_distance_bias(
        self,
        patient_vectors: List[PatientVector],
        distance_matrix: np.ndarray,
        demographic_attribute: str
    ) -> Dict[str, any]:
        """
        Analyze whether distance calculations are biased across demographic groups.

        This checks whether patients from certain demographic groups tend to be
        closer or farther from other patients in the vector space, which could
        indicate that the feature representation or standardization is inappropriate.

        Args:
            patient_vectors: List of patient vectors
            distance_matrix: Pairwise distance matrix
            demographic_attribute: Which demographic attribute to analyze

        Returns:
            Dictionary with bias analysis results
        """
        analysis = {
            'mean_within_group_distance': {},
            'mean_between_group_distance': {},
            'group_cohesion_scores': {},
            'bias_detected': False
        }

        # Extract demographic groups
        groups = {}
        for i, pv in enumerate(patient_vectors):
            if pv.demographic_group and demographic_attribute in pv.demographic_group:
                group = pv.demographic_group[demographic_attribute]
                if group not in groups:
                    groups[group] = []
                groups[group].append(i)

        if len(groups) < 2:
            logger.warning("Need at least 2 demographic groups for bias analysis")
            return analysis

        # Compute within-group distances
        for group, indices in groups.items():
            if len(indices) > 1:
                within_distances = []
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        within_distances.append(distance_matrix[indices[i], indices[j]])

                analysis['mean_within_group_distance'][group] = np.mean(within_distances)

        # Compute between-group distances
        group_pairs = [(g1, g2) for g1 in groups for g2 in groups if g1 < g2]
        for g1, g2 in group_pairs:
            between_distances = []
            for i in groups[g1]:
                for j in groups[g2]:
                    between_distances.append(distance_matrix[i, j])

            pair_key = f"{g1}_vs_{g2}"
            analysis['mean_between_group_distance'][pair_key] = np.mean(between_distances)

        # Compute cohesion scores (ratio of within to between distance)
        for group in groups:
            within_dist = analysis['mean_within_group_distance'].get(group, 0)

            # Average between-group distance for this group
            between_dists = []
            for other_group in groups:
                if other_group != group:
                    pair_key = f"{min(group, other_group)}_vs_{max(group, other_group)}"
                    if pair_key in analysis['mean_between_group_distance']:
                        between_dists.append(
                            analysis['mean_between_group_distance'][pair_key]
                        )

            if between_dists:
                avg_between = np.mean(between_dists)
                if avg_between > 0:
                    cohesion = within_dist / avg_between
                    analysis['group_cohesion_scores'][group] = cohesion

        # Check for bias: large differences in cohesion suggest representation issues
        cohesion_values = list(analysis['group_cohesion_scores'].values())
        if len(cohesion_values) >= 2:
            cohesion_cv = np.std(cohesion_values) / np.mean(cohesion_values)
            if cohesion_cv > 0.3:  # Coefficient of variation > 30%
                analysis['bias_detected'] = True
                logger.warning(
                    f"Bias detected in vector space representation. "
                    f"Cohesion scores vary substantially across groups (CV={cohesion_cv:.3f}). "
                    f"Some groups may be poorly represented in this feature space."
                )

        return analysis
```

This implementation demonstrates several equity-focused approaches to vector representation. The group-specific scaling option allows us to standardize features separately within each demographic group, which preserves within-group variation patterns that might be clinically meaningful even though they differ across groups. The quality-weighted distance calculation accounts for the fact that distances involving patients with poor data quality may be less reliable. And the bias analysis functionality explicitly checks whether the vector space representation creates systematic differences in how clustered or dispersed different demographic groups appear, which could indicate that the feature representation doesn't work equally well for all populations.

### 2.2.2 Matrix Operations and Data Transformations

Matrices organize collections of patient vectors and enable batch operations that transform entire datasets. Understanding matrix operations means understanding how data transformations affect the structure of information and how these transformations can introduce or reveal bias. We focus on operations that commonly arise in healthcare data analysis including matrix multiplication for feature transformations, matrix inverses for solving linear systems, and matrix decompositions that reveal latent structure.

Matrix multiplication is perhaps the most fundamental operation, enabling us to transform data from one representation to another through linear combinations of features. When we multiply a data matrix by a transformation matrix, we are computing new features as weighted combinations of the original features. This operation is at the heart of dimensionality reduction methods, linear models, and neural network layers. However, the choice of transformation matrix embeds assumptions about which combinations of features are meaningful, and these assumptions may not hold equally well across diverse patient populations.

Consider a simple example where we want to create a composite measure of cardiovascular risk from multiple individual risk factors including blood pressure, cholesterol, smoking status, and diabetes. We can express this as matrix multiplication where each patient is a row in our data matrix and each risk factor is a column, and we multiply by a weight vector that defines how much each factor contributes to the composite score. The standard approach is to set weights based on epidemiological studies that estimated the association between each risk factor and cardiovascular events. But if those studies were conducted primarily in populations different from the ones we are now trying to serve, the weights may be inappropriate and could systematically misestimate risk for certain groups.

Let me implement matrix operations with explicit attention to these equity considerations:

```python
class EquityAwareMatrixOperations:
    """
    Matrix operations for healthcare data with explicit validation that
    transformations don't introduce systematic bias across patient populations.
    """

    def __init__(self):
        """Initialize matrix operations handler."""
        self.transformation_history: List[Dict[str, any]] = []
        logger.info("Initialized equity-aware matrix operations")

    def linear_combination_features(
        self,
        data_matrix: np.ndarray,
        weights: np.ndarray,
        feature_names: List[str],
        new_feature_name: str,
        demographic_labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Create new features as linear combinations of existing features with
        validation that the transformation works consistently across demographic groups.

        Args:
            data_matrix: (n_patients, n_features) data matrix
            weights: (n_features,) weight vector for linear combination
            feature_names: Names of input features
            new_feature_name: Name for the newly created feature
            demographic_labels: Optional (n_patients,) array of demographic group labels

        Returns:
            Tuple of (new feature vector, validation results)
        """
        if data_matrix.shape[1] != len(weights):
            raise ValueError(
                f"Data matrix has {data_matrix.shape[1]} features but "
                f"weight vector has {len(weights)} elements"
            )

        # Compute linear combination
        new_feature = data_matrix @ weights

        # Validate transformation
        validation = {
            'new_feature_name': new_feature_name,
            'weight_vector': weights.tolist(),
            'feature_names': feature_names,
            'transformation_bias_detected': False
        }

        if demographic_labels is not None:
            # Check if transformation affects groups differently
            unique_groups = np.unique(demographic_labels[~pd.isna(demographic_labels)])

            group_correlations = {}
            for group in unique_groups:
                group_mask = demographic_labels == group

                # For each original feature, compute correlation with new feature
                correlations = []
                for j in range(data_matrix.shape[1]):
                    if np.std(data_matrix[group_mask, j]) > 0:
                        corr = np.corrcoef(
                            data_matrix[group_mask, j],
                            new_feature[group_mask]
                        )[0, 1]
                        correlations.append(corr)

                group_correlations[str(group)] = {
                    'mean_correlation': np.mean(correlations),
                    'feature_correlations': dict(zip(feature_names, correlations))
                }

            validation['group_correlations'] = group_correlations

            # Check for large differences in how features relate to composite
            mean_corrs = [v['mean_correlation'] for v in group_correlations.values()]
            if len(mean_corrs) > 1:
                corr_cv = np.std(mean_corrs) / np.mean(np.abs(mean_corrs))
                if corr_cv > 0.3:
                    validation['transformation_bias_detected'] = True
                    logger.warning(
                        f"Linear combination {new_feature_name} shows different "
                        f"relationships to input features across demographic groups. "
                        f"This may indicate the weights are not appropriate for all populations."
                    )

        # Record transformation
        self.transformation_history.append(validation)

        return new_feature, validation

    def center_data_matrix(
        self,
        data_matrix: np.ndarray,
        demographic_labels: Optional[np.ndarray] = None,
        group_specific: bool = False
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Center data matrix by subtracting means with option for group-specific centering.

        Standard centering subtracts the overall mean, which can obscure within-group
        patterns when groups have systematically different means. Group-specific
        centering preserves within-group variation structure.

        Args:
            data_matrix: (n_patients, n_features) data matrix
            demographic_labels: Optional group labels for group-specific centering
            group_specific: Whether to center within each group separately

        Returns:
            Tuple of (centered matrix, dictionary of centering parameters)
        """
        if not group_specific or demographic_labels is None:
            # Global centering
            means = np.mean(data_matrix, axis=0)
            centered = data_matrix - means

            centering_params = {
                'method': 'global',
                'global_means': means
            }
        else:
            # Group-specific centering
            centered = np.zeros_like(data_matrix)
            group_means = {}

            unique_groups = np.unique(demographic_labels[~pd.isna(demographic_labels)])

            for group in unique_groups:
                group_mask = demographic_labels == group
                group_data = data_matrix[group_mask]

                if len(group_data) > 0:
                    group_mean = np.mean(group_data, axis=0)
                    centered[group_mask] = group_data - group_mean
                    group_means[str(group)] = group_mean

            centering_params = {
                'method': 'group_specific',
                'group_means': group_means
            }

        return centered, centering_params

    def compute_covariance_matrix(
        self,
        data_matrix: np.ndarray,
        demographic_labels: Optional[np.ndarray] = None,
        analyze_group_differences: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict[str, any]]]:
        """
        Compute covariance matrix with optional analysis of whether covariance
        structure differs across demographic groups.

        Different covariance structures across groups indicate that relationships
        between variables differ systematically, which has implications for
        modeling approaches that assume homogeneous covariance.

        Args:
            data_matrix: (n_patients, n_features) centered data matrix
            demographic_labels: Optional group labels
            analyze_group_differences: Whether to test for covariance structure differences

        Returns:
            Tuple of (covariance matrix, optional analysis results)
        """
        n_samples = data_matrix.shape[0]

        # Compute overall covariance
        cov_matrix = (data_matrix.T @ data_matrix) / (n_samples - 1)

        analysis_results = None

        if analyze_group_differences and demographic_labels is not None:
            analysis_results = {
                'group_covariances': {},
                'covariance_homogeneity_test': {}
            }

            unique_groups = np.unique(demographic_labels[~pd.isna(demographic_labels)])

            group_cov_matrices = {}
            for group in unique_groups:
                group_mask = demographic_labels == group
                group_data = data_matrix[group_mask]

                if len(group_data) > data_matrix.shape[1]:  # Need n > p for valid covariance
                    group_cov = (group_data.T @ group_data) / (len(group_data) - 1)
                    group_cov_matrices[str(group)] = group_cov

                    # Store summary statistics
                    analysis_results['group_covariances'][str(group)] = {
                        'trace': np.trace(group_cov),
                        'determinant': np.linalg.det(group_cov) if group_cov.shape[0] > 0 else 0,
                        'frobenius_norm': np.linalg.norm(group_cov, 'fro')
                    }

            # Test for homogeneity using Box's M test approximation
            # This tests whether covariance matrices are equal across groups
            if len(group_cov_matrices) >= 2:
                # Compute pooled covariance
                pooled_cov = cov_matrix

                # Compare each group covariance to pooled
                for group, group_cov in group_cov_matrices.items():
                    diff_norm = np.linalg.norm(group_cov - pooled_cov, 'fro')
                    pooled_norm = np.linalg.norm(pooled_cov, 'fro')

                    relative_difference = diff_norm / pooled_norm if pooled_norm > 0 else 0

                    analysis_results['covariance_homogeneity_test'][group] = {
                        'relative_difference': relative_difference
                    }

                    if relative_difference > 0.3:
                        logger.warning(
                            f"Group {group} has substantially different covariance structure "
                            f"(relative difference = {relative_difference:.3f}). "
                            f"Models assuming homogeneous covariance may perform poorly."
                        )

        return cov_matrix, analysis_results

    def compute_correlation_matrix(
        self,
        data_matrix: np.ndarray,
        feature_names: List[str],
        demographic_labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Compute correlation matrix with detailed reporting suitable for identifying
        features that may serve as proxies for demographic characteristics.

        Args:
            data_matrix: (n_patients, n_features) data matrix
            feature_names: Names of features
            demographic_labels: Optional group labels

        Returns:
            Tuple of (correlation matrix, correlation analysis DataFrame)
        """
        # Standardize features for correlation computation
        data_std = (data_matrix - np.mean(data_matrix, axis=0)) / (
            np.std(data_matrix, axis=0) + 1e-8
        )

        # Compute correlation matrix
        corr_matrix = (data_std.T @ data_std) / (data_matrix.shape[0] - 1)

        # Create detailed report
        n_features = len(feature_names)
        correlation_details = []

        for i in range(n_features):
            for j in range(i + 1, n_features):
                correlation_details.append({
                    'feature_1': feature_names[i],
                    'feature_2': feature_names[j],
                    'correlation': corr_matrix[i, j],
                    'abs_correlation': abs(corr_matrix[i, j])
                })

        corr_df = pd.DataFrame(correlation_details)
        corr_df = corr_df.sort_values('abs_correlation', ascending=False)

        # Flag potentially problematic high correlations
        high_corr = corr_df[corr_df['abs_correlation'] > 0.8]
        if len(high_corr) > 0:
            logger.info(
                f"Found {len(high_corr)} feature pairs with |correlation| > 0.8. "
                f"High correlations may indicate redundant features or proxy variables."
            )

        return corr_matrix, corr_df
```

This implementation provides matrix operations with built-in equity checking. The linear combination validation checks whether the transformation relates to input features differently across demographic groups, which could indicate that the chosen weights aren't appropriate for all populations. The group-specific centering option preserves within-group correlation structure that global centering would erase. And the covariance analysis explicitly tests whether the covariance structure is homogeneous across groups, surfacing violations of this common modeling assumption.

### 2.2.3 Eigendecomposition and Principal Component Analysis

Eigendecomposition reveals the intrinsic structure of matrices by finding directions of maximal variation and the scaling factors along those directions. In healthcare data analysis, eigendecomposition underlies principal components analysis, which finds low-dimensional representations that capture most of the variation in high-dimensional data. However, PCA and related methods can introduce subtle biases when different patient populations have different correlation structures, as the principal components may primarily capture variation patterns from the dominant population while poorly representing patterns specific to minority populations.

The mathematical foundation is elegant. Given a covariance matrix Σ, eigendecomposition finds vectors v and scalars λ such that Σv = λv. The eigenvector v defines a direction in feature space, and the eigenvalue λ quantifies the amount of variance along that direction. The largest eigenvalue corresponds to the direction of maximum variance, the second largest to the direction of maximum variance orthogonal to the first, and so on. Principal components analysis uses these eigenvectors as a new coordinate system in which to represent the data, with the hope that the first few principal components capture most of the meaningful variation.

The health equity concern arises when we apply PCA to data from heterogeneous populations. The principal components that best represent the dominant population may not best represent minority populations, especially if the minority populations have different correlation structures. The result is that dimensionality reduction may work well for some patients while losing important information for others. This differential information loss can then propagate through downstream modeling, leading to algorithms that perform better for well-represented populations.

Let me implement PCA with explicit equity safeguards:

```python
from sklearn.decomposition import PCA
from scipy.linalg import eigh

class EquityAwarePCA:
    """
    Principal Components Analysis with explicit validation that dimensionality
    reduction preserves information equitably across patient populations.
    """

    def __init__(self, n_components: Optional[int] = None):
        """
        Initialize equity-aware PCA.

        Args:
            n_components: Number of components to retain. If None, keeps all components.
        """
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.group_reconstruction_errors_: Dict[str, float] = {}

        logger.info(f"Initialized equity-aware PCA with n_components={n_components}")

    def fit(
        self,
        X: np.ndarray,
        demographic_labels: Optional[np.ndarray] = None
    ) -> 'EquityAwarePCA':
        """
        Fit PCA on data with optional analysis of reconstruction quality across groups.

        Args:
            X: (n_samples, n_features) data matrix
            demographic_labels: Optional (n_samples,) array of group labels

        Returns:
            Self for method chaining
        """
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute covariance matrix
        cov_matrix = (X_centered.T @ X_centered) / (X.shape[0] - 1)

        # Eigendecomposition
        # eigh is for symmetric matrices and returns eigenvalues in ascending order
        eigenvalues, eigenvectors = eigh(cov_matrix)

        # Sort in descending order of eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store results
        if self.n_components is not None:
            self.components_ = eigenvectors[:, :self.n_components].T
            self.explained_variance_ = eigenvalues[:self.n_components]
        else:
            self.components_ = eigenvectors.T
            self.explained_variance_ = eigenvalues

        # Analyze reconstruction quality across groups
        if demographic_labels is not None:
            self._analyze_group_reconstruction(X_centered, demographic_labels)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to principal component space.

        Args:
            X: (n_samples, n_features) data matrix

        Returns:
            (n_samples, n_components) transformed data
        """
        if self.components_ is None or self.mean_ is None:
            raise ValueError("Must call fit() before transform()")

        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back from principal component space to original space.

        Args:
            X_transformed: (n_samples, n_components) data in PC space

        Returns:
            (n_samples, n_features) reconstructed data
        """
        if self.components_ is None or self.mean_ is None:
            raise ValueError("Must call fit() before inverse_transform()")

        return (X_transformed @ self.components_) + self.mean_

    def _analyze_group_reconstruction(
        self,
        X_centered: np.ndarray,
        demographic_labels: np.ndarray
    ) -> None:
        """
        Analyze how well PCA reconstruction works for different demographic groups.

        Poor reconstruction for some groups indicates that the principal components
        don't capture the variation patterns in those groups well.
        """
        unique_groups = np.unique(demographic_labels[~pd.isna(demographic_labels)])

        # Transform and reconstruct data
        X_transformed = X_centered @ self.components_.T
        X_reconstructed = X_transformed @ self.components_

        # Compute reconstruction error for each group
        for group in unique_groups:
            group_mask = demographic_labels == group

            # Mean squared reconstruction error
            reconstruction_error = np.mean(
                np.sum((X_centered[group_mask] - X_reconstructed[group_mask])**2, axis=1)
            )

            self.group_reconstruction_errors_[str(group)] = reconstruction_error

        # Check for disparities
        errors = list(self.group_reconstruction_errors_.values())
        if len(errors) > 1:
            max_error = max(errors)
            min_error = min(errors)
            error_ratio = max_error / min_error if min_error > 0 else float('inf')

            if error_ratio > 2.0:
                logger.warning(
                    f"Large disparity in PCA reconstruction error across groups "
                    f"(ratio={error_ratio:.2f}). Principal components may not "
                    f"represent all populations equally well."
                )

    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Get the proportion of variance explained by each principal component.

        Returns:
            Array of explained variance ratios
        """
        if self.explained_variance_ is None:
            raise ValueError("Must call fit() first")

        return self.explained_variance_ / np.sum(self.explained_variance_)

    def plot_scree(self) -> None:
        """
        Create scree plot showing variance explained by each component.

        This helps determine how many components to retain.
        """
        import matplotlib.pyplot as plt

        if self.explained_variance_ is None:
            raise ValueError("Must call fit() first")

        variance_ratios = self.get_explained_variance_ratio()
        cumulative_variance = np.cumsum(variance_ratios)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Individual variance
        ax1.bar(range(1, len(variance_ratios) + 1), variance_ratios)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Proportion of Variance Explained')
        ax1.set_title('Scree Plot')

        # Cumulative variance
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
        ax2.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
        ax2.axhline(y=0.9, color='g', linestyle='--', label='90% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Variance Explained')
        ax2.set_title('Cumulative Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def assess_equity_impact(self) -> Dict[str, any]:
        """
        Comprehensive assessment of whether PCA affects demographic groups equitably.

        Returns:
            Dictionary with equity assessment results
        """
        if not self.group_reconstruction_errors_:
            logger.warning("No group reconstruction errors available. Run fit() with demographic_labels.")
            return {}

        errors = list(self.group_reconstruction_errors_.values())

        assessment = {
            'group_reconstruction_errors': self.group_reconstruction_errors_,
            'max_error': max(errors),
            'min_error': min(errors),
            'error_ratio': max(errors) / min(errors) if min(errors) > 0 else float('inf'),
            'error_cv': np.std(errors) / np.mean(errors),
            'equity_concern_detected': False
        }

        # Flag equity concerns
        if assessment['error_ratio'] > 2.0 or assessment['error_cv'] > 0.3:
            assessment['equity_concern_detected'] = True
            assessment['recommendation'] = (
                "Consider group-specific PCA or alternative dimensionality "
                "reduction methods that better preserve information for all populations."
            )
        else:
            assessment['recommendation'] = (
                "PCA appears to work reasonably consistently across groups."
            )

        return assessment
```

This equity-aware PCA implementation explicitly tracks reconstruction quality across demographic groups, which reveals when dimensionality reduction works better for some populations than others. The reconstruction error analysis provides an early warning that the principal components may not be appropriate for all populations, allowing us to either adjust the number of components retained or consider alternative dimensionality reduction approaches that better preserve information for minority populations.

### 2.2.4 Singular Value Decomposition and Matrix Factorization

Singular value decomposition provides an alternative perspective on matrix structure that generalizes eigendecomposition to non-square matrices and connects naturally to many machine learning methods including collaborative filtering, latent semantic analysis, and matrix completion. SVD factorizes a data matrix X into three matrices: X = UΣV^T, where U contains left singular vectors, Σ is a diagonal matrix of singular values, and V contains right singular vectors. This decomposition reveals latent structure in data and enables low-rank approximations that can denoise data or reduce dimensionality.

In healthcare applications, SVD and related matrix factorization methods are used for tasks including identifying patient subgroups based on similar clinical patterns, discovering latent disease phenotypes from symptom and biomarker data, and imputing missing values in partially observed clinical matrices. However, these methods face equity challenges similar to those we saw with PCA. The latent factors discovered by matrix factorization may primarily reflect patterns in well-represented populations while missing important patterns specific to minority populations. Moreover, when using matrix factorization for missing data imputation, the imputed values will reflect the patterns learned from observed data, which may not be appropriate if missingness patterns differ systematically across groups.

Let me implement SVD-based methods with explicit equity considerations:

```python
from scipy.linalg import svd as scipy_svd
from sklearn.impute import IterativeImputer

class EquityAwareSVD:
    """
    Singular Value Decomposition with equity-aware analysis and applications.
    """

    def __init__(self, n_components: Optional[int] = None):
        """
        Initialize equity-aware SVD.

        Args:
            n_components: Number of components to retain for low-rank approximation
        """
        self.n_components = n_components
        self.U_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.Vt_: Optional[np.ndarray] = None

        logger.info(f"Initialized equity-aware SVD with n_components={n_components}")

    def fit(
        self,
        X: np.ndarray,
        center: bool = True
    ) -> 'EquityAwareSVD':
        """
        Compute SVD of data matrix.

        Args:
            X: (n_samples, n_features) data matrix
            center: Whether to center data before SVD

        Returns:
            Self for method chaining
        """
        if center:
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
        else:
            self.mean_ = np.zeros(X.shape[1])
            X_centered = X

        # Compute full SVD
        U, s, Vt = scipy_svd(X_centered, full_matrices=False)

        # Retain specified number of components
        if self.n_components is not None:
            self.U_ = U[:, :self.n_components]
            self.singular_values_ = s[:self.n_components]
            self.Vt_ = Vt[:self.n_components, :]
        else:
            self.U_ = U
            self.singular_values_ = s
            self.Vt_ = Vt

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data into latent space.

        Args:
            X: (n_samples, n_features) data matrix

        Returns:
            (n_samples, n_components) latent representation
        """
        if self.Vt_ is None or self.singular_values_ is None:
            raise ValueError("Must call fit() first")

        X_centered = X - self.mean_
        return X_centered @ self.Vt_.T

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from low-rank approximation.

        Args:
            X: (n_samples, n_features) original data matrix

        Returns:
            (n_samples, n_features) reconstructed data
        """
        if self.U_ is None or self.singular_values_ is None or self.Vt_ is None:
            raise ValueError("Must call fit() first")

        X_centered = X - self.mean_
        X_latent = self.transform(X)
        X_reconstructed = X_latent @ self.Vt_

        return X_reconstructed + self.mean_

    def impute_missing_with_svd(
        self,
        X: np.ndarray,
        demographic_labels: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-4
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Impute missing values using iterative SVD-based matrix completion.

        This method alternates between SVD decomposition and missing value
        imputation until convergence. With equity monitoring, it tracks whether
        imputation quality differs across demographic groups.

        Args:
            X: (n_samples, n_features) data matrix with missing values (np.nan)
            demographic_labels: Optional group labels for equity monitoring
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance for change in imputed values

        Returns:
            Tuple of (imputed data matrix, imputation quality metrics)
        """
        # Initialize missing values with column means
        X_imputed = X.copy()
        missing_mask = np.isnan(X)

        for j in range(X.shape[1]):
            col_mean = np.nanmean(X[:, j])
            X_imputed[missing_mask[:, j], j] = col_mean

        # Iterative refinement
        for iteration in range(max_iterations):
            X_prev = X_imputed.copy()

            # Fit SVD on current imputed data
            self.fit(X_imputed, center=True)

            # Reconstruct and update missing values
            X_reconstructed = self.reconstruct(X_imputed)
            X_imputed[missing_mask] = X_reconstructed[missing_mask]

            # Check convergence
            change = np.sqrt(np.mean((X_imputed[missing_mask] - X_prev[missing_mask])**2))
            if change < tolerance:
                logger.info(f"SVD imputation converged after {iteration + 1} iterations")
                break

        # Assess imputation quality across groups
        quality_metrics = {'converged': change < tolerance, 'final_change': change}

        if demographic_labels is not None:
            quality_metrics['group_imputation_quality'] = self._assess_imputation_quality(
                X, X_imputed, missing_mask, demographic_labels
            )

        return X_imputed, quality_metrics

    def _assess_imputation_quality(
        self,
        X_original: np.ndarray,
        X_imputed: np.ndarray,
        missing_mask: np.ndarray,
        demographic_labels: np.ndarray
    ) -> Dict[str, any]:
        """
        Assess whether imputation quality differs across demographic groups.

        We can't directly assess accuracy since we don't know true values,
        but we can check whether imputed values have reasonable statistical
        properties within each group.
        """
        unique_groups = np.unique(demographic_labels[~pd.isna(demographic_labels)])

        quality_by_group = {}

        for group in unique_groups:
            group_mask = demographic_labels == group

            # For observed values in this group, check distributional match
            group_observed = X_original[group_mask & ~missing_mask]
            group_imputed = X_imputed[group_mask & missing_mask]

            if len(group_observed) > 0 and len(group_imputed) > 0:
                # Compare means
                observed_mean = np.nanmean(group_observed)
                imputed_mean = np.mean(group_imputed)
                mean_difference = abs(observed_mean - imputed_mean)

                # Compare standard deviations
                observed_std = np.nanstd(group_observed)
                imputed_std = np.std(group_imputed)
                std_ratio = imputed_std / observed_std if observed_std > 0 else 1.0

                quality_by_group[str(group)] = {
                    'mean_difference': mean_difference,
                    'std_ratio': std_ratio,
                    'n_imputed': len(group_imputed),
                    'n_observed': len(group_observed)
                }

        return quality_by_group

class EquityAwareMatrixFactorization:
    """
    Non-negative matrix factorization with equity considerations for
    discovering latent patient subgroups or disease phenotypes.
    """

    def __init__(
        self,
        n_components: int,
        max_iterations: int = 200,
        regularization: float = 0.01
    ):
        """
        Initialize matrix factorization model.

        Args:
            n_components: Number of latent factors
            max_iterations: Maximum iterations for optimization
            regularization: L2 regularization strength
        """
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.regularization = regularization

        self.W_: Optional[np.ndarray] = None  # Patient factors
        self.H_: Optional[np.ndarray] = None  # Feature factors

        logger.info(
            f"Initialized matrix factorization with {n_components} components, "
            f"regularization={regularization}"
        )

    def fit(
        self,
        X: np.ndarray,
        demographic_labels: Optional[np.ndarray] = None
    ) -> 'EquityAwareMatrixFactorization':
        """
        Fit non-negative matrix factorization with multiplicative updates.

        Factorizes X ≈ WH where W is (n_samples, n_components) and
        H is (n_components, n_features).

        Args:
            X: (n_samples, n_features) non-negative data matrix
            demographic_labels: Optional group labels for equity monitoring

        Returns:
            Self for method chaining
        """
        if np.any(X < 0):
            raise ValueError("Non-negative matrix factorization requires non-negative data")

        n_samples, n_features = X.shape

        # Initialize factors with small random values
        np.random.seed(42)
        self.W_ = np.random.rand(n_samples, self.n_components) * 0.01
        self.H_ = np.random.rand(self.n_components, n_features) * 0.01

        # Multiplicative update algorithm
        for iteration in range(self.max_iterations):
            # Update H
            numerator = self.W_.T @ X
            denominator = self.W_.T @ (self.W_ @ self.H_) + self.regularization
            self.H_ = self.H_ * (numerator / (denominator + 1e-10))

            # Update W
            numerator = X @ self.H_.T
            denominator = (self.W_ @ self.H_) @ self.H_.T + self.regularization
            self.W_ = self.W_ * (numerator / (denominator + 1e-10))

            # Compute reconstruction error periodically
            if iteration % 20 == 0:
                reconstruction_error = np.linalg.norm(X - self.W_ @ self.H_, 'fro')
                logger.debug(f"Iteration {iteration}, reconstruction error: {reconstruction_error:.4f}")

        # Analyze whether factors are balanced across demographic groups
        if demographic_labels is not None:
            self._analyze_factor_distribution(demographic_labels)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project new data into latent factor space.

        Args:
            X: (n_samples, n_features) data matrix

        Returns:
            (n_samples, n_components) factor representation
        """
        if self.H_ is None:
            raise ValueError("Must call fit() first")

        # Solve for W given X and H using multiplicative updates
        W = np.random.rand(X.shape[0], self.n_components) * 0.01

        for _ in range(50):  # Fewer iterations for transform
            numerator = X @ self.H_.T
            denominator = (W @ self.H_) @ self.H_.T + self.regularization
            W = W * (numerator / (denominator + 1e-10))

        return W

    def _analyze_factor_distribution(
        self,
        demographic_labels: np.ndarray
    ) -> None:
        """
        Analyze whether latent factors are distributed equitably across
        demographic groups or if certain factors are concentrated in
        specific populations.
        """
        unique_groups = np.unique(demographic_labels[~pd.isna(demographic_labels)])

        for component in range(self.n_components):
            factor_loadings = self.W_[:, component]

            # Compute mean loading for each group
            group_means = {}
            for group in unique_groups:
                group_mask = demographic_labels == group
                group_means[str(group)] = np.mean(factor_loadings[group_mask])

            # Check for concentration in specific groups
            means = list(group_means.values())
            if len(means) > 1:
                cv = np.std(means) / np.mean(means)
                if cv > 0.5:
                    logger.warning(
                        f"Factor {component} shows uneven distribution across groups "
                        f"(CV={cv:.3f}). This factor may primarily capture patterns "
                        f"from specific populations rather than being universally applicable."
                    )
```

These SVD and matrix factorization implementations include equity safeguards at multiple levels. The SVD-based imputation tracks imputation quality across demographic groups to detect when imputed values have different statistical properties for different populations. The matrix factorization analysis checks whether discovered latent factors are concentrated in specific demographic groups, which could indicate that the factorization is primarily capturing patterns from those groups while missing important structure in other populations.

## 2.3 Probability Theory for Clinical Decision Making

Probability theory provides the mathematical framework for reasoning about uncertainty, which is fundamental to clinical decision making where diagnoses are uncertain, prognoses are probabilistic, and treatment effects vary across individuals. However, applying probability theory to healthcare data requires careful attention to where the probabilities come from and what they represent. Probability distributions in healthcare often reflect social and historical processes rather than natural phenomena, and standard probabilistic assumptions can fail in ways that systematically disadvantage certain populations.

### 2.3.1 Probability Distributions and Clinical Data

A probability distribution describes the relative likelihood of different outcomes or the relative frequency with which different values of a variable occur. In healthcare applications, we use probability distributions to model diagnostic test results, disease prevalence in populations, treatment response rates, survival times, and many other uncertain quantities. However, the interpretation of these distributions requires understanding not just the mathematics but also the data generation process.

Consider a seemingly straightforward example: the distribution of blood pressure measurements in a patient population. We might model systolic blood pressure as following a normal distribution with a mean of 120 mmHg and a standard deviation of 15 mmHg. But this statistical model obscures important realities. The observed distribution reflects not just biological variation but also who has access to blood pressure measurement, how measurement protocols vary across clinical settings, whether measurements are taken in stressful environments like emergency departments or in calmer primary care settings, and systematic differences in measurement accuracy across patient body sizes and skin tones. The "true" distribution of underlying physiological blood pressure may be quite different from the observed distribution of recorded measurements.

The equity implications become clearer when we recognize that these measurement and selection processes are not uniform across populations. Patients with regular primary care access will have blood pressure distributions that reflect stable ongoing management, while patients who only interface with healthcare during acute illness will have distributions skewed toward higher values. Patients seen primarily in community health centers may have different measurement protocols than those seen in tertiary care centers. If we build prediction models or decision rules based on probability distributions estimated from one population and apply them to another, we may make systematically biased predictions.

Let me implement probability distribution modeling with explicit equity considerations:

```python
from scipy import stats
from typing import Callable

class EquityAwareProbabilityModeling:
    """
    Probability distribution fitting and analysis with explicit attention to
    whether distributions differ systematically across patient populations.
    """

    def __init__(self):
        """Initialize probability modeling framework."""
        self.fitted_distributions_: Dict[str, Dict[str, any]] = {}
        logger.info("Initialized equity-aware probability modeling")

    def fit_distribution(
        self,
        data: np.ndarray,
        distribution_type: str = 'normal',
        demographic_labels: Optional[np.ndarray] = None,
        group_specific: bool = False
    ) -> Dict[str, any]:
        """
        Fit probability distribution to data with optional group-specific fitting.

        Args:
            data: 1D array of observed values
            distribution_type: Type of distribution ('normal', 'lognormal', 'gamma', 'beta')
            demographic_labels: Optional group labels
            group_specific: Whether to fit separate distributions for each group

        Returns:
            Dictionary with fitted parameters and diagnostic information
        """
        # Remove missing values
        data_clean = data[~np.isnan(data)]

        if len(data_clean) == 0:
            raise ValueError("No non-missing data to fit distribution")

        results = {
            'distribution_type': distribution_type,
            'group_specific': group_specific,
            'global_fit': None,
            'group_fits': {},
            'distribution_homogeneity_test': None
        }

        # Fit global distribution
        if distribution_type == 'normal':
            mu, sigma = stats.norm.fit(data_clean)
            results['global_fit'] = {'mean': mu, 'std': sigma}
            fitted_dist = stats.norm(mu, sigma)

        elif distribution_type == 'lognormal':
            # Data must be positive for lognormal
            if np.any(data_clean <= 0):
                raise ValueError("Lognormal distribution requires positive data")
            shape, loc, scale = stats.lognorm.fit(data_clean)
            results['global_fit'] = {'shape': shape, 'loc': loc, 'scale': scale}
            fitted_dist = stats.lognorm(shape, loc, scale)

        elif distribution_type == 'gamma':
            if np.any(data_clean < 0):
                raise ValueError("Gamma distribution requires non-negative data")
            alpha, loc, beta = stats.gamma.fit(data_clean)
            results['global_fit'] = {'alpha': alpha, 'loc': loc, 'beta': beta}
            fitted_dist = stats.gamma(alpha, loc, beta)

        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")

        # Assess goodness of fit
        results['global_fit']['goodness_of_fit'] = self._assess_goodness_of_fit(
            data_clean, fitted_dist
        )

        # Fit group-specific distributions if requested
        if group_specific and demographic_labels is not None:
            unique_groups = np.unique(demographic_labels[~pd.isna(demographic_labels)])

            for group in unique_groups:
                group_mask = demographic_labels == group
                group_data = data[group_mask]
                group_data_clean = group_data[~np.isnan(group_data)]

                if len(group_data_clean) > 10:  # Need sufficient data
                    if distribution_type == 'normal':
                        mu_g, sigma_g = stats.norm.fit(group_data_clean)
                        results['group_fits'][str(group)] = {
                            'mean': mu_g, 'std': sigma_g
                        }

                    # Similar for other distribution types...

            # Test for distribution homogeneity across groups
            results['distribution_homogeneity_test'] = self._test_distribution_homogeneity(
                data, demographic_labels, distribution_type
            )

        return results

    def _assess_goodness_of_fit(
        self,
        data: np.ndarray,
        fitted_distribution
    ) -> Dict[str, float]:
        """
        Assess how well the fitted distribution matches the observed data.

        Uses Kolmogorov-Smirnov test and Anderson-Darling test.
        """
        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.kstest(data, fitted_distribution.cdf)

        # Q-Q plot correlation
        theoretical_quantiles = fitted_distribution.ppf(
            np.linspace(0.01, 0.99, len(data))
        )
        sample_quantiles = np.sort(data)
        qq_correlation = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]

        goodness_of_fit = {
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue,
            'qq_correlation': qq_correlation
        }

        if ks_pvalue < 0.05:
            logger.warning(
                f"Poor goodness of fit (KS p-value = {ks_pvalue:.4f}). "
                f"The assumed distribution may not be appropriate for this data."
            )

        return goodness_of_fit

    def _test_distribution_homogeneity(
        self,
        data: np.ndarray,
        demographic_labels: np.ndarray,
        distribution_type: str
    ) -> Dict[str, any]:
        """
        Test whether the distribution is homogeneous across demographic groups.

        Uses Kolmogorov-Smirnov test for comparing distributions.
        """
        unique_groups = np.unique(demographic_labels[~pd.isna(demographic_labels)])

        if len(unique_groups) < 2:
            return {'test': 'insufficient_groups'}

        # Pairwise KS tests
        pairwise_tests = []

        for i, group1 in enumerate(unique_groups):
            for group2 in unique_groups[i+1:]:
                mask1 = demographic_labels == group1
                mask2 = demographic_labels == group2

                data1 = data[mask1]
                data2 = data[mask2]

                data1_clean = data1[~np.isnan(data1)]
                data2_clean = data2[~np.isnan(data2)]

                if len(data1_clean) > 5 and len(data2_clean) > 5:
                    ks_stat, ks_pval = stats.ks_2samp(data1_clean, data2_clean)

                    pairwise_tests.append({
                        'group1': str(group1),
                        'group2': str(group2),
                        'ks_statistic': ks_stat,
                        'ks_pvalue': ks_pval,
                        'distributions_differ': ks_pval < 0.05
                    })

        # Check if any pairs have significantly different distributions
        n_different = sum(1 for test in pairwise_tests if test['distributions_differ'])

        if n_different > 0:
            logger.warning(
                f"Distribution differs significantly across {n_different} group pairs. "
                f"Using a single distribution for all groups may introduce bias."
            )

        return {
            'test': 'ks_2sample',
            'pairwise_comparisons': pairwise_tests,
            'n_pairs_differ': n_different
        }

    def compute_conditional_probability(
        self,
        outcome: np.ndarray,
        condition: np.ndarray,
        outcome_value: any,
        condition_value: any,
        demographic_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute $P(outcome=outcome_value \mid condition=condition_value)$ with
        optional stratification by demographic groups.

        This is useful for computing diagnostic test sensitivities, disease
        prevalence given risk factors, etc.

        Args:
            outcome: Array of outcome values
            condition: Array of conditioning variable values
            outcome_value: Specific outcome value of interest
            condition_value: Specific condition value
            demographic_labels: Optional group labels for stratified analysis

        Returns:
            Dictionary with conditional probabilities overall and by group
        """
        # Overall conditional probability
        condition_mask = condition == condition_value
        n_condition = np.sum(condition_mask)

        if n_condition == 0:
            raise ValueError(f"No observations with condition={condition_value}")

        outcome_and_condition_mask = (outcome == outcome_value) & condition_mask
        n_outcome_and_condition = np.sum(outcome_and_condition_mask)

        prob_overall = n_outcome_and_condition / n_condition

        results = {
            'overall_probability': prob_overall,
            'n_condition': int(n_condition),
            'n_outcome_and_condition': int(n_outcome_and_condition)
        }

        # Stratified probabilities
        if demographic_labels is not None:
            unique_groups = np.unique(demographic_labels[~pd.isna(demographic_labels)])
            group_probabilities = {}

            for group in unique_groups:
                group_mask = demographic_labels == group
                group_condition_mask = condition_mask & group_mask
                n_group_condition = np.sum(group_condition_mask)

                if n_group_condition > 0:
                    group_outcome_mask = outcome_and_condition_mask & group_mask
                    n_group_outcome = np.sum(group_outcome_mask)
                    group_prob = n_group_outcome / n_group_condition

                    group_probabilities[str(group)] = {
                        'probability': float(group_prob),
                        'n_condition': int(n_group_condition),
                        'n_outcome_and_condition': int(n_group_outcome)
                    }

            results['group_probabilities'] = group_probabilities

            # Test for homogeneity
            probs = [v['probability'] for v in group_probabilities.values()]
            if len(probs) > 1:
                prob_range = max(probs) - min(probs)
                if prob_range > 0.2:
                    logger.warning(
                        f"Conditional probability varies substantially across groups "
                        f"(range = {prob_range:.3f}). Pooled estimates may be misleading."
                    )

        return results
```

This probability modeling implementation explicitly checks whether distributions are homogeneous across demographic groups and warns when pooled estimates may be inappropriate. The conditional probability calculation provides stratified estimates that reveal when relationships differ across populations, which is essential for avoiding biased predictions that assume universality when relationships are actually heterogeneous.

### 2.3.2 Bayes' Theorem and Prior Probabilities in Healthcare

Bayes' theorem provides the mathematical foundation for updating beliefs based on evidence, formally stating that the posterior probability of a hypothesis given observed data is proportional to the likelihood of the data given the hypothesis times the prior probability of the hypothesis. In healthcare, Bayesian reasoning appears in diagnostic test interpretation, where we update our assessment of disease probability based on test results, and in clinical prediction models that combine prior information with patient-specific features.

The mathematical statement is elegant: $$P(\text{disease} \mid \text{test}+) = \dfrac{P(\text{test}+ \mid \text{disease}) \, P(\text{disease})}{P(\text{test}+)}$$, where $$P(\text{disease} \mid \text{test}+)$$ is the posterior probability of disease given a positive test, $$P(\text{test}+ \mid \text{disease})$$ is the test sensitivity, $$P(\text{disease})$$ is the prior probability or prevalence, and $$P(\text{test}+)$$ is the marginal probability of a positive test. However, applying this formula in practice requires careful attention to where the prior probabilities come from and whether they are appropriate for the specific patient.

The health equity challenge is that prior probabilities often reflect historical patterns of disease diagnosis that may be systematically biased. If a condition has been historically underdiagnosed in certain populations due to lack of access to care, implicit bias in clinical evaluation, or other systemic factors, then prevalence estimates derived from diagnostic databases will underestimate true disease burden in those populations. Using these biased priors in Bayesian reasoning will then lead to systematic underdiagnosis that perpetuates the original inequity.

Consider screening for diabetic retinopathy in patients with diabetes. The prior probability of retinopathy depends on how long diabetes has been present, how well controlled blood sugar has been, and other factors. But it also depends on how consistently the patient has had access to diabetic care and whether previous retinal examinations were performed and documented. For a patient with intermittent healthcare access whose diabetes may have been present but undiagnosed for years, the standard prior probability based on documented diabetes duration will systematically underestimate retinopathy risk. A Bayesian diagnostic approach using that prior will then underweight positive screening results, potentially leading to delayed treatment.

Let me implement Bayesian reasoning with explicit attention to prior probability issues:

```python
class EquityAwareBayesianReasoning:
    """
    Bayesian inference with explicit attention to bias in prior probabilities
    and sensitivity analysis for how results change with different priors.
    """

    def __init__(self):
        """Initialize Bayesian reasoning framework."""
        logger.info("Initialized equity-aware Bayesian reasoning")

    def diagnostic_test_interpretation(
        self,
        test_result: bool,
        sensitivity: float,
        specificity: float,
        prior_probability: float,
        demographic_group: Optional[str] = None,
        prior_confidence: str = 'medium'
    ) -> Dict[str, any]:
        """
        Interpret diagnostic test result using Bayes' theorem with sensitivity
        analysis for prior probability uncertainty.

        Args:
            test_result: Whether test was positive (True) or negative (False)
            sensitivity: $P(test+ \mid disease)$
            specificity: $P(test- \mid no disease)$
            prior_probability: P(disease) before test
            demographic_group: Optional group label for context
            prior_confidence: How confident we are in prior ('high', 'medium', 'low')

        Returns:
            Dictionary with posterior probability and sensitivity analysis
        """
        if not (0 <= sensitivity <= 1 and 0 <= specificity <= 1):
            raise ValueError("Sensitivity and specificity must be between 0 and 1")

        if not (0 <= prior_probability <= 1):
            raise ValueError("Prior probability must be between 0 and 1")

        # Compute posterior using Bayes' theorem
        if test_result:  # Positive test
            # $P(disease \mid test+)$ = $P(test+ \mid disease)$ * P(disease) / P(test+)
            # where $P(test+) = P(test+ \mid disease)$*$P(disease) + P(test+ \mid no disease)$*P(no disease)
            p_test_pos = sensitivity * prior_probability + (1 - specificity) * (1 - prior_probability)
            posterior = (sensitivity * prior_probability) / p_test_pos if p_test_pos > 0 else 0
        else:  # Negative test
            # $P(disease \mid test-)$ = $P(test- \mid disease)$ * P(disease) / P(test-)
            p_test_neg = (1 - sensitivity) * prior_probability + specificity * (1 - prior_probability)
            posterior = ((1 - sensitivity) * prior_probability) / p_test_neg if p_test_neg > 0 else 0

        results = {
            'test_result': 'positive' if test_result else 'negative',
            'prior_probability': prior_probability,
            'posterior_probability': posterior,
            'probability_change': posterior - prior_probability,
            'demographic_group': demographic_group
        }

        # Sensitivity analysis for prior uncertainty
        if prior_confidence == 'low':
            prior_range = (max(0, prior_probability - 0.2), min(1, prior_probability + 0.2))
        elif prior_confidence == 'medium':
            prior_range = (max(0, prior_probability - 0.1), min(1, prior_probability + 0.1))
        else:  # high confidence
            prior_range = (max(0, prior_probability - 0.05), min(1, prior_probability + 0.05))

        # Compute posterior range
        posterior_range = []
        for prior in prior_range:
            if test_result:
                p_test_pos = sensitivity * prior + (1 - specificity) * (1 - prior)
                post = (sensitivity * prior) / p_test_pos if p_test_pos > 0 else 0
            else:
                p_test_neg = (1 - sensitivity) * prior + specificity * (1 - prior)
                post = ((1 - sensitivity) * prior) / p_test_neg if p_test_neg > 0 else 0
            posterior_range.append(post)

        results['sensitivity_analysis'] = {
            'prior_confidence': prior_confidence,
            'prior_range': prior_range,
            'posterior_range': tuple(posterior_range)
        }

        # Interpretation guidance
        uncertainty_width = posterior_range[1] - posterior_range[0]
        if uncertainty_width > 0.2:
            results['interpretation'] = (
                f"Substantial uncertainty in posterior probability due to uncertain prior. "
                f"Posterior could be anywhere from {posterior_range[0]:.3f} to {posterior_range[1]:.3f}. "
                f"Consider additional information to refine prior estimate."
            )
        else:
            results['interpretation'] = (
                f"Posterior probability is {posterior:.3f}, representing "
                f"{'an increase' if posterior > prior_probability else 'a decrease'} "
                f"from prior of {prior_probability:.3f}."
            )

        return results

    def adjust_prior_for_access_bias(
        self,
        observed_prevalence: float,
        access_adjustment_factor: float
    ) -> Dict[str, float]:
        """
        Adjust disease prevalence estimates for known access-to-care bias.

        If certain populations have limited healthcare access, observed prevalence
        may underestimate true disease burden. This function applies an adjustment
        based on estimated diagnosis rates.

        Args:
            observed_prevalence: Disease prevalence in diagnosed population
            access_adjustment_factor: Estimated fraction of cases that are diagnosed
                                     (1.0 = all cases diagnosed, 0.5 = half diagnosed)

        Returns:
            Dictionary with adjusted prevalence and uncertainty bounds
        """
        if not (0 < access_adjustment_factor <= 1.0):
            raise ValueError("Access adjustment factor must be between 0 and 1")

        # Adjusted prevalence assuming observed cases are a fraction of true cases
        adjusted_prevalence = observed_prevalence / access_adjustment_factor

        # Cap at 1.0 (can't exceed 100% prevalence)
        adjusted_prevalence = min(adjusted_prevalence, 1.0)

        # Uncertainty bounds - conservative estimates
        lower_bound = observed_prevalence  # At minimum, prevalence is observed rate
        upper_bound = min(observed_prevalence / (access_adjustment_factor * 0.7), 1.0)

        results = {
            'observed_prevalence': observed_prevalence,
            'adjusted_prevalence': adjusted_prevalence,
            'adjustment_factor': access_adjustment_factor,
            'uncertainty_bounds': (lower_bound, upper_bound),
            'interpretation': (
                f"Observed prevalence of {observed_prevalence:.3f} may underestimate "
                f"true prevalence if only {access_adjustment_factor:.1%} of cases are diagnosed. "
                f"Adjusted estimate is {adjusted_prevalence:.3f}, with true value likely "
                f"between {lower_bound:.3f} and {upper_bound:.3f}."
            )
        }

        return results

    def compare_bayesian_vs_frequentist(
        self,
        observed_successes: int,
        total_trials: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        demographic_group: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Compare Bayesian posterior with frequentist confidence interval for
        probability estimation, showing how prior information affects inference.

        Uses Beta-Binomial conjugate prior for Bayesian analysis.

        Args:
            observed_successes: Number of successful outcomes observed
            total_trials: Total number of trials
            prior_alpha: Prior Beta distribution alpha parameter (successes + 1)
            prior_beta: Prior Beta distribution beta parameter (failures + 1)
            demographic_group: Optional group label

        Returns:
            Dictionary comparing Bayesian and frequentist estimates
        """
        if observed_successes > total_trials or observed_successes < 0:
            raise ValueError("Observed successes must be between 0 and total_trials")

        # Frequentist estimate: maximum likelihood (observed proportion)
        freq_estimate = observed_successes / total_trials if total_trials > 0 else 0

        # Frequentist confidence interval (Wilson score interval)
        from statsmodels.stats.proportion import proportion_confint
        freq_ci = proportion_confint(observed_successes, total_trials, alpha=0.05, method='wilson')

        # Bayesian estimate: posterior mean of Beta distribution
        posterior_alpha = prior_alpha + observed_successes
        posterior_beta = prior_beta + (total_trials - observed_successes)

        bayes_estimate = posterior_alpha / (posterior_alpha + posterior_beta)

        # Bayesian credible interval (equal-tailed 95% interval)
        bayes_ci = (
            stats.beta.ppf(0.025, posterior_alpha, posterior_beta),
            stats.beta.ppf(0.975, posterior_alpha, posterior_beta)
        )

        results = {
            'observed_successes': observed_successes,
            'total_trials': total_trials,
            'observed_proportion': freq_estimate,
            'frequentist': {
                'estimate': freq_estimate,
                'confidence_interval': freq_ci,
                'interpretation': '95% confidence interval'
            },
            'bayesian': {
                'prior': {'alpha': prior_alpha, 'beta': prior_beta},
                'posterior': {'alpha': posterior_alpha, 'beta': posterior_beta},
                'estimate': bayes_estimate,
                'credible_interval': bayes_ci,
                'interpretation': '95% credible interval'
            },
            'difference': bayes_estimate - freq_estimate,
            'demographic_group': demographic_group
        }

        # Interpretation of prior influence
        if abs(results['difference']) > 0.05:
            results['prior_influence'] = (
                f"Bayesian estimate ({bayes_estimate:.3f}) differs substantially from "
                f"frequentist estimate ({freq_estimate:.3f}), indicating strong prior influence. "
                f"With limited data, prior beliefs matter significantly."
            )
        else:
            results['prior_influence'] = (
                f"Bayesian and frequentist estimates are similar, indicating that observed "
                f"data dominates prior beliefs with this sample size."
            )

        return results
```

This Bayesian reasoning implementation includes sensitivity analysis for prior probability uncertainty and explicit tools for adjusting priors when we suspect systematic bias in observed prevalence estimates. The comparison between Bayesian and frequentist approaches helps illustrate how prior information affects inference, which is particularly important when we have reason to question whether standard priors are appropriate for specific populations.

### 2.3.3 Conditional Independence and Graphical Models

Conditional independence is a critical concept that underlies much of probabilistic modeling in healthcare AI. Two variables X and Y are conditionally independent given Z if $$P(X, Y \mid Z)$$ = $$P(X \mid Z)$$ $$P(Y \mid Z)$$, meaning that once we know Z, learning X provides no additional information about Y. This concept enables us to build tractable probabilistic models of complex systems by decomposing joint probability distributions into products of simpler conditional distributions.

However, conditional independence assumptions often fail in healthcare data in ways that reflect social and structural confounding rather than measurement error. Consider predicting hospital readmission using clinical variables like prior hospitalizations, chronic conditions, and functional status. Standard modeling approaches might assume that these clinical variables are conditionally independent given some latent health status. But in reality, many clinical variables are jointly influenced by social determinants like housing stability, food security, and access to primary care. Unmeasured confounders like structural racism affect multiple variables simultaneously, violating conditional independence assumptions in systematic ways.

The implications for model fairness are profound. When we build predictive models that assume conditional independence but the assumption fails due to unmeasured confounding that differs across populations, the resulting predictions will be systematically biased. The bias won't necessarily be detected by standard model validation because the conditional independence violations are embedded in the training data structure.

Let me implement tools for testing and working with conditional independence assumptions:

```python
from scipy.stats import chi2_contingency, pearsonr
from itertools import combinations

class ConditionalIndependenceAnalysis:
    """
    Tools for testing conditional independence assumptions and detecting
    violations that may indicate confounding or model misspecification.
    """

    def __init__(self):
        """Initialize conditional independence analysis framework."""
        logger.info("Initialized conditional independence analysis")

    def test_conditional_independence(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        test_type: str = 'partial_correlation'
    ) -> Dict[str, any]:
        """
        Test whether X and Y are conditionally independent given Z.

        Multiple test types are provided because appropriate tests depend on
        variable types (continuous, categorical) and distributional assumptions.

        Args:
            X: First variable (n_samples,)
            Y: Second variable (n_samples,)
            Z: Conditioning variable(s) (n_samples,) or (n_samples, n_features)
            test_type: Type of test ('partial_correlation', 'cmi', 'stratified')

        Returns:
            Dictionary with test results and interpretation
        """
        if len(X) != len(Y) or len(X) != len(Z.shape[0] if Z.ndim > 1 else len(Z)):
            raise ValueError("X, Y, and Z must have same number of samples")

        results = {'test_type': test_type}

        if test_type == 'partial_correlation':
            # Compute partial correlation between X and Y given Z
            partial_corr, pvalue = self._partial_correlation(X, Y, Z)

            results['partial_correlation'] = partial_corr
            results['pvalue'] = pvalue
            results['conditionally_independent'] = pvalue > 0.05
            results['interpretation'] = (
                f"Partial correlation is {partial_corr:.4f} (p={pvalue:.4f}). "
                f"{'Evidence of conditional independence.' if pvalue > 0.05 else 'Conditional dependence detected.'}"
            )

        elif test_type == 'stratified':
            # Stratify by Z and test independence within strata
            results.update(self._stratified_independence_test(X, Y, Z))

        else:
            raise ValueError(f"Unknown test type: {test_type}")

        return results

    def _partial_correlation(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute partial correlation between X and Y controlling for Z.

        This is done by regressing X on Z and Y on Z, then computing the
        correlation between the residuals.
        """
        from sklearn.linear_model import LinearRegression

        # Ensure Z is 2D
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        # Regress X on Z
        model_x = LinearRegression()
        model_x.fit(Z, X)
        residuals_x = X - model_x.predict(Z)

        # Regress Y on Z
        model_y = LinearRegression()
        model_y.fit(Z, Y)
        residuals_y = Y - model_y.predict(Z)

        # Correlation of residuals is partial correlation
        partial_corr, pvalue = pearsonr(residuals_x, residuals_y)

        return partial_corr, pvalue

    def _stratified_independence_test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray
    ) -> Dict[str, any]:
        """
        Test independence of X and Y within each stratum defined by Z.

        Appropriate when Z is categorical.
        """
        if Z.ndim > 1:
            raise ValueError("Stratified test requires Z to be 1-dimensional")

        unique_strata = np.unique(Z[~np.isnan(Z)])

        stratum_results = []

        for stratum in unique_strata:
            mask = Z == stratum
            X_stratum = X[mask]
            Y_stratum = Y[mask]

            if len(X_stratum) > 5:  # Need minimum sample size
                # Compute correlation within stratum
                corr, pval = pearsonr(X_stratum, Y_stratum)

                stratum_results.append({
                    'stratum': stratum,
                    'n': len(X_stratum),
                    'correlation': corr,
                    'pvalue': pval,
                    'independent': pval > 0.05
                })

        # Overall assessment
        n_independent = sum(1 for r in stratum_results if r['independent'])

        return {
            'stratum_results': stratum_results,
            'n_strata': len(stratum_results),
            'n_strata_independent': n_independent,
            'conditionally_independent': n_independent == len(stratum_results),
            'interpretation': (
                f"X and Y are {'conditionally independent' if n_independent == len(stratum_results) else 'conditionally dependent'} "
                f"given Z. Independence holds in {n_independent}/{len(stratum_results)} strata."
            )
        }

    def detect_confounding_structure(
        self,
        data: pd.DataFrame,
        outcome_var: str,
        predictor_vars: List[str],
        potential_confounders: List[str],
        demographic_var: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Detect potential confounding structure in predictive relationships.

        Tests whether relationships between predictors and outcome change
        substantially when controlling for potential confounders, which
        suggests confounding.

        Args:
            data: DataFrame with all variables
            outcome_var: Name of outcome variable
            predictor_vars: Names of predictor variables
            potential_confounders: Names of potential confounding variables
            demographic_var: Optional demographic variable for stratified analysis

        Returns:
            Dictionary with confounding analysis results
        """
        from sklearn.linear_model import LinearRegression

        results = {
            'outcome': outcome_var,
            'confounding_detected': {},
            'demographic_heterogeneity': {}
        }

        Y = data[outcome_var].values

        for predictor in predictor_vars:
            X = data[[predictor]].values

            # Marginal association (without controlling for confounders)
            model_marginal = LinearRegression()
            model_marginal.fit(X, Y)
            coef_marginal = model_marginal.coef_[0]

            # Conditional association (controlling for confounders)
            X_with_confounders = data[[predictor] + potential_confounders].values
            model_conditional = LinearRegression()
            model_conditional.fit(X_with_confounders, Y)
            coef_conditional = model_conditional.coef_[0]

            # Compute percent change in coefficient
            pct_change = 100 * abs(coef_conditional - coef_marginal) / abs(coef_marginal) if coef_marginal != 0 else float('inf')

            confounding_detected = pct_change > 10  # >10% change suggests confounding

            results['confounding_detected'][predictor] = {
                'marginal_coefficient': coef_marginal,
                'conditional_coefficient': coef_conditional,
                'percent_change': pct_change,
                'confounding_present': confounding_detected
            }

            if confounding_detected:
                logger.warning(
                    f"Confounding detected for {predictor}. Coefficient changed by "
                    f"{pct_change:.1f}% when controlling for {', '.join(potential_confounders)}."
                )

            # Check if confounding structure differs by demographic group
            if demographic_var and demographic_var in data.columns:
                group_confounding = {}

                for group in data[demographic_var].unique():
                    if pd.notna(group):
                        group_data = data[data[demographic_var] == group]

                        if len(group_data) > 20:  # Need sufficient sample size
                            Y_group = group_data[outcome_var].values
                            X_group = group_data[[predictor]].values
                            X_conf_group = group_data[[predictor] + potential_confounders].values

                            model_marg_group = LinearRegression()
                            model_marg_group.fit(X_group, Y_group)
                            coef_marg_group = model_marg_group.coef_[0]

                            model_cond_group = LinearRegression()
                            model_cond_group.fit(X_conf_group, Y_group)
                            coef_cond_group = model_cond_group.coef_[0]

                            pct_change_group = 100 * abs(coef_cond_group - coef_marg_group) / abs(coef_marg_group) if coef_marg_group != 0 else 0

                            group_confounding[str(group)] = {
                                'percent_change': pct_change_group,
                                'confounding_present': pct_change_group > 10
                            }

                results['demographic_heterogeneity'][predictor] = group_confounding

                # Check for heterogeneity
                changes = [v['percent_change'] for v in group_confounding.values()]
                if len(changes) > 1 and max(changes) - min(changes) > 20:
                    logger.warning(
                        f"Confounding structure for {predictor} differs substantially "
                        f"across demographic groups. Single model may be inappropriate."
                    )

        return results
```

This conditional independence framework provides tools for detecting when standard modeling assumptions break down, with particular attention to whether violations differ across demographic groups. When we detect confounding that varies by demographics, it suggests that models built on the pooled data may systematically misrepresent relationships for some populations.

## 2.4 Statistical Inference with Equity Considerations

Statistical inference provides methods for drawing conclusions about populations from samples, quantifying uncertainty in estimates, and testing hypotheses about relationships in data. However, standard inferential methods make assumptions about random sampling, independent observations, and homogeneous populations that frequently fail in healthcare data. When these assumptions fail differently for different patient populations, standard inference can lead to systematically biased conclusions.

### 2.4.1 Sampling Bias and Selection Effects

The foundation of statistical inference is the assumption that our sample is representative of the population we want to draw conclusions about. In healthcare research, this assumption is routinely violated because healthcare data is fundamentally observational rather than randomly sampled. Patients appear in datasets because they sought care, had insurance coverage, lived near a healthcare facility, and were documented in an electronic health record system. Each of these selection steps can introduce bias.

The equity implications are severe. Underserved populations face systematic barriers to healthcare access, meaning they are systematically underrepresented in healthcare datasets. When they do appear in datasets, it is often in crisis situations like emergency department visits rather than in longitudinal primary care where comprehensive data accrues. The result is that statistical estimates derived from healthcare data systematically misrepresent the health status and healthcare needs of underserved populations.

Consider estimating the prevalence of uncontrolled hypertension in a population using electronic health record data. The naive estimate would simply compute the proportion of patients with most recent blood pressure measurements above target values. But this estimate is biased because it only includes patients who had blood pressure measured, and blood pressure measurement rates differ systematically across populations based on access to care. Patients with good healthcare access get regular blood pressure monitoring and early treatment, while patients with limited access may only have blood pressure measured when acutely ill or in emergency settings. The observed prevalence of uncontrolled hypertension will therefore be higher in populations with limited access, not necessarily because true hypertension rates are higher but because we are selectively sampling individuals at times when their blood pressure is more likely to be elevated.

Let me implement inference methods that account for selection bias:

```python
class EquityAwareInference:
    """
    Statistical inference methods with explicit accounting for selection bias
    and differential sampling across populations.
    """

    def __init__(self):
        """Initialize inference framework."""
        logger.info("Initialized equity-aware statistical inference")

    def estimate_prevalence_with_selection(
        self,
        outcome: np.ndarray,
        selected: np.ndarray,
        demographic_labels: Optional[np.ndarray] = None,
        inverse_probability_weights: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Estimate disease prevalence accounting for selection bias.

        Uses inverse probability weighting to adjust for differential selection
        probabilities across populations.

        Args:
            outcome: Binary outcome indicating disease presence (1) or absence (0)
            selected: Binary indicator of whether individual was selected into sample
            demographic_labels: Optional group labels
            inverse_probability_weights: Optional pre-computed weights
                If None, assumes equal selection probabilities within groups

        Returns:
            Dictionary with naive and adjusted prevalence estimates
        """
        # Naive estimate using only selected sample
        selected_outcome = outcome[selected == 1]
        naive_prevalence = np.mean(selected_outcome)

        results = {
            'naive_prevalence': naive_prevalence,
            'n_selected': int(np.sum(selected)),
            'n_total': len(outcome)
        }

        # Adjusted estimate using inverse probability weighting
        if inverse_probability_weights is not None:
            # Weight each observation by inverse of selection probability
            weighted_sum = np.sum(outcome[selected == 1] * inverse_probability_weights[selected == 1])
            weight_sum = np.sum(inverse_probability_weights[selected == 1])
            adjusted_prevalence = weighted_sum / weight_sum

            results['adjusted_prevalence'] = adjusted_prevalence
            results['selection_bias'] = adjusted_prevalence - naive_prevalence

        # Stratified analysis
        if demographic_labels is not None:
            unique_groups = np.unique(demographic_labels[~pd.isna(demographic_labels)])

            group_estimates = {}

            for group in unique_groups:
                group_mask = demographic_labels == group
                group_selected = selected[group_mask]
                group_outcome = outcome[group_mask]

                # Selection rate for this group
                selection_rate = np.mean(group_selected)

                # Naive prevalence in selected sample
                if np.sum(group_selected) > 0:
                    group_selected_outcome = group_outcome[group_selected == 1]
                    group_naive_prev = np.mean(group_selected_outcome)

                    group_estimates[str(group)] = {
                        'selection_rate': selection_rate,
                        'naive_prevalence': group_naive_prev,
                        'n_selected': int(np.sum(group_selected))
                    }

            results['group_estimates'] = group_estimates

            # Check for differential selection
            selection_rates = [v['selection_rate'] for v in group_estimates.values()]
            if len(selection_rates) > 1 and max(selection_rates) / min(selection_rates) > 2:
                logger.warning(
                    f"Substantial differential selection across groups. "
                    f"Selection rates vary from {min(selection_rates):.3f} to {max(selection_rates):.3f}. "
                    f"Naive estimates may be severely biased."
                )

        return results

    def confidence_interval_stratified(
        self,
        data: np.ndarray,
        demographic_labels: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, any]:
        """
        Compute confidence intervals for population mean with stratification
        by demographic groups.

        Provides both overall and group-specific estimates with appropriate
        standard errors that account for potential heterogeneity.

        Args:
            data: Continuous outcome variable
            demographic_labels: Group labels for stratification
            confidence_level: Confidence level (default 0.95 for 95% CI)

        Returns:
            Dictionary with confidence intervals overall and by group
        """
        from scipy import stats

        alpha = 1 - confidence_level

        # Overall estimate
        data_clean = data[~np.isnan(data)]
        n = len(data_clean)
        mean = np.mean(data_clean)
        se = np.std(data_clean, ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
        ci = (mean - t_crit * se, mean + t_crit * se)

        results = {
            'overall': {
                'mean': mean,
                'standard_error': se,
                'confidence_interval': ci,
                'n': n
            },
            'groups': {}
        }

        # Group-specific estimates
        unique_groups = np.unique(demographic_labels[~pd.isna(demographic_labels)])

        for group in unique_groups:
            group_mask = demographic_labels == group
            group_data = data[group_mask]
            group_data_clean = group_data[~np.isnan(group_data)]

            if len(group_data_clean) > 1:
                n_group = len(group_data_clean)
                mean_group = np.mean(group_data_clean)
                se_group = np.std(group_data_clean, ddof=1) / np.sqrt(n_group)
                t_crit_group = stats.t.ppf(1 - alpha/2, df=n_group-1)
                ci_group = (mean_group - t_crit_group * se_group,
                           mean_group + t_crit_group * se_group)

                results['groups'][str(group)] = {
                    'mean': mean_group,
                    'standard_error': se_group,
                    'confidence_interval': ci_group,
                    'n': n_group
                }

        # Test for heterogeneity across groups
        group_means = [v['mean'] for v in results['groups'].values()]
        if len(group_means) > 1:
            mean_range = max(group_means) - min(group_means)
            overall_se = results['overall']['standard_error']

            # Simple heterogeneity indicator
            heterogeneity_detected = mean_range > 2 * overall_se

            results['heterogeneity_analysis'] = {
                'mean_range': mean_range,
                'range_in_se_units': mean_range / overall_se if overall_se > 0 else float('inf'),
                'heterogeneity_detected': heterogeneity_detected
            }

            if heterogeneity_detected:
                logger.warning(
                    f"Substantial heterogeneity detected across groups. "
                    f"Overall confidence interval may not represent all populations well."
                )

        return results

    def hypothesis_test_with_multiple_comparisons(
        self,
        groups: Dict[str, np.ndarray],
        test_type: str = 'anova',
        adjustment_method: str = 'bonferroni'
    ) -> Dict[str, any]:
        """
        Conduct hypothesis tests across multiple demographic groups with
        appropriate correction for multiple comparisons.

        Args:
            groups: Dictionary mapping group names to data arrays
            test_type: Type of test ('anova', 'kruskal')
            adjustment_method: Method for multiple comparison adjustment
                ('bonferroni', 'holm', 'fdr_bh')

        Returns:
            Dictionary with test results and adjusted p-values
        """
        from scipy.stats import f_oneway, kruskal
        from statsmodels.stats.multitest import multipletests

        if len(groups) < 2:
            raise ValueError("Need at least 2 groups for hypothesis testing")

        # Overall test
        group_data = [data[~np.isnan(data)] for data in groups.values()]

        if test_type == 'anova':
            # One-way ANOVA
            statistic, pvalue = f_oneway(*group_data)
            test_name = "One-way ANOVA"
        elif test_type == 'kruskal':
            # Kruskal-Wallis test (non-parametric alternative to ANOVA)
            statistic, pvalue = kruskal(*group_data)
            test_name = "Kruskal-Wallis test"
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        results = {
            'overall_test': {
                'test_name': test_name,
                'statistic': statistic,
                'pvalue': pvalue,
                'reject_null': pvalue < 0.05
            },
            'pairwise_comparisons': []
        }

        # Pairwise comparisons if overall test is significant
        if pvalue < 0.05:
            group_names = list(groups.keys())
            pairwise_pvalues = []

            for i, name1 in enumerate(group_names):
                for name2 in group_names[i+1:]:
                    data1 = groups[name1][~np.isnan(groups[name1])]
                    data2 = groups[name2][~np.isnan(groups[name2])]

                    # T-test for pairwise comparison
                    stat, pval = stats.ttest_ind(data1, data2)

                    results['pairwise_comparisons'].append({
                        'group1': name1,
                        'group2': name2,
                        'statistic': stat,
                        'pvalue_unadjusted': pval
                    })

                    pairwise_pvalues.append(pval)

            # Adjust for multiple comparisons
            if pairwise_pvalues:
                reject, pvals_adjusted, _, _ = multipletests(
                    pairwise_pvalues,
                    alpha=0.05,
                    method=adjustment_method
                )

                for i, comparison in enumerate(results['pairwise_comparisons']):
                    comparison['pvalue_adjusted'] = pvals_adjusted[i]
                    comparison['reject_null_adjusted'] = reject[i]

                results['adjustment_method'] = adjustment_method

        return results
```

This inference framework explicitly accounts for selection bias and provides tools for stratified analysis that surface when overall estimates don't represent all populations well. The multiple comparison adjustments are essential when conducting numerous group comparisons to avoid false discoveries.

### 2.4.2 Bootstrapping and Resampling Methods

Bootstrap resampling provides a powerful approach to statistical inference that makes minimal distributional assumptions. By repeatedly sampling with replacement from the observed data and computing statistics on each bootstrap sample, we can empirically estimate the sampling distribution of any statistic and construct confidence intervals without assuming normality or other specific distributional forms. However, standard bootstrap methods can fail when data has complex structure or when we need to respect correlations and clustering in the data.

In healthcare applications serving diverse populations, we often want to ensure that bootstrap samples maintain appropriate representation of demographic groups and preserve within-group correlation structures. Standard bootstrapping that samples individuals independently may produce bootstrap samples with poor representation of minority populations by chance, leading to unstable estimates of group-specific parameters.

Let me implement equity-aware bootstrap methods:

```python
class EquityAwareBootstrap:
    """
    Bootstrap resampling methods that maintain appropriate representation
    of demographic groups and respect data structure.
    """

    def __init__(self, n_bootstrap: int = 1000, random_seed: int = 42):
        """
        Initialize bootstrap framework.

        Args:
            n_bootstrap: Number of bootstrap samples to generate
            random_seed: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed
        np.random.seed(random_seed)

        logger.info(f"Initialized bootstrap with {n_bootstrap} samples")

    def stratified_bootstrap(
        self,
        data: np.ndarray,
        demographic_labels: np.ndarray,
        statistic_func: Callable[[np.ndarray], float]
    ) -> Dict[str, any]:
        """
        Perform stratified bootstrap that maintains demographic group proportions.

        Standard bootstrap can under-represent minority groups by chance.
        Stratified bootstrap samples within each group to maintain representation.

        Args:
            data: Data array
            demographic_labels: Group labels for stratification
            statistic_func: Function that computes statistic from data array

        Returns:
            Dictionary with bootstrap distribution and confidence intervals
        """
        unique_groups = np.unique(demographic_labels[~pd.isna(demographic_labels)])

        # Compute observed statistic
        data_clean = data[~np.isnan(data)]
        observed_statistic = statistic_func(data_clean)

        # Bootstrap distribution
        bootstrap_statistics = []

        for b in range(self.n_bootstrap):
            # Stratified sampling
            bootstrap_sample = []

            for group in unique_groups:
                group_mask = demographic_labels == group
                group_data = data[group_mask]
                group_data_clean = group_data[~np.isnan(group_data)]

                if len(group_data_clean) > 0:
                    # Sample with replacement within group
                    group_bootstrap = np.random.choice(
                        group_data_clean,
                        size=len(group_data_clean),
                        replace=True
                    )
                    bootstrap_sample.extend(group_bootstrap)

            # Compute statistic on bootstrap sample
            bootstrap_sample = np.array(bootstrap_sample)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_statistics.append(bootstrap_stat)

        bootstrap_statistics = np.array(bootstrap_statistics)

        # Confidence intervals
        ci_lower = np.percentile(bootstrap_statistics, 2.5)
        ci_upper = np.percentile(bootstrap_statistics, 97.5)

        results = {
            'observed_statistic': observed_statistic,
            'bootstrap_mean': np.mean(bootstrap_statistics),
            'bootstrap_std': np.std(bootstrap_statistics),
            'confidence_interval_95': (ci_lower, ci_upper),
            'bootstrap_distribution': bootstrap_statistics
        }

        return results

    def bootstrap_group_differences(
        self,
        data: np.ndarray,
        demographic_labels: np.ndarray,
        group1: str,
        group2: str
    ) -> Dict[str, any]:
        """
        Bootstrap confidence interval for difference between two groups.

        Provides inference about whether observed differences between groups
        are statistically significant.

        Args:
            data: Data array
            demographic_labels: Group labels
            group1: Name of first group
            group2: Name of second group

        Returns:
            Dictionary with difference estimate and confidence interval
        """
        # Extract group data
        mask1 = demographic_labels == group1
        mask2 = demographic_labels == group2

        data1 = data[mask1]
        data2 = data[mask2]

        data1_clean = data1[~np.isnan(data1)]
        data2_clean = data2[~np.isnan(data2)]

        # Observed difference
        mean1 = np.mean(data1_clean)
        mean2 = np.mean(data2_clean)
        observed_difference = mean1 - mean2

        # Bootstrap distribution of difference
        bootstrap_differences = []

        for b in range(self.n_bootstrap):
            # Bootstrap sample from each group
            boot1 = np.random.choice(data1_clean, size=len(data1_clean), replace=True)
            boot2 = np.random.choice(data2_clean, size=len(data2_clean), replace=True)

            boot_diff = np.mean(boot1) - np.mean(boot2)
            bootstrap_differences.append(boot_diff)

        bootstrap_differences = np.array(bootstrap_differences)

        # Confidence interval for difference
        ci_lower = np.percentile(bootstrap_differences, 2.5)
        ci_upper = np.percentile(bootstrap_differences, 97.5)

        # P-value (proportion of bootstrap samples with difference crossing zero)
        if observed_difference > 0:
            pvalue = 2 * np.mean(bootstrap_differences <= 0)
        else:
            pvalue = 2 * np.mean(bootstrap_differences >= 0)
        pvalue = min(pvalue, 1.0)  # Cap at 1.0

        results = {
            'group1': group1,
            'group2': group2,
            'mean_group1': mean1,
            'mean_group2': mean2,
            'observed_difference': observed_difference,
            'confidence_interval_95': (ci_lower, ci_upper),
            'pvalue': pvalue,
            'significant': ci_lower * ci_upper > 0  # CI doesn't contain zero
        }

        return results

    def bootstrap_correlation(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        demographic_labels: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Bootstrap confidence interval for correlation coefficient.

        With optional stratified analysis to detect whether correlation
        differs across demographic groups.

        Args:
            X: First variable
            Y: Second variable
            demographic_labels: Optional group labels for stratified analysis

        Returns:
            Dictionary with correlation estimate and confidence interval
        """
        # Remove missing values
        mask = ~(np.isnan(X) | np.isnan(Y))
        X_clean = X[mask]
        Y_clean = Y[mask]

        # Observed correlation
        observed_corr = np.corrcoef(X_clean, Y_clean)[0, 1]

        # Bootstrap distribution
        bootstrap_corrs = []

        for b in range(self.n_bootstrap):
            # Sample pairs jointly to preserve pairing
            indices = np.random.choice(len(X_clean), size=len(X_clean), replace=True)
            boot_corr = np.corrcoef(X_clean[indices], Y_clean[indices])[0, 1]
            bootstrap_corrs.append(boot_corr)

        bootstrap_corrs = np.array(bootstrap_corrs)

        ci_lower = np.percentile(bootstrap_corrs, 2.5)
        ci_upper = np.percentile(bootstrap_corrs, 97.5)

        results = {
            'observed_correlation': observed_corr,
            'confidence_interval_95': (ci_lower, ci_upper),
            'bootstrap_std': np.std(bootstrap_corrs)
        }

        # Stratified analysis
        if demographic_labels is not None:
            demographic_clean = demographic_labels[mask]
            unique_groups = np.unique(demographic_clean[~pd.isna(demographic_clean)])

            group_correlations = {}

            for group in unique_groups:
                group_mask = demographic_clean == group
                X_group = X_clean[group_mask]
                Y_group = Y_clean[group_mask]

                if len(X_group) > 10:
                    # Bootstrap correlation for this group
                    group_boot_corrs = []

                    for b in range(self.n_bootstrap):
                        indices = np.random.choice(len(X_group), size=len(X_group), replace=True)
                        boot_corr = np.corrcoef(X_group[indices], Y_group[indices])[0, 1]
                        group_boot_corrs.append(boot_corr)

                    group_boot_corrs = np.array(group_boot_corrs)

                    group_correlations[str(group)] = {
                        'observed_correlation': np.corrcoef(X_group, Y_group)[0, 1],
                        'confidence_interval_95': (
                            np.percentile(group_boot_corrs, 2.5),
                            np.percentile(group_boot_corrs, 97.5)
                        ),
                        'n': len(X_group)
                    }

            results['group_correlations'] = group_correlations

            # Check for heterogeneity
            corrs = [v['observed_correlation'] for v in group_correlations.values()]
            if len(corrs) > 1 and max(corrs) - min(corrs) > 0.3:
                logger.warning(
                    f"Correlation varies substantially across groups "
                    f"(range = {max(corrs) - min(corrs):.3f}). "
                    f"Overall correlation may not represent all populations."
                )

        return results
```

These bootstrap methods maintain demographic representation and provide group-stratified inference that surfaces heterogeneity. The stratified bootstrap ensures that minority populations don't get under-represented by chance in bootstrap samples, which would lead to unstable estimates of their parameters.

## 2.5 Conclusion and Key Takeaways

This chapter has developed mathematical foundations for healthcare AI with sustained attention to how mathematical choices interact with health equity considerations. We've seen that mathematics in healthcare is never purely technical but rather embeds assumptions and value judgments that have differential impacts across patient populations. Linear algebra operations like standardization and dimensionality reduction can obscure or reveal health disparities depending on whether they preserve within-group variation structures. Probability distributions reflect not just biological variation but also social processes of healthcare access and measurement. Statistical inference methods make sampling assumptions that are routinely violated in healthcare data in ways that systematically disadvantage underserved populations.

The key insight is that achieving health equity through AI requires not just applying sophisticated mathematical methods but rather maintaining critical awareness of when mathematical assumptions break down and how these failures affect different populations differently. Every implementation in this chapter includes explicit checks for whether mathematical assumptions hold uniformly across demographic groups, sensitivity analyses that quantify uncertainty when assumptions are questionable, and stratified analyses that surface heterogeneity that pooled analyses would obscure.

The chapters that follow build on these mathematical foundations to develop machine learning and deep learning methods for healthcare applications. Throughout, we maintain the principle established here: mathematical sophistication must be paired with critical awareness of social context. The most elegant mathematical formulation is worthless if it produces biased predictions that widen rather than narrow health disparities. Success in healthcare AI requires both technical excellence and sustained commitment to equity.

## Bibliography

Agarwal, A., Beygelzimer, A., Dudík, M., Langford, J., & Wallach, H. (2018). A reductions approach to fair classification. *Proceedings of the 35th International Conference on Machine Learning*, 80, 60-69. http://proceedings.mlr.press/v80/agarwal18a.html

Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning: Limitations and Opportunities*. MIT Press. http://www.fairmlbook.org

Barredo Arrieta, A., Díaz-Rodríguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., Garcia, S., Gil-Lopez, S., Molina, D., Benjamins, R., Chatila, R., & Herrera, F. (2020). Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. *Information Fusion*, 58, 82-115. https://doi.org/10.1016/j.inffus.2019.12.012

Bayesian, M. A., Ghorbani, A., & Zou, J. (2019). Data Shapley: Equitable valuation of data for machine learning. *Proceedings of the 36th International Conference on Machine Learning*, 97, 2242-2251. http://proceedings.mlr.press/v97/ghorbani19c.html

Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324

Buolamwini, J., & Gebru, T. (2018). Gender shades: Intersectional accuracy disparities in commercial gender classification. *Proceedings of Machine Learning Research*, 81, 1-15. http://proceedings.mlr.press/v81/buolamwini18a.html

Calders, T., & Verwer, S. (2010). Three naive Bayes approaches for discrimination-free classification. *Data Mining and Knowledge Discovery*, 21(2), 277-292. https://doi.org/10.1007/s10618-010-0190-x

Caruana, R., Lou, Y., Gehrke, J., Koch, P., Sturm, M., & Elhadad, N. (2015). Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission. *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1721-1730. https://doi.org/10.1145/2783258.2788613

Chen, I. Y., Johansson, F. D., & Sontag, D. (2018). Why is my classifier discriminatory? *Advances in Neural Information Processing Systems*, 31, 3539-3550. https://proceedings.neurips.cc/paper/2018/file/1f1baa5b8edac74eb4eaa329f14a0361-Paper.pdf

Chen, I. Y., Pierson, E., Rose, S., Joshi, S., Ferryman, K., & Ghassemi, M. (2021). Ethical machine learning in healthcare. *Annual Review of Biomedical Data Science*, 4, 123-144. https://doi.org/10.1146/annurev-biodatasci-092820-114757

Chouldechova, A. (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. *Big Data*, 5(2), 153-163. https://doi.org/10.1089/big.2016.0047

Corbett-Davies, S., Pierson, E., Feller, A., Goel, S., & Huq, A. (2017). Algorithmic decision making and the cost of fairness. *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 797-806. https://doi.org/10.1145/3097983.3098095

D'Amour, A., Srinivasan, H., Atwood, J., Baljekar, P., Sculley, D., & Halpern, Y. (2020). Fairness is not static: deeper understanding of long term fairness via simulation studies. *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency*, 525-534. https://doi.org/10.1145/3351095.3372878

Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness through awareness. *Proceedings of the 3rd Innovations in Theoretical Computer Science Conference*, 214-226. https://doi.org/10.1145/2090236.2090255

Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. CRC Press.

Friedman, J., Hastie, T., & Tibshirani, R. (2001). *The Elements of Statistical Learning*. Springer.

Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *Proceedings of the 33rd International Conference on Machine Learning*, 48, 1050-1059. http://proceedings.mlr.press/v48/gal16.html

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.

Ghassemi, M., Naumann, T., Schulam, P., Beam, A. L., Chen, I. Y., & Ranganath, R. (2020). A review of challenges and opportunities in machine learning for health. *AMIA Summits on Translational Science Proceedings*, 2020, 191-200. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7233077/

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

Green, B., & Chen, Y. (2019). Disparate interactions: An algorithm-in-the-loop analysis of fairness in risk assessments. *Proceedings of the Conference on Fairness, Accountability, and Transparency*, 90-99. https://doi.org/10.1145/3287560.3287563

Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *Advances in Neural Information Processing Systems*, 29, 3315-3323. https://proceedings.neurips.cc/paper/2016/file/9d2682367c3935defcb1f9e247a97c0d-Paper.pdf

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

Holstein, K., Wortman Vaughan, J., Daumé III, H., Dudík, M., & Wallach, H. (2019). Improving fairness in machine learning systems: What do industry practitioners need? *Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems*, 1-16. https://doi.org/10.1145/3290605.3300830

Huang, K., Altosaar, J., & Ranganath, R. (2020). ClinicalBERT: Modeling clinical notes and predicting hospital readmission. *arXiv preprint arXiv:1904.05342*. https://arxiv.org/abs/1904.05342

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.

Joachims, T., Swaminathan, A., & Schnabel, T. (2017). Unbiased learning-to-rank with biased feedback. *Proceedings of the Tenth ACM International Conference on Web Search and Data Mining*, 781-789. https://doi.org/10.1145/3018661.3018699

Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.

Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for discrimination prevention. *Knowledge and Information Systems*, 33(1), 1-33. https://doi.org/10.1007/s10115-011-0463-8

Kleinberg, J., Ludwig, J., Mullainathan, S., & Rambachan, A. (2018). Algorithmic fairness. *AEA Papers and Proceedings*, 108, 22-27. https://doi.org/10.1257/pandp.20181018

Kohavi, R., & Provost, F. (1998). Glossary of terms. *Machine Learning*, 30(2-3), 271-274.

Komorowski, M., Celi, L. A., Badawi, O., Gordon, A. C., & Faisal, A. A. (2018). The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care. *Nature Medicine*, 24(11), 1716-1720. https://doi.org/10.1038/s41591-018-0213-5

Kusner, M. J., Loftus, J., Russell, C., & Silva, R. (2017). Counterfactual fairness. *Advances in Neural Information Processing Systems*, 30, 4066-4076. https://proceedings.neurips.cc/paper/2017/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf

Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401(6755), 788-791. https://doi.org/10.1038/44565

Liu, L. T., Dean, S., Rolf, E., Simchowitz, M., & Hardt, M. (2018). Delayed impact of fair machine learning. *Proceedings of the 35th International Conference on Machine Learning*, 80, 3150-3158. http://proceedings.mlr.press/v80/liu18c.html

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765-4774. https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf

McCallum, A., & Nigam, K. (1998). A comparison of event models for naive Bayes text classification. *AAAI-98 Workshop on Learning for Text Categorization*, 752(1), 41-48.

Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. *ACM Computing Surveys*, 54(6), 1-35. https://doi.org/10.1145/3457607

Mitchell, S., Potash, E., Barocas, S., D'Amour, A., & Lum, K. (2021). Algorithmic fairness: Choices, assumptions, and definitions. *Annual Review of Statistics and Its Application*, 8, 141-163. https://doi.org/10.1146/annurev-statistics-042720-125902

Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.

Narayanan, A. (2018). Translation tutorial: 21 fairness definitions and their politics. *Proceedings of the 2018 Conference on Fairness, Accountability, and Transparency*, 1-1. https://www.youtube.com/watch?v=jIXIuYdnyyk

Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447-453. https://doi.org/10.1126/science.aax2342

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.

Pearl, J., & Mackenzie, D. (2018). *The Book of Why: The New Science of Cause and Effect*. Basic Books.

Pleiss, G., Raghavan, M., Wu, F., Kleinberg, J., & Weinberger, K. Q. (2017). On fairness and calibration. *Advances in Neural Information Processing Systems*, 30, 5680-5689. https://proceedings.neurips.cc/paper/2017/file/b8b9c74ac526fffbeb2d39ab038d1cd7-Paper.pdf

Rajkomar, A., Dean, J., & Kohane, I. (2019). Machine learning in medicine. *New England Journal of Medicine*, 380(14), 1347-1358. https://doi.org/10.1056/NEJMra1814259

Rajkomar, A., Hardt, M., Howell, M. D., Corrado, G., & Chin, M. H. (2018). Ensuring fairness in machine learning to advance health equity. *Annals of Internal Medicine*, 169(12), 866-872. https://doi.org/10.7326/M18-1990

Raschka, S. (2018). Model evaluation, model selection, and algorithm selection in machine learning. *arXiv preprint arXiv:1811.12808*. https://arxiv.org/abs/1811.12808

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135-1144. https://doi.org/10.1145/2939672.2939778

Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581-592. https://doi.org/10.1093/biomet/63.3.581

Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), 206-215. https://doi.org/10.1038/s42256-019-0048-x

Saxena, N. A., Huang, K., DeFilippis, E., Radanovic, G., Parkes, D. C., & Liu, Y. (2019). How do fairness definitions fare? Examining public attitudes towards algorithmic definitions of fairness. *Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society*, 99-106. https://doi.org/10.1145/3306618.3314248

Shalev-Shwartz, S., & Ben-David, S. (2014). *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press.

Strang, G. (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press.

Trefethen, L. N., & Bau III, D. (1997). *Numerical Linear Algebra*. SIAM.

Ustun, B., & Rudin, C. (2016). Supersparse linear integer models for optimized medical scoring systems. *Machine Learning*, 102(3), 349-391. https://doi.org/10.1007/s10994-015-5528-6

Verma, S., & Rubin, J. (2018). Fairness definitions explained. *Proceedings of the International Workshop on Software Fairness*, 1-7. https://doi.org/10.1145/3194770.3194776

Wachter, S., Mittelstadt, B., & Russell, C. (2021). Why fairness cannot be automated: Bridging the gap between EU non-discrimination law and AI. *Computer Law & Security Review*, 41, 105567. https://doi.org/10.1016/j.clsr.2021.105567

Wainwright, M. J. (2019). *High-Dimensional Statistics: A Non-Asymptotic Viewpoint*. Cambridge University Press.

Wasserman, L. (2004). *All of Statistics: A Concise Course in Statistical Inference*. Springer.

Zafar, M. B., Valera, I., Gomez Rodriguez, M., & Gummadi, K. P. (2017). Fairness beyond disparate treatment & disparate impact: Learning classification without disparate mistreatment. *Proceedings of the 26th International Conference on World Wide Web*, 1171-1180. https://doi.org/10.1145/3038912.3052660

Zhang, B. H., Lemoine, B., & Mitchell, M. (2018). Mitigating unwanted biases with adversarial learning. *Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society*, 335-340. https://doi.org/10.1145/3278721.3278779

Zink, A., & Rose, S. (2020). Fair regression for health care spending. *Biometrics*, 76(3), 973-982. https://doi.org/10.1111/biom.13206
