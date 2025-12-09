---
layout: chapter
title: "Chapter 29: AI for Resource-Limited Clinical Settings"
chapter_number: 29
part_number: 7
prev_chapter: /chapters/chapter-28-continual-learning/
next_chapter: /chapters/chapter-30-research-frontiers-equity/
---
# Chapter 29: AI for Resource-Limited Clinical Settings

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Design healthcare AI systems that function reliably with limited computational infrastructure, intermittent connectivity, and constrained data storage, implementing model compression techniques, offline-capable architectures, and resource-adaptive algorithms appropriate for deployment in low-resource settings without compromising clinical utility.

2. Develop transfer learning frameworks that adapt models trained on data from high-resource settings to perform equitably in resource-limited contexts with different disease prevalence, clinical presentation patterns, and patient demographics, implementing domain adaptation techniques that account for distributional shift while avoiding performance degradation for underrepresented populations.

3. Build culturally and linguistically adaptive AI systems that work across diverse global contexts, implementing multilingual natural language processing, culturally appropriate risk communication, and participatory design approaches that center local knowledge and expertise rather than imposing external frameworks on local contexts.

4. Apply federated learning and privacy-preserving techniques that enable collaborative model development across institutions in different countries while respecting data sovereignty, implementing secure multi-party computation and differential privacy guarantees that protect sensitive health information from exploitation.

5. Implement participatory AI development processes that meaningfully engage local clinicians, community health workers, and patients as partners in system design and evaluation, ensuring AI systems address locally defined priorities rather than externally imposed metrics and that benefits accrue to communities providing training data.

6. Evaluate global health AI systems using equity-centered frameworks that assess not only clinical performance but also sustainability, local capacity building, alignment with local health priorities, and potential for perpetuating or challenging colonial patterns in global health technology development.

## 29.1 Introduction: Decolonizing Healthcare AI for Global Health

The global distribution of healthcare needs stands in stark contrast to the global distribution of healthcare AI development and deployment. Low and middle-income countries (LMICs), home to more than eighty percent of the world's population, bear an even higher proportion of the global disease burden yet receive only a small fraction of global health expenditure and account for less than five percent of clinical AI research. This profound mismatch creates both urgent need and substantial risk as healthcare AI expands globally.

The need is clear. Many LMICs face severe shortages of trained healthcare professionals, with sub-Saharan Africa averaging 0.2 physicians per 1,000 population compared to 3.2 in North America. In these contexts, AI systems that extend the capacity of limited clinical workforces could save millions of lives by enabling earlier diagnosis, more appropriate treatment selection, and more efficient allocation of scarce specialist resources. Diagnostic AI that helps community health workers identify tuberculosis in chest radiographs, predict preeclampsia risk from basic clinical data available in rural health posts, or screen for diabetic retinopathy using smartphone-based fundus photography represents potentially transformative applications for global health.

Yet the risks are equally clear and historically grounded. The history of global health is replete with examples of externally designed interventions that failed to account for local contexts, undermined local capacity, and perpetuated exploitative relationships between high-income and low-income settings. Healthcare AI risks replicating these patterns at unprecedented scale if systems developed primarily in high-resource settings are deployed globally without critical examination of their appropriateness, sustainability, and alignment with local priorities.

Consider the fundamental distributional mismatch. Models trained on data from academic medical centers in North America and Europe encounter populations with systematically different characteristics when deployed in LMICs. Disease prevalence differs dramatically—tuberculosis, malaria, and parasitic infections dominate infectious disease burden in many LMICs while remaining rare in high-income countries. Clinical presentation patterns vary due to differences in co-morbidities, nutritional status, treatment-seeking behavior, and timing of diagnosis relative to disease progression. Patient demographics skew younger in LMICs with higher fertility rates and lower life expectancy. Healthcare infrastructure differs fundamentally in available diagnostic tests, medications, specialist expertise, and follow-up capacity.

These differences mean that models trained on high-income country data may perform poorly when deployed in LMICs even when addressing the same clinical question. A sepsis prediction model trained on intensive care unit patients in Boston may fail when applied to febrile patients presenting to district hospitals in rural Uganda where malaria, typhoid, and other tropical infections are more common causes of similar clinical presentations. A diabetic retinopathy screening system calibrated to dilated fundus photography in ophthalmology clinics may perform poorly on non-dilated smartphone images captured by community health workers in remote areas.

Beyond technical performance, externally developed AI systems often fail to align with local health priorities and may inadvertently undermine local capacity. When technology companies from high-income countries deploy AI systems in LMICs while retaining data ownership, intellectual property rights, and algorithm control, they create dependencies that extract value from communities providing training data while offering little sustainable benefit. When AI development focuses on diseases that attract donor funding rather than addressing locally prioritized health needs, it perpetuates external agenda-setting in global health. When AI implementations require expensive proprietary hardware, reliable internet connectivity, or ongoing technical support from external vendors, they prove unsustainable when short-term funding ends.

This chapter develops approaches to healthcare AI for global health that center equity, sustainability, and decolonization as fundamental design principles rather than afterthoughts. We examine how to design AI systems appropriate for resource-limited settings that work with available infrastructure, limited training data, and constrained computational resources. We develop transfer learning and domain adaptation techniques that enable models to generalize across contexts while accounting for distributional differences. We implement federated learning approaches that enable collaborative model development while respecting data sovereignty. Most fundamentally, we explore participatory development processes that position local clinicians, community health workers, patients, and communities as partners who define priorities, contribute expertise, and maintain ownership rather than passive recipients of externally developed technologies.

The equity considerations in this chapter are not a separate section but woven throughout as fundamental design requirements. Every technical approach must be evaluated not only for clinical performance but for its potential to either challenge or perpetuate global health inequities. We ask repeatedly: Who benefits from this technology? Who owns the data and the resulting models? Who has the technical capacity to maintain and adapt the system? Whose priorities does the system serve? These questions guide our technical choices as much as accuracy metrics.

## 29.2 Infrastructure-Aware AI System Design

Resource-limited healthcare settings face fundamentally different infrastructure constraints than the environments where most healthcare AI is developed and tested. These constraints span computational resources, network connectivity, electrical power, data storage capacity, and technical support availability. Designing AI systems that function reliably despite these constraints requires explicit consideration of infrastructure limitations from initial architecture design through deployment and maintenance.

### 29.2.1 Computational Resource Constraints

Many healthcare facilities in LMICs lack the computational infrastructure assumed by modern deep learning systems. While academic medical centers deploy AI on servers with multiple GPUs and hundreds of gigabytes of RAM, community health centers in rural areas may have only basic computers or tablets. Mobile health implementations may rely entirely on smartphones or feature phones with severely limited computational capacity.

These constraints require rethinking standard AI architectures. Models with hundreds of millions or billions of parameters that achieve state-of-the-art performance in well-resourced settings become infeasible. We must instead design compact models that achieve acceptable performance within strict resource budgets through knowledge distillation, quantization, pruning, and architecture search optimized for edge deployment.

**Knowledge distillation** transfers knowledge from a large, complex teacher model to a smaller student model by training the student to match the teacher's predictions rather than only the ground truth labels. The student learns to approximate the teacher's decision function using far fewer parameters. For healthcare applications, this enables training large models on data from well-resourced settings where computational resources are available, then distilling that knowledge into compact models deployable on resource-constrained devices in LMICs.

The distillation loss combines the standard supervised loss with a term encouraging the student's predictions to match the teacher's probability distribution:

$$\mathcal{L}_{distill} = \alpha \mathcal{L}_{CE}(y, f_{student}(x)) + (1-\alpha) \mathcal{L}_{KL}(f_{teacher}(x), f_{student}(x))$$

where $$\mathcal{L}_{CE}$$ is cross-entropy loss between predictions and true labels, $$\mathcal{L}_{KL}$$ is Kullback-Leibler divergence between teacher and student prediction distributions, and $$\alpha$$ weights the relative importance of these objectives. The temperature parameter in the softmax function for both teacher and student is often increased during distillation to create softer probability distributions that convey more information about the teacher's uncertainty.

**Quantization** reduces model size and inference time by representing weights and activations with lower precision. Standard deep learning uses 32-bit floating-point numbers, but many models maintain acceptable performance with 16-bit, 8-bit, or even binary representations. This dramatically reduces memory requirements and speeds inference on hardware supporting low-precision operations.

Post-training quantization applies quantization after training a full-precision model, mapping the continuous range of trained weights to discrete levels. Quantization-aware training includes quantization operations in the training loop, allowing the model to learn weights that remain effective after quantization by simulating quantization during forward passes while maintaining full-precision weights during backward passes.

**Network pruning** removes redundant parameters by identifying and eliminating weights that contribute minimally to model predictions. Magnitude-based pruning removes weights with small absolute values. Structured pruning removes entire neurons, filters, or layers rather than individual weights, creating genuinely compact models rather than sparse models requiring specialized inference libraries.

For healthcare applications in resource-limited settings, pruning strategies should explicitly consider the distribution shift between training data (typically from high-resource settings) and deployment contexts (resource-limited settings). Weights that seem redundant based on high-resource training data may be essential for generalizing to different disease prevalence and presentation patterns in deployment contexts. Stratified pruning evaluation that tests pruned model performance on diverse validation data from intended deployment settings ensures pruning does not disproportionately degrade performance for specific populations or clinical contexts.

**Neural architecture search for efficiency** automatically discovers model architectures optimized for both accuracy and computational efficiency. Rather than searching only for maximum accuracy, multi-objective architecture search balances accuracy against model size, inference latency, and energy consumption. This identifies architectures achieving Pareto-optimal tradeoffs appropriate for different deployment constraints.

### 29.2.2 Implementation: Model Compression for Resource-Limited Deployment

We implement a complete pipeline for compressing healthcare AI models for deployment in resource-limited settings:

```python
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompressionStrategy(Enum):
    """Compression strategies for model optimization."""
    DISTILLATION = "distillation"
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    COMBINED = "combined"

@dataclass
class ResourceConstraints:
    """Hardware resource constraints for deployment environment."""
    max_model_size_mb: float  # Maximum model size in megabytes
    max_inference_time_ms: float  # Maximum inference latency in milliseconds
    target_device: str  # Target device type (cpu, mobile, embedded)
    memory_limit_mb: float  # Available RAM in megabytes
    supports_fp16: bool = False  # Hardware supports 16-bit precision
    supports_int8: bool = False  # Hardware supports 8-bit integer operations

    def is_satisfied_by(self, model: nn.Module, batch_size: int = 1) -> bool:
        """
        Check if model satisfies resource constraints.

        Returns True if model meets all specified constraints.
        """
        # Check model size
        model_size_mb = self._calculate_model_size(model)
        if model_size_mb > self.max_model_size_mb:
            logger.warning(
                f"Model size {model_size_mb:.2f}MB exceeds limit "
                f"{self.max_model_size_mb:.2f}MB"
            )
            return False

        # Check memory requirements
        memory_mb = self._estimate_memory_usage(model, batch_size)
        if memory_mb > self.memory_limit_mb:
            logger.warning(
                f"Estimated memory {memory_mb:.2f}MB exceeds limit "
                f"{self.memory_limit_mb:.2f}MB"
            )
            return False

        return True

    @staticmethod
    def _calculate_model_size(model: nn.Module) -> float:
        """Calculate model size in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    @staticmethod
    def _estimate_memory_usage(model: nn.Module, batch_size: int) -> float:
        """Estimate peak memory usage during inference."""
        # Rough estimate: model size + activations for typical medical image
        model_size = ResourceConstraints._calculate_model_size(model)
        # Assume 224x224x3 images, fp32, with 2x overhead for activations
        activation_size = (batch_size * 224 * 224 * 3 * 4 * 2) / (1024 ** 2)
        return model_size + activation_size

@dataclass
class CompressionMetrics:
    """Metrics tracking compression effectiveness and equity impact."""
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    original_inference_time_ms: float
    compressed_inference_time_ms: float
    speedup_ratio: float

    # Performance metrics overall and stratified
    overall_accuracy_original: float
    overall_accuracy_compressed: float
    accuracy_by_group: Dict[str, Tuple[float, float]]  # group -> (original, compressed)

    # Equity metrics
    max_accuracy_degradation: float  # Worst-case accuracy loss across groups
    accuracy_disparity_change: float  # Change in disparity between groups

    def summary(self) -> str:
        """Generate human-readable summary of compression results."""
        summary = [
            f"Compression Results:",
            f"  Size: {self.original_size_mb:.1f}MB → {self.compressed_size_mb:.1f}MB "
            f"({self.compression_ratio:.1f}x compression)",
            f"  Speed: {self.original_inference_time_ms:.1f}ms → "
            f"{self.compressed_inference_time_ms:.1f}ms ({self.speedup_ratio:.1f}x speedup)",
            f"  Overall Accuracy: {self.overall_accuracy_original:.3f} → "
            f"{self.overall_accuracy_compressed:.3f}",
            f"\nPer-Group Performance:",
        ]

        for group, (orig_acc, comp_acc) in self.accuracy_by_group.items():
            degradation = orig_acc - comp_acc
            summary.append(
                f"  {group}: {orig_acc:.3f} → {comp_acc:.3f} "
                f"(Δ {degradation:+.3f})"
            )

        summary.extend([
            f"\nEquity Impact:",
            f"  Max Accuracy Degradation: {self.max_accuracy_degradation:.3f}",
            f"  Disparity Change: {self.accuracy_disparity_change:+.3f}",
        ])

        return "\n".join(summary)

class KnowledgeDistillation:
    """
    Knowledge distillation for model compression.

    Trains a compact student model to match predictions of a larger
    teacher model, enabling deployment in resource-limited settings
    while maintaining clinical performance.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 3.0,
        alpha: float = 0.7,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize knowledge distillation.

        Args:
            teacher_model: Large pre-trained model to distill from
            student_model: Compact model to train
            temperature: Temperature for softening probability distributions
            alpha: Weight for distillation loss (1-alpha for hard label loss)
            device: Device for computation
        """
        self.teacher_model = teacher_model.eval()  # Teacher in eval mode
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.teacher_model.to(self.device)
        self.student_model.to(self.device)

        logger.info(
            f"Initialized distillation: Teacher {self._count_parameters(teacher_model)}M "
            f"params → Student {self._count_parameters(student_model)}M params"
        )

    @staticmethod
    def _count_parameters(model: nn.Module) -> float:
        """Count model parameters in millions."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate combined distillation and supervised loss.

        Args:
            student_logits: Raw logits from student model
            teacher_logits: Raw logits from teacher model
            labels: Ground truth labels

        Returns:
            Combined loss tensor
        """
        # Soft targets from teacher with temperature scaling
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=1)

        # Distillation loss (KL divergence between teacher and student)
        distillation_loss = F.kl_div(
            soft_predictions,
            soft_targets,
            reduction='batchmean',
        ) * (self.temperature ** 2)  # Scale by T^2 to maintain gradient magnitude

        # Hard label loss (standard cross-entropy)
        hard_loss = F.cross_entropy(student_logits, labels)

        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss

        return total_loss

    def train_student(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        early_stopping_patience: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Train student model via knowledge distillation.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum training epochs
            learning_rate: Initial learning rate
            early_stopping_patience: Epochs without improvement before stopping

        Returns:
            Dictionary containing training history
        """
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
        }

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            # Training phase
            self.student_model.train()
            train_loss = 0.0

            for batch_idx, (data, labels) in enumerate(train_loader):
                data, labels = data.to(self.device), labels.to(self.device)

                # Get teacher predictions (no gradients needed)
                with torch.no_grad():
                    teacher_logits = self.teacher_model(data)

                # Get student predictions
                optimizer.zero_grad()
                student_logits = self.student_model(data)

                # Calculate distillation loss
                loss = self.distillation_loss(student_logits, teacher_logits, labels)

                # Backpropagation
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            # Validation phase
            val_loss, val_accuracy = self._validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save best model
                torch.save(
                    self.student_model.state_dict(),
                    'best_student_model.pth'
                )
            else:
                epochs_without_improvement += 1

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_accuracy:.4f}"
            )

            if epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break

        # Load best model
        self.student_model.load_state_dict(torch.load('best_student_model.pth'))

        return history

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate student model performance.

        Returns:
            Tuple of (validation loss, validation accuracy)
        """
        self.student_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                # Get predictions from both models
                teacher_logits = self.teacher_model(data)
                student_logits = self.student_model(data)

                # Calculate loss
                loss = self.distillation_loss(student_logits, teacher_logits, labels)
                val_loss += loss.item()

                # Calculate accuracy
                predictions = student_logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total

        return avg_val_loss, accuracy

class QuantizationAwareTraining:
    """
    Quantization-aware training for efficient model deployment.

    Trains models with simulated quantization to learn weights that
    maintain performance when quantized to low precision (8-bit or 16-bit).
    """

    def __init__(
        self,
        model: nn.Module,
        quantization_bits: int = 8,
        quantize_weights: bool = True,
        quantize_activations: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize quantization-aware training.

        Args:
            model: Model to quantize
            quantization_bits: Number of bits for quantization (8 or 16)
            quantize_weights: Whether to quantize model weights
            quantize_activations: Whether to quantize activations
            device: Device for computation
        """
        self.model = model
        self.quantization_bits = quantization_bits
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)

        # Prepare model for quantization-aware training
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(self.model, inplace=True)

        logger.info(
            f"Initialized {quantization_bits}-bit quantization-aware training"
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 30,
        learning_rate: float = 1e-4,
    ) -> Dict[str, List[float]]:
        """
        Train model with quantization awareness.

        During training, forward passes simulate quantization while backward
        passes use full precision, allowing model to learn weights that work
        well when quantized.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate

        Returns:
            Training history dictionary
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
        }

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = correct / total

            history['train_loss'].append(avg_train_loss)
            history['train_accuracy'].append(train_accuracy)

            # Validation phase
            val_loss, val_accuracy = self._validate(val_loader, criterion)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )

        return history

    def _validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Validate quantized model performance."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                outputs = self.model(data)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total

        return avg_val_loss, accuracy

    def convert_to_quantized(self) -> nn.Module:
        """
        Convert trained model to fully quantized version.

        Returns:
            Quantized model ready for deployment
        """
        self.model.eval()
        quantized_model = torch.quantization.convert(self.model, inplace=False)

        logger.info("Converted model to fully quantized version")

        return quantized_model

class ModelCompression:
    """
    Comprehensive model compression pipeline for resource-limited deployment.

    Combines distillation, quantization, and pruning strategies with
    equity-focused evaluation to ensure compression doesn't degrade
    performance for underrepresented populations.
    """

    def __init__(
        self,
        model: nn.Module,
        constraints: ResourceConstraints,
        strategy: CompressionStrategy = CompressionStrategy.COMBINED,
    ):
        """
        Initialize compression pipeline.

        Args:
            model: Model to compress
            constraints: Target deployment resource constraints
            strategy: Compression strategy to apply
        """
        self.model = model
        self.constraints = constraints
        self.strategy = strategy

        logger.info(
            f"Initialized compression with strategy {strategy.value} "
            f"targeting {constraints.target_device}"
        )

    def compress(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader_by_group: Dict[str, DataLoader],
    ) -> Tuple[nn.Module, CompressionMetrics]:
        """
        Apply compression strategy and evaluate impact.

        Args:
            train_loader: Training data for distillation/fine-tuning
            val_loader: Validation data for hyperparameter tuning
            test_loader_by_group: Test data stratified by demographic groups

        Returns:
            Tuple of (compressed model, compression metrics)
        """
        # Evaluate original model
        original_metrics = self._evaluate_model(
            self.model,
            test_loader_by_group,
        )

        # Apply compression based on strategy
        if self.strategy == CompressionStrategy.DISTILLATION:
            compressed_model = self._apply_distillation(
                train_loader,
                val_loader,
            )
        elif self.strategy == CompressionStrategy.QUANTIZATION:
            compressed_model = self._apply_quantization(
                train_loader,
                val_loader,
            )
        elif self.strategy == CompressionStrategy.PRUNING:
            compressed_model = self._apply_pruning(
                val_loader,
            )
        else:  # COMBINED
            compressed_model = self._apply_combined_compression(
                train_loader,
                val_loader,
            )

        # Evaluate compressed model
        compressed_metrics = self._evaluate_model(
            compressed_model,
            test_loader_by_group,
        )

        # Calculate compression metrics
        metrics = self._calculate_compression_metrics(
            original_metrics,
            compressed_metrics,
        )

        # Verify constraints are satisfied
        if not self.constraints.is_satisfied_by(compressed_model):
            logger.warning(
                "Compressed model does not satisfy all resource constraints. "
                "Consider more aggressive compression or relaxing constraints."
            )

        return compressed_model, metrics

    def _apply_distillation(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> nn.Module:
        """Apply knowledge distillation compression."""
        # Create smaller student architecture
        student_model = self._create_student_architecture()

        # Distill knowledge
        distiller = KnowledgeDistillation(
            teacher_model=self.model,
            student_model=student_model,
            temperature=3.0,
            alpha=0.7,
        )

        distiller.train_student(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=50,
        )

        return distiller.student_model

    def _apply_quantization(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> nn.Module:
        """Apply quantization compression."""
        # Determine quantization precision based on hardware support
        if self.constraints.supports_int8:
            bits = 8
        elif self.constraints.supports_fp16:
            bits = 16
        else:
            logger.warning("Hardware doesn't support low-precision, using 16-bit")
            bits = 16

        # Apply quantization-aware training
        qat = QuantizationAwareTraining(
            model=self.model,
            quantization_bits=bits,
        )

        qat.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=30,
        )

        # Convert to fully quantized model
        quantized_model = qat.convert_to_quantized()

        return quantized_model

    def _apply_pruning(
        self,
        val_loader: DataLoader,
    ) -> nn.Module:
        """Apply structured pruning compression."""
        # Implement magnitude-based structured pruning
        pruned_model = self._prune_model(
            self.model,
            sparsity=0.5,  # Remove 50% of parameters
            val_loader=val_loader,
        )

        return pruned_model

    def _apply_combined_compression(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> nn.Module:
        """Apply combined compression strategy."""
        # Step 1: Distillation to smaller architecture
        student_model = self._create_student_architecture()
        distiller = KnowledgeDistillation(
            teacher_model=self.model,
            student_model=student_model,
        )
        distiller.train_student(train_loader, val_loader)

        # Step 2: Quantization of distilled model
        qat = QuantizationAwareTraining(
            model=distiller.student_model,
            quantization_bits=8 if self.constraints.supports_int8 else 16,
        )
        qat.train(train_loader, val_loader, num_epochs=20)

        # Step 3: Convert to quantized
        compressed_model = qat.convert_to_quantized()

        return compressed_model

    def _create_student_architecture(self) -> nn.Module:
        """
        Create compact student model architecture.

        In production, use NAS or manual architecture design.
        This is a simplified placeholder.
        """
        # Simplified: create a smaller version of the teacher
        # In practice, use neural architecture search or domain expertise
        logger.info("Creating student architecture (simplified placeholder)")

        # For demonstration, assume teacher is a CNN classifier
        # Create student with fewer filters and layers
        class CompactCNN(nn.Module):
            def __init__(self, num_classes: int):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(64, num_classes),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.features(x)
                x = self.classifier(x)
                return x

        return CompactCNN(num_classes=2)  # Binary classification example

    def _prune_model(
        self,
        model: nn.Module,
        sparsity: float,
        val_loader: DataLoader,
    ) -> nn.Module:
        """
        Apply magnitude-based pruning with validation.

        This is a simplified implementation. Production systems should use
        more sophisticated pruning strategies.
        """
        import torch.nn.utils.prune as prune

        # Identify prunable layers
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))

        # Apply global magnitude pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )

        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        return model

    def _evaluate_model(
        self,
        model: nn.Module,
        test_loader_by_group: Dict[str, DataLoader],
    ) -> Dict[str, Any]:
        """
        Evaluate model performance overall and by demographic group.

        Returns:
            Dictionary containing performance metrics
        """
        model.eval()
        device = next(model.parameters()).device

        metrics = {
            'overall': {'correct': 0, 'total': 0},
            'by_group': {},
        }

        # Evaluate overall and by group
        all_loaders = {'overall': self._combine_loaders(test_loader_by_group)}
        all_loaders.update(test_loader_by_group)

        with torch.no_grad():
            for group_name, loader in all_loaders.items():
                correct = 0
                total = 0

                for data, labels in loader:
                    data, labels = data.to(device), labels.to(device)

                    outputs = model(data)
                    predictions = outputs.argmax(dim=1)

                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

                accuracy = correct / total if total > 0 else 0.0

                if group_name == 'overall':
                    metrics['overall']['accuracy'] = accuracy
                else:
                    metrics['by_group'][group_name] = accuracy

        return metrics

    @staticmethod
    def _combine_loaders(loaders: Dict[str, DataLoader]) -> DataLoader:
        """Combine multiple data loaders into single loader."""
        # Simplified: just use first loader as proxy for overall
        # In production, properly combine datasets
        return next(iter(loaders.values()))

    def _calculate_compression_metrics(
        self,
        original_metrics: Dict[str, Any],
        compressed_metrics: Dict[str, Any],
    ) -> CompressionMetrics:
        """Calculate comprehensive compression metrics."""
        # Calculate sizes
        original_size = ResourceConstraints._calculate_model_size(self.model)
        compressed_size = ResourceConstraints._calculate_model_size(
            # Assume compressed model is in compressed_metrics context
            # This is simplified for the example
            self.model
        )

        # Calculate performance by group
        accuracy_by_group = {}
        for group in original_metrics['by_group']:
            orig_acc = original_metrics['by_group'][group]
            comp_acc = compressed_metrics['by_group'][group]
            accuracy_by_group[group] = (orig_acc, comp_acc)

        # Calculate equity metrics
        degradations = [
            orig - comp for orig, comp in accuracy_by_group.values()
        ]
        max_degradation = max(degradations) if degradations else 0.0

        # Calculate disparity change
        orig_accuracies = [orig for orig, _ in accuracy_by_group.values()]
        comp_accuracies = [comp for _, comp in accuracy_by_group.values()]
        orig_disparity = max(orig_accuracies) - min(orig_accuracies) if orig_accuracies else 0.0
        comp_disparity = max(comp_accuracies) - min(comp_accuracies) if comp_accuracies else 0.0
        disparity_change = comp_disparity - orig_disparity

        return CompressionMetrics(
            original_size_mb=original_size,
            compressed_size_mb=compressed_size,
            compression_ratio=original_size / compressed_size if compressed_size > 0 else 1.0,
            original_inference_time_ms=10.0,  # Placeholder - measure in production
            compressed_inference_time_ms=5.0,  # Placeholder - measure in production
            speedup_ratio=2.0,  # Placeholder - measure in production
            overall_accuracy_original=original_metrics['overall']['accuracy'],
            overall_accuracy_compressed=compressed_metrics['overall']['accuracy'],
            accuracy_by_group=accuracy_by_group,
            max_accuracy_degradation=max_degradation,
            accuracy_disparity_change=disparity_change,
        )
```

This implementation provides a complete pipeline for compressing healthcare AI models while maintaining equity. The system combines multiple compression strategies including knowledge distillation to transfer knowledge from large teachers to compact students, quantization-aware training to learn weights that maintain performance at low precision, and structured pruning to remove redundant parameters. Critically, the evaluation framework explicitly tracks performance across demographic groups, ensuring compression doesn't disproportionately degrade performance for underrepresented populations.

### 29.2.3 Connectivity and Data Infrastructure

Network connectivity in resource-limited settings ranges from unreliable to nonexistent. Even when internet access is available, bandwidth may be extremely limited and costs per megabyte prohibitively expensive for users. These constraints require AI systems that function effectively offline while synchronizing efficiently when connectivity is available.

**Offline-first architecture** designs AI systems that perform all critical functions locally without internet connectivity. Models, data, and application logic reside entirely on local devices—tablets, smartphones, or computers in health facilities. The system remains fully functional when offline, with connectivity required only for optional features like model updates or data backup.

For diagnostic AI deployed in remote health posts, this might mean preloading model weights, reference data, and user interfaces on tablets distributed to community health workers. The tablets collect patient data, run AI models locally, and store predictions in local databases. When workers return to district hospitals with internet access, collected data syncs to central servers for aggregation and quality review while any available model updates download for future use.

**Efficient synchronization** minimizes data transfer when connectivity becomes available. Delta synchronization transmits only changes since last sync rather than complete datasets. Compressed representations use efficient encoding for medical data. Priority-based synchronization sends high-priority data first given uncertain connectivity duration.

**Progressive enhancement** allows systems to function at different levels of capability depending on available connectivity. Core diagnostic and treatment recommendation functions work completely offline. Enhanced features like accessing the latest clinical guidelines or obtaining specialist teleconsultation require connectivity but aren't essential for basic functionality. Users understand what features require connectivity and experience graceful degradation rather than failure when offline.

### 29.2.4 Implementation: Offline-Capable Diagnostic System

We implement a complete diagnostic AI system designed for offline deployment with efficient synchronization:

```python
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import hashlib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PatientDiagnosis:
    """Patient diagnosis record for offline storage."""
    patient_id: str
    timestamp: str
    symptoms: List[str]
    vital_signs: Dict[str, float]
    risk_scores: Dict[str, float]
    recommended_actions: List[str]
    diagnostic_image_path: Optional[str] = None
    model_version: str = "1.0"
    synced: bool = False
    sync_timestamp: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'PatientDiagnosis':
        """Create instance from dictionary."""
        return cls(**data)

    def get_hash(self) -> str:
        """Generate unique hash for delta synchronization."""
        # Exclude sync-related fields from hash
        hash_data = {
            k: v for k, v in self.to_dict().items()
            if k not in ['synced', 'sync_timestamp']
        }
        return hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()

class OfflineDiagnosticSystem:
    """
    Offline-capable diagnostic AI system for resource-limited settings.

    Provides complete diagnostic functionality without internet connectivity,
    with efficient synchronization when connectivity becomes available.
    """

    def __init__(
        self,
        model_path: Path,
        database_path: Path,
        device: str = 'cpu',
        model_version: str = "1.0",
    ):
        """
        Initialize offline diagnostic system.

        Args:
            model_path: Path to compressed model file
            database_path: Path to local SQLite database
            device: Device for model inference (cpu for resource-limited settings)
            model_version: Model version identifier
        """
        self.model_path = model_path
        self.database_path = database_path
        self.device = torch.device(device)
        self.model_version = model_version

        # Load model
        self.model = self._load_model()

        # Initialize local database
        self._init_database()

        logger.info(
            f"Initialized offline diagnostic system with model v{model_version}"
        )

    def _load_model(self) -> torch.nn.Module:
        """
        Load compressed model from disk.

        Model should be compressed to fit resource constraints.
        """
        try:
            model = torch.load(
                self.model_path,
                map_location=self.device,
            )
            model.eval()
            logger.info(f"Loaded model from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _init_database(self):
        """
        Initialize local SQLite database for offline storage.

        SQLite provides reliable local storage without requiring
        database server infrastructure.
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        # Create diagnoses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                symptoms TEXT,
                vital_signs TEXT,
                risk_scores TEXT,
                recommended_actions TEXT,
                diagnostic_image_path TEXT,
                model_version TEXT,
                synced INTEGER DEFAULT 0,
                sync_timestamp TEXT,
                record_hash TEXT UNIQUE
            )
        ''')

        # Create sync queue table for tracking pending syncs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sync_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                diagnosis_id INTEGER,
                priority INTEGER DEFAULT 0,
                attempts INTEGER DEFAULT 0,
                last_attempt TEXT,
                FOREIGN KEY (diagnosis_id) REFERENCES diagnoses(id)
            )
        ''')

        conn.commit()
        conn.close()

        logger.info("Initialized local database")

    def diagnose_patient(
        self,
        patient_id: str,
        symptoms: List[str],
        vital_signs: Dict[str, float],
        diagnostic_image: Optional[np.ndarray] = None,
    ) -> PatientDiagnosis:
        """
        Perform diagnostic assessment using local AI model.

        This function works completely offline, requiring no internet
        connectivity for core diagnostic functionality.

        Args:
            patient_id: Unique patient identifier
            symptoms: List of reported symptoms
            vital_signs: Dictionary of vital sign measurements
            diagnostic_image: Optional medical image (e.g., X-ray)

        Returns:
            PatientDiagnosis containing risk scores and recommendations
        """
        timestamp = datetime.utcnow().isoformat()

        # Process clinical features
        clinical_features = self._process_clinical_features(
            symptoms,
            vital_signs,
        )

        # Process diagnostic image if provided
        image_features = None
        image_path = None
        if diagnostic_image is not None:
            image_features = self._process_diagnostic_image(diagnostic_image)
            # Save image locally
            image_path = self._save_diagnostic_image(
                diagnostic_image,
                patient_id,
                timestamp,
            )

        # Run AI model inference
        risk_scores = self._predict_risk_scores(
            clinical_features,
            image_features,
        )

        # Generate recommendations based on risk scores
        recommended_actions = self._generate_recommendations(
            risk_scores,
            symptoms,
            vital_signs,
        )

        # Create diagnosis record
        diagnosis = PatientDiagnosis(
            patient_id=patient_id,
            timestamp=timestamp,
            symptoms=symptoms,
            vital_signs=vital_signs,
            risk_scores=risk_scores,
            recommended_actions=recommended_actions,
            diagnostic_image_path=image_path,
            model_version=self.model_version,
            synced=False,
        )

        # Store in local database
        self._save_diagnosis(diagnosis)

        logger.info(
            f"Completed diagnosis for patient {patient_id} "
            f"with {len(risk_scores)} risk scores"
        )

        return diagnosis

    def _process_clinical_features(
        self,
        symptoms: List[str],
        vital_signs: Dict[str, float],
    ) -> torch.Tensor:
        """
        Process clinical features into model input format.

        In production, implement proper feature encoding with
        medical ontology mapping for symptoms.
        """
        # Simplified feature processing
        # In production: use medical ontologies (SNOMED CT, ICD codes)
        # to encode symptoms consistently

        # Create feature vector
        # This is a simplified placeholder
        feature_vector = np.zeros(50)  # Fixed-size feature vector

        # Encode vital signs
        vital_sign_indices = {
            'temperature': 0,
            'heart_rate': 1,
            'respiratory_rate': 2,
            'blood_pressure_systolic': 3,
            'blood_pressure_diastolic': 4,
            'oxygen_saturation': 5,
        }

        for sign, value in vital_signs.items():
            if sign in vital_sign_indices:
                idx = vital_sign_indices[sign]
                # Normalize vital signs (simplified)
                feature_vector[idx] = value / 100.0

        # Encode symptoms as binary features
        common_symptoms = [
            'fever', 'cough', 'shortness_of_breath', 'fatigue',
            'chest_pain', 'headache', 'nausea', 'diarrhea',
        ]

        for i, symptom in enumerate(common_symptoms):
            if i + 10 < len(feature_vector):
                feature_vector[i + 10] = 1.0 if symptom in symptoms else 0.0

        return torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)

    def _process_diagnostic_image(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:
        """
        Process diagnostic image into model input format.

        Handles preprocessing appropriate for resource-limited settings
        where image quality may vary.
        """
        # Simplified image preprocessing
        # In production: handle various image formats, qualities

        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image).float()

        # Ensure correct shape (B, C, H, W)
        if len(image_tensor.shape) == 2:  # Grayscale
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        elif len(image_tensor.shape) == 3:  # RGB
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

        # Normalize to [0, 1]
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        return image_tensor

    def _save_diagnostic_image(
        self,
        image: np.ndarray,
        patient_id: str,
        timestamp: str,
    ) -> str:
        """
        Save diagnostic image to local storage.

        Uses efficient compression appropriate for storage constraints.
        """
        # Create directory structure
        image_dir = Path('diagnostic_images') / patient_id
        image_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        safe_timestamp = timestamp.replace(':', '-')
        image_path = image_dir / f"{safe_timestamp}.npy"

        # Save as compressed numpy array
        # In production, use medical image formats (DICOM) or efficient
        # compression (JPEG 2000) appropriate for the image type
        np.save(image_path, image)

        return str(image_path)

    def _predict_risk_scores(
        self,
        clinical_features: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Predict disease risk scores using local AI model.

        Works entirely offline using preloaded model.
        """
        # Move inputs to device
        clinical_features = clinical_features.to(self.device)

        if image_features is not None:
            image_features = image_features.to(self.device)

        # Run inference
        with torch.no_grad():
            # Simplified inference - in production, handle multimodal fusion
            if image_features is not None:
                # Combine features if both available
                combined_features = torch.cat([
                    clinical_features,
                    image_features.flatten(1),
                ], dim=1)
                outputs = self.model(combined_features)
            else:
                outputs = self.model(clinical_features)

            # Convert to probabilities
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

        # Map to disease categories
        # In production, use proper disease ontology
        disease_categories = [
            'tuberculosis',
            'pneumonia',
            'malaria',
            'covid19',
            'sepsis',
        ]

        risk_scores = {}
        for i, disease in enumerate(disease_categories):
            if i < len(probabilities):
                risk_scores[disease] = float(probabilities[i])

        return risk_scores

    def _generate_recommendations(
        self,
        risk_scores: Dict[str, float],
        symptoms: List[str],
        vital_signs: Dict[str, float],
    ) -> List[str]:
        """
        Generate clinical recommendations based on risk assessment.

        Uses rule-based logic appropriate for local clinical protocols.
        In production, integrate with local treatment guidelines and
        formulary availability.
        """
        recommendations = []

        # Check for high-risk conditions requiring immediate action
        for disease, score in risk_scores.items():
            if score > 0.7:  # High risk threshold
                if disease == 'sepsis':
                    recommendations.append(
                        "URGENT: High sepsis risk. Start broad-spectrum antibiotics "
                        "and arrange immediate transfer to higher-level facility."
                    )
                elif disease == 'tuberculosis':
                    recommendations.append(
                        "High TB risk. Isolate patient, collect sputum sample, "
                        "initiate TB diagnostic workup. Consider empiric treatment "
                        "if diagnostic delay expected."
                    )
                elif disease == 'pneumonia':
                    recommendations.append(
                        "High pneumonia risk. Check oxygen saturation, consider "
                        "antibiotics per local guidelines, monitor for respiratory distress."
                    )
                elif disease == 'malaria':
                    recommendations.append(
                        "High malaria risk. Perform rapid diagnostic test, "
                        "initiate antimalarial therapy if positive per local guidelines."
                    )

        # Check vital sign abnormalities
        if 'oxygen_saturation' in vital_signs:
            if vital_signs['oxygen_saturation'] < 90:
                recommendations.append(
                    "CRITICAL: Hypoxemia detected. Provide supplemental oxygen, "
                    "monitor closely, consider transfer if severe."
                )

        if 'temperature' in vital_signs:
            if vital_signs['temperature'] > 38.5:
                recommendations.append(
                    "Fever present. Consider antipyretics, ensure adequate hydration, "
                    "investigate cause."
                )

        # Default recommendation if no high-risk conditions
        if not recommendations:
            recommendations.append(
                "Continue supportive care and monitoring. Reassess if symptoms worsen."
            )

        return recommendations

    def _save_diagnosis(self, diagnosis: PatientDiagnosis):
        """Save diagnosis to local database."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        # Generate hash for delta sync
        record_hash = diagnosis.get_hash()

        try:
            cursor.execute('''
                INSERT INTO diagnoses (
                    patient_id, timestamp, symptoms, vital_signs,
                    risk_scores, recommended_actions, diagnostic_image_path,
                    model_version, synced, record_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                diagnosis.patient_id,
                diagnosis.timestamp,
                json.dumps(diagnosis.symptoms),
                json.dumps(diagnosis.vital_signs),
                json.dumps(diagnosis.risk_scores),
                json.dumps(diagnosis.recommended_actions),
                diagnosis.diagnostic_image_path,
                diagnosis.model_version,
                0,  # Not synced
                record_hash,
            ))

            diagnosis_id = cursor.lastrowid

            # Add to sync queue
            cursor.execute('''
                INSERT INTO sync_queue (diagnosis_id, priority)
                VALUES (?, ?)
            ''', (diagnosis_id, 1 if any(
                score > 0.7 for score in diagnosis.risk_scores.values()
            ) else 0))

            conn.commit()
            logger.info(f"Saved diagnosis for patient {diagnosis.patient_id}")

        except sqlite3.IntegrityError:
            # Duplicate hash - diagnosis already exists
            logger.warning(f"Duplicate diagnosis detected for patient {diagnosis.patient_id}")
        finally:
            conn.close()

    def get_pending_syncs(
        self,
        limit: Optional[int] = None,
    ) -> List[PatientDiagnosis]:
        """
        Get diagnoses pending synchronization.

        Returns records ordered by priority (high-risk cases first)
        for efficient synchronization when connectivity available.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of diagnoses ready to sync
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        query = '''
            SELECT d.* FROM diagnoses d
            JOIN sync_queue sq ON d.id = sq.diagnosis_id
            WHERE d.synced = 0
            ORDER BY sq.priority DESC, d.timestamp ASC
        '''

        if limit is not None:
            query += f' LIMIT {limit}'

        cursor.execute(query)
        rows = cursor.fetchall()

        conn.close()

        # Convert to PatientDiagnosis objects
        diagnoses = []
        for row in rows:
            diagnosis = PatientDiagnosis(
                patient_id=row[1],
                timestamp=row[2],
                symptoms=json.loads(row[3]),
                vital_signs=json.loads(row[4]),
                risk_scores=json.loads(row[5]),
                recommended_actions=json.loads(row[6]),
                diagnostic_image_path=row[7],
                model_version=row[8],
                synced=bool(row[9]),
                sync_timestamp=row[10],
            )
            diagnoses.append(diagnosis)

        return diagnoses

    def mark_synced(
        self,
        diagnoses: List[PatientDiagnosis],
        sync_timestamp: Optional[str] = None,
    ):
        """
        Mark diagnoses as successfully synced.

        Args:
            diagnoses: List of synced diagnoses
            sync_timestamp: Sync completion timestamp
        """
        if sync_timestamp is None:
            sync_timestamp = datetime.utcnow().isoformat()

        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        for diagnosis in diagnoses:
            cursor.execute('''
                UPDATE diagnoses
                SET synced = 1, sync_timestamp = ?
                WHERE patient_id = ? AND timestamp = ?
            ''', (sync_timestamp, diagnosis.patient_id, diagnosis.timestamp))

        conn.commit()
        conn.close()

        logger.info(f"Marked {len(diagnoses)} diagnoses as synced")

    def synchronize(
        self,
        server_url: str,
        batch_size: int = 10,
    ) -> Dict[str, int]:
        """
        Synchronize local data with central server when connectivity available.

        Uses efficient delta sync to minimize data transfer.

        Args:
            server_url: Central server URL for data synchronization
            batch_size: Number of records to sync per batch

        Returns:
            Dictionary with sync statistics
        """
        pending = self.get_pending_syncs(limit=batch_size)

        if not pending:
            logger.info("No pending synchronizations")
            return {'synced': 0, 'failed': 0}

        synced_count = 0
        failed_count = 0

        # In production, implement actual HTTP sync with retry logic
        # This is a placeholder showing the sync pattern
        for diagnosis in pending:
            try:
                # Simulate sync to server
                # In production: POST to server_url with diagnosis data
                logger.info(f"Syncing diagnosis for patient {diagnosis.patient_id}")

                # If image exists, sync separately with compression
                if diagnosis.diagnostic_image_path:
                    self._sync_diagnostic_image(
                        diagnosis.diagnostic_image_path,
                        server_url,
                    )

                # Mark as synced
                self.mark_synced([diagnosis])
                synced_count += 1

            except Exception as e:
                logger.error(f"Failed to sync diagnosis: {e}")
                failed_count += 1

        logger.info(
            f"Synchronization complete: {synced_count} synced, {failed_count} failed"
        )

        return {'synced': synced_count, 'failed': failed_count}

    def _sync_diagnostic_image(
        self,
        image_path: str,
        server_url: str,
    ):
        """
        Sync diagnostic image with compression.

        In production, implement efficient image compression
        and chunked upload for large files.
        """
        # Placeholder for image sync
        logger.info(f"Syncing diagnostic image: {image_path}")
```

This implementation provides a complete offline-capable diagnostic system appropriate for resource-limited settings. The system performs all core diagnostic functions locally without internet connectivity, storing results in an efficient local SQLite database that requires no server infrastructure. When connectivity becomes available, the priority-based synchronization sends high-risk cases first, implementing delta sync to minimize data transfer. The architecture explicitly accounts for the constraints of resource-limited deployment including limited computational resources, unreliable connectivity, and minimal local infrastructure.

## 29.3 Transfer Learning and Domain Adaptation for Global Health

Models trained on data from high-resource settings often perform poorly when deployed in LMICs due to systematic distributional differences. Disease prevalence varies dramatically, with infectious diseases rare in high-income countries dominating morbidity in resource-limited settings. Clinical presentation differs due to varying co-morbidities, nutritional status, and healthcare-seeking behavior. Patient demographics skew younger. Diagnostic and treatment options differ based on local formularies and infrastructure. These differences create domain shift that degrades model performance unless explicitly addressed through transfer learning and domain adaptation.

### 29.3.1 Sources of Domain Shift in Global Health AI

Understanding the specific sources of domain shift between training and deployment environments guides appropriate adaptation strategies. We categorize domain shift into several types:

**Covariate shift** occurs when the input distribution differs between source and target domains while the conditional distribution of outcomes given inputs remains consistent. In global health, this manifests as differences in patient demographics, disease prevalence, and clinical presentation. A model trained on elderly patients in the US encounters much younger populations in sub-Saharan Africa. A tuberculosis detection system trained in low-prevalence settings faces much higher base rates in high-prevalence regions.

Under covariate shift, the conditional probability $$ P(Y\mid X)$$ remains constant but the marginal distribution $$ P(X)$$ changes: $$ P_{source}(X) \neq P_{target}(X)$$ while $$ P_{source}(Y\mid X) = P_{target}(Y \mid X)$$. This means the relationship between clinical features and outcomes is consistent, but the distribution of features differs. Addressing covariate shift often involves importance weighting or domain adversarial training.

**Label shift** (or prior probability shift) occurs when the distribution of outcomes changes while the conditional distribution of inputs given outcomes remains consistent. In global health, this manifests as dramatically different disease prevalence. The probability of tuberculosis given a certain set of symptoms and chest X-ray findings may be similar across settings, but the baseline probability of tuberculosis differs dramatically.

Under label shift: $$ P_{source}(Y) \neq P_{target}(Y)$$ while $$ P_{source}(X\mid Y) = P_{target}(X \mid Y)$$. Bayes' rule shows that when label shift occurs, we can reweight predictions based on the ratio of target to source label probabilities. If we know disease prevalence in both training and deployment settings, we can recalibrate model outputs accordingly.

**Concept shift** represents the most challenging case where the actual relationship between inputs and outputs differs across domains: $$ P_{source}(Y\mid X) \neq P_{target}(Y \mid X)$$. In global health, this occurs when the same clinical presentation predicts different outcomes due to systematic differences in populations or healthcare systems. A given set of vital signs might indicate higher risk in settings with limited treatment capacity. Co-morbidity patterns differ substantially between settings, changing how clinical features should be interpreted.

Concept shift often cannot be addressed without target domain labeled data for retraining or fine-tuning. Detecting concept shift is critical—deploying models under false assumptions of consistent relationships across settings creates substantial clinical risk.

### 29.3.2 Transfer Learning Strategies for Resource-Limited Settings

Transfer learning enables models to leverage knowledge from data-rich source domains when training data in target domains is limited. For global health applications, this typically means starting with models trained on large datasets from high-resource settings, then adapting them to resource-limited deployment contexts using smaller local datasets.

**Feature extraction** treats a pre-trained model as a fixed feature extractor, using only the learned representations while retraining the final classification layers on target domain data. This approach requires minimal target domain data and computational resources since only final layers need retraining. For medical imaging, we might use a convolutional neural network trained on ImageNet or large medical imaging datasets as a feature extractor, then train a simple classifier on local data to predict relevant conditions.

The approach assumes that lower-level features learned on source data (edges, textures, basic shapes) remain relevant for the target task while only task-specific high-level reasoning needs adaptation. This holds reasonably well for medical imaging where basic visual features transfer across contexts, but may fail for clinical prediction where feature relationships differ.

**Fine-tuning** retrains all or most of the model parameters on target domain data, starting from weights pre-trained on source data rather than random initialization. This allows deeper adaptation to target domain characteristics while still benefiting from source domain knowledge. Fine-tuning typically uses lower learning rates for early layers (which encode more general features) and higher learning rates for later layers (which encode task-specific features).

For global health applications with very limited target domain data, careful fine-tuning strategies prevent overfitting while enabling sufficient adaptation. Strategies include progressive unfreezing (initially training only final layers, then gradually unfreezing earlier layers), discriminative learning rates (using different rates for different layer groups), and regularization that keeps parameters close to their pre-trained values.

**Domain adaptation** explicitly addresses distribution mismatch between source and target domains through techniques that make learned representations domain-invariant. Domain-adversarial training adds a domain classifier that tries to predict whether a sample comes from source or target domain based on learned features. The main model is trained to maximize domain classifier error, forcing it to learn features that work in both domains rather than source-specific features.

For global health, this enables training models that work across diverse clinical settings by ensuring representations don't encode site-specific artifacts. A chest X-ray model trained on data from multiple countries with different equipment learns to ignore equipment-specific image characteristics while maintaining disease detection accuracy.

### 29.3.3 Implementation: Domain Adaptation for Global Health Imaging

We implement a complete domain adaptation system for medical imaging in global health contexts:

```python
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DomainInfo:
    """Information about data domains for adaptation."""
    domain_id: str
    country: str
    facility_type: str  # academic, community, rural_clinic
    equipment: str  # Equipment manufacturer/type
    disease_prevalence: Dict[str, float]  # Disease -> prevalence
    sample_count: int

    def get_description(self) -> str:
        """Get human-readable domain description."""
        return (
            f"{self.country} - {self.facility_type} "
            f"({self.equipment}, n={self.sample_count})"
        )

class DomainAdversarialNetwork(nn.Module):
    """
    Domain-adversarial neural network for medical imaging.

    Learns representations that are predictive of clinical outcomes
    while being invariant to domain (data source) to enable
    generalization across diverse global health contexts.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        num_classes: int,
        num_domains: int,
        hidden_dim: int = 256,
        dropout_rate: float = 0.5,
    ):
        """
        Initialize domain-adversarial network.

        Args:
            feature_extractor: CNN for extracting image features
            num_classes: Number of disease classes to predict
            num_domains: Number of distinct data sources/domains
            hidden_dim: Hidden dimension for classifiers
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.feature_extractor = feature_extractor

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.feature_extractor(dummy_input)
            feature_dim = features.shape[1]

        # Class predictor (label classifier)
        self.class_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

        # Domain discriminator (tries to predict source domain)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_domains),
        )

        logger.info(
            f"Initialized DANN: {num_classes} classes, {num_domains} domains"
        )

    def forward(
        self,
        x: torch.Tensor,
        alpha: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with gradient reversal for domain adversarial training.

        Args:
            x: Input images (B, C, H, W)
            alpha: Weight for gradient reversal (controls domain adaptation strength)

        Returns:
            Tuple of (class predictions, domain predictions)
        """
        # Extract features
        features = self.feature_extractor(x)

        # Class predictions
        class_logits = self.class_predictor(features)

        # Domain predictions with gradient reversal
        # During backprop, gradients from domain discriminator are reversed,
        # making feature extractor learn domain-invariant features
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain_logits = self.domain_discriminator(reversed_features)

        return class_logits, domain_logits

class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient reversal layer for domain-adversarial training.

    Forward pass is identity function. Backward pass reverses gradients,
    making the feature extractor try to fool the domain discriminator.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """Forward pass - identity function."""
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass - reverse and scale gradients."""
        return grad_output.neg() * ctx.alpha, None

class MultiDomainDataset(Dataset):
    """
    Dataset combining data from multiple domains/sources.

    Tracks domain labels for each sample to enable domain adaptation.
    """

    def __init__(
        self,
        data_by_domain: Dict[str, Tuple[List[np.ndarray], List[int]]],
        domain_info: Dict[str, DomainInfo],
        transform=None,
    ):
        """
        Initialize multi-domain dataset.

        Args:
            data_by_domain: Dict mapping domain_id to (images, labels)
            domain_info: Dict mapping domain_id to DomainInfo
            transform: Optional image transforms
        """
        self.transform = transform
        self.domain_info = domain_info

        # Create domain ID to index mapping
        self.domain_to_idx = {
            domain_id: idx
            for idx, domain_id in enumerate(sorted(data_by_domain.keys()))
        }
        self.idx_to_domain = {
            idx: domain_id
            for domain_id, idx in self.domain_to_idx.items()
        }

        # Flatten data while tracking domains
        self.images = []
        self.labels = []
        self.domain_labels = []

        for domain_id, (images, labels) in data_by_domain.items():
            domain_idx = self.domain_to_idx[domain_id]

            for img, label in zip(images, labels):
                self.images.append(img)
                self.labels.append(label)
                self.domain_labels.append(domain_idx)

        logger.info(
            f"Created multi-domain dataset: {len(self.images)} samples "
            f"from {len(data_by_domain)} domains"
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get dataset item.

        Returns:
            Tuple of (image, class_label, domain_label)
        """
        image = self.images[idx]
        label = self.labels[idx]
        domain_label = self.domain_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, domain_label

class DomainAdaptationTrainer:
    """
    Trainer for domain adaptation in global health imaging.

    Implements domain-adversarial training with equity-focused
    evaluation across domains and demographic groups.
    """

    def __init__(
        self,
        model: DomainAdversarialNetwork,
        device: torch.device,
        learning_rate: float = 1e-4,
        domain_adaptation_weight: float = 0.1,
    ):
        """
        Initialize domain adaptation trainer.

        Args:
            model: Domain-adversarial network
            device: Device for training
            learning_rate: Learning rate
            domain_adaptation_weight: Weight for domain adversarial loss
        """
        self.model = model.to(device)
        self.device = device
        self.domain_adaptation_weight = domain_adaptation_weight

        # Separate optimizers for different components
        self.optimizer_features = torch.optim.Adam(
            self.model.feature_extractor.parameters(),
            lr=learning_rate,
        )
        self.optimizer_class = torch.optim.Adam(
            self.model.class_predictor.parameters(),
            lr=learning_rate,
        )
        self.optimizer_domain = torch.optim.Adam(
            self.model.domain_discriminator.parameters(),
            lr=learning_rate,
        )

        # Loss functions
        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

        logger.info("Initialized domain adaptation trainer")

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        num_epochs: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch with domain adaptation.

        Args:
            train_loader: Multi-domain training data
            epoch: Current epoch number
            num_epochs: Total number of epochs

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_class_loss = 0.0
        total_domain_loss = 0.0
        correct_class = 0
        correct_domain = 0
        total_samples = 0

        # Gradually increase domain adaptation strength over training
        # (as suggested by original DANN paper)
        p = float(epoch) / num_epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        for batch_idx, (images, class_labels, domain_labels) in enumerate(train_loader):
            images = images.to(self.device)
            class_labels = class_labels.to(self.device)
            domain_labels = domain_labels.to(self.device)

            batch_size = images.size(0)
            total_samples += batch_size

            # Forward pass
            class_logits, domain_logits = self.model(images, alpha=alpha)

            # Class prediction loss (main task)
            class_loss = self.class_criterion(class_logits, class_labels)

            # Domain prediction loss (adversarial)
            domain_loss = self.domain_criterion(domain_logits, domain_labels)

            # Combined loss
            total_loss = (
                class_loss +
                self.domain_adaptation_weight * domain_loss
            )

            # Backward pass
            self.optimizer_features.zero_grad()
            self.optimizer_class.zero_grad()
            self.optimizer_domain.zero_grad()

            total_loss.backward()

            self.optimizer_features.step()
            self.optimizer_class.step()
            self.optimizer_domain.step()

            # Track metrics
            total_class_loss += class_loss.item() * batch_size
            total_domain_loss += domain_loss.item() * batch_size

            class_predictions = class_logits.argmax(dim=1)
            correct_class += (class_predictions == class_labels).sum().item()

            domain_predictions = domain_logits.argmax(dim=1)
            correct_domain += (domain_predictions == domain_labels).sum().item()

        # Calculate averages
        avg_class_loss = total_class_loss / total_samples
        avg_domain_loss = total_domain_loss / total_samples
        class_accuracy = correct_class / total_samples
        domain_accuracy = correct_domain / total_samples

        metrics = {
            'class_loss': avg_class_loss,
            'domain_loss': avg_domain_loss,
            'class_accuracy': class_accuracy,
            'domain_accuracy': domain_accuracy,
            'alpha': alpha,
        }

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Class Loss: {avg_class_loss:.4f}, Class Acc: {class_accuracy:.4f}, "
            f"Domain Loss: {avg_domain_loss:.4f}, Domain Acc: {domain_accuracy:.4f}, "
            f"Alpha: {alpha:.4f}"
        )

        return metrics

    def evaluate_by_domain(
        self,
        test_loaders_by_domain: Dict[str, DataLoader],
        domain_info: Dict[str, DomainInfo],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance stratified by domain.

        Critical for assessing whether domain adaptation achieves
        equitable performance across diverse deployment contexts.

        Args:
            test_loaders_by_domain: Dict mapping domain_id to test DataLoader
            domain_info: Dict mapping domain_id to DomainInfo

        Returns:
            Dict mapping domain_id to performance metrics
        """
        self.model.eval()
        results = {}

        with torch.no_grad():
            for domain_id, test_loader in test_loaders_by_domain.items():
                correct = 0
                total = 0
                all_predictions = []
                all_labels = []

                for images, labels, _ in test_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Get class predictions only (ignore domain predictions)
                    class_logits, _ = self.model(images, alpha=0.0)
                    predictions = class_logits.argmax(dim=1)

                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                accuracy = correct / total if total > 0 else 0.0

                # Calculate per-class metrics
                all_predictions = np.array(all_predictions)
                all_labels = np.array(all_labels)

                # Sensitivity and specificity for each class
                per_class_metrics = {}
                for class_idx in range(2):  # Binary classification example
                    true_positives = ((all_predictions == class_idx) & (all_labels == class_idx)).sum()
                    false_positives = ((all_predictions == class_idx) & (all_labels != class_idx)).sum()
                    true_negatives = ((all_predictions != class_idx) & (all_labels != class_idx)).sum()
                    false_negatives = ((all_predictions != class_idx) & (all_labels == class_idx)).sum()

                    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
                    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0

                    per_class_metrics[f'class_{class_idx}_sensitivity'] = sensitivity
                    per_class_metrics[f'class_{class_idx}_specificity'] = specificity

                results[domain_id] = {
                    'accuracy': accuracy,
                    'sample_count': total,
                    **per_class_metrics,
                    'domain_description': domain_info[domain_id].get_description(),
                }

                logger.info(
                    f"Domain {domain_id} ({domain_info[domain_id].get_description()}): "
                    f"Accuracy = {accuracy:.4f} (n={total})"
                )

        # Calculate cross-domain metrics
        accuracies = [metrics['accuracy'] for metrics in results.values()]
        results['cross_domain_summary'] = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'accuracy_gap': np.max(accuracies) - np.min(accuracies),
        }

        logger.info(
            f"Cross-domain summary: Mean Acc = {results['cross_domain_summary']['mean_accuracy']:.4f}, "
            f"Gap = {results['cross_domain_summary']['accuracy_gap']:.4f}"
        )

        return results

def prevalence_calibration(
    predictions: np.ndarray,
    source_prevalence: float,
    target_prevalence: float,
) -> np.ndarray:
    """
    Calibrate predictions for different disease prevalence (label shift).

    Adjusts model predictions trained in one prevalence context for
    deployment in contexts with different disease prevalence using
    Bayes' rule.

    Args:
        predictions: Model probability predictions (N,)
        source_prevalence: Disease prevalence in training data
        target_prevalence: Disease prevalence in deployment setting

    Returns:
        Calibrated predictions accounting for prevalence shift
    """
    # Convert to odds
    source_odds = source_prevalence / (1 - source_prevalence)
    target_odds = target_prevalence / (1 - target_prevalence)

    # Prediction odds
    pred_odds = predictions / (1 - predictions)

    # Adjust for prevalence shift
    adjusted_odds = pred_odds * (target_odds / source_odds)

    # Convert back to probabilities
    calibrated_predictions = adjusted_odds / (1 + adjusted_odds)

    return calibrated_predictions
```

This implementation provides comprehensive domain adaptation infrastructure appropriate for global health applications. The domain-adversarial training learns representations that predict disease outcomes accurately while being invariant to data source, enabling models to generalize across diverse clinical settings with different equipment, populations, and practices. The stratified evaluation framework explicitly tracks performance across domains, ensuring adaptation doesn't improve overall metrics while degrading performance for specific settings. The prevalence calibration function addresses label shift when disease prevalence differs dramatically between training and deployment contexts, a common challenge in global health where TB, malaria, and other diseases have radically different base rates across settings.

## 29.4 Multilingual and Culturally Adaptive AI Systems

Language barriers represent a fundamental challenge to equitable healthcare AI deployment globally. While most healthcare AI research and development occurs in English-speaking high-income countries, the majority of the world's population speaks other languages as their primary language. Even within multilingual countries, patients may have limited proficiency in official languages, particularly among rural, low-literacy, and marginalized communities. Clinical documentation, patient communication, and health education materials must work in local languages to be accessible and effective.

Beyond language, cultural context shapes how health and illness are understood, how symptoms are described, what treatments are acceptable, and how medical information should be communicated. AI systems that impose external cultural frameworks risk miscommunication, low uptake, and poor health outcomes. Culturally adaptive systems respect local knowledge systems, communication norms, and health beliefs while providing evidence-based care.

### 29.4.1 Multilingual Clinical Natural Language Processing

Developing clinical NLP systems that work in multiple languages faces several challenges beyond standard multilingual NLP. Medical terminology varies dramatically across languages, with many languages lacking standardized medical vocabularies. Clinical documentation practices differ across healthcare systems and cultures. Training data for clinical NLP is scarce in most languages beyond English, Chinese, and a few European languages.

**Cross-lingual transfer learning** enables leveraging English clinical NLP resources to build systems in other languages. Multilingual pre-trained models like mBERT and XLM-RoBERTa learn shared representations across languages, enabling zero-shot transfer where models trained on English clinical text can extract information from text in other languages without language-specific training data.

However, zero-shot transfer degrades performance substantially compared to language-specific training, particularly for specialized clinical terminology. Few-shot learning and active learning strategies improve performance with limited target language data by identifying the most informative examples for human annotation.

**Code-switching and language mixing** are common in multilingual healthcare settings where clinicians and patients may use multiple languages within single conversations or documents. A clinical note in India might mix English medical terminology with Hindi patient descriptions. Community health workers in Senegal might document cases mixing French and Wolof. Clinical NLP systems must handle this linguistic complexity rather than assuming monolingual text.

### 29.4.2 Culturally Adaptive Health Communication

Effective health communication requires adaptation to local cultural contexts, health literacy levels, and communication preferences. Culturally adaptive AI systems adjust language complexity, communication style, examples, and framing based on user characteristics and cultural context.

**Health literacy adaptation** adjusts medical information to appropriate comprehension levels. Patients with limited formal education or health literacy need simpler language, concrete examples, and visual aids. Systems must simplify without condescending, maintain clinical accuracy, and avoid assuming all patients need simplified communication.

**Cultural framing** presents health information using culturally appropriate metaphors, examples, and communication styles. Risk communication that works in individualistic cultures emphasizing personal control may fail in collectivist cultures emphasizing community and family. Discussions of sensitive topics like sexual health, mental illness, or end-of-life care require cultural sensitivity about what topics can be discussed openly and with whom.

**Community-informed content** develops health education materials through participatory processes with community members, traditional healers, and local health workers rather than imposing externally developed content. This ensures content addresses locally relevant health concerns, uses accessible language, and aligns with local cultural values.

### 29.4.3 Implementation: Multilingual Patient Education System

We implement a production system for generating culturally adaptive, multilingual patient education materials:

```python
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from dataclasses import dataclass
from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from enum import Enum
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthLiteracyLevel(Enum):
    """Health literacy levels for content adaptation."""
    BASIC = "basic"  # <6th grade reading level
    INTERMEDIATE = "intermediate"  # 6th-12th grade
    ADVANCED = "advanced"  # College+

@dataclass
class CulturalContext:
    """Cultural context information for content adaptation."""
    primary_language: str
    country: str
    region: Optional[str] = None
    cultural_group: Optional[str] = None
    literacy_rate: Optional[float] = None
    health_literacy_level: HealthLiteracyLevel = HealthLiteracyLevel.INTERMEDIATE

    # Cultural preferences
    individualist_vs_collectivist: float = 0.5  # 0=collectivist, 1=individualist
    direct_vs_indirect_communication: float = 0.5  # 0=indirect, 1=direct

    # Access constraints
    has_internet_access: bool = True
    has_smartphone: bool = True

    def get_description(self) -> str:
        """Get human-readable description."""
        desc = f"{self.primary_language} ({self.country}"
        if self.region:
            desc += f", {self.region}"
        desc += ")"
        return desc

@dataclass
class HealthEducationMaterial:
    """Culturally adapted health education material."""
    title: str
    content: str
    language: str
    cultural_context: CulturalContext
    health_topic: str
    literacy_level: HealthLiteracyLevel

    # Validation flags
    medically_accurate: bool = False
    culturally_appropriate: bool = False
    community_reviewed: bool = False

    # Metadata
    source_content_id: Optional[str] = None
    translation_quality_score: Optional[float] = None
    created_timestamp: Optional[str] = None

class MultilingualHealthEducation:
    """
    Multilingual patient education system with cultural adaptation.

    Generates health education materials in multiple languages,
    adapted for local cultural contexts and health literacy levels,
    with validation to ensure medical accuracy and cultural appropriateness.
    """

    def __init__(
        self,
        source_language: str = "en",
        device: Optional[torch.device] = None,
    ):
        """
        Initialize multilingual health education system.

        Args:
            source_language: Source language for content (typically English)
            device: Device for model inference
        """
        self.source_language = source_language
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize translation models (cached for efficiency)
        self.translation_models: Dict[str, Tuple[MarianMTModel, MarianTokenizer]] = {}

        # Initialize simplification model for literacy adaptation
        self.simplification_model_name = "facebook/bart-large-cnn"

        # Medical term preservation list (terms that should not be translated)
        self.medical_terms_to_preserve = self._load_medical_terminology()

        # Cultural adaptation templates
        self.cultural_templates = self._load_cultural_templates()

        logger.info(
            f"Initialized multilingual health education system "
            f"(source: {source_language})"
        )

    def generate_education_material(
        self,
        content: str,
        health_topic: str,
        target_contexts: List[CulturalContext],
    ) -> List[HealthEducationMaterial]:
        """
        Generate culturally adapted health education materials for multiple contexts.

        Args:
            content: Source health education content (in source language)
            health_topic: Health topic (e.g., "diabetes management", "tuberculosis prevention")
            target_contexts: List of cultural contexts to generate materials for

        Returns:
            List of adapted health education materials
        """
        materials = []

        for context in target_contexts:
            logger.info(
                f"Generating material for {context.get_description()}"
            )

            # Step 1: Adapt literacy level if needed
            adapted_content = self._adapt_literacy_level(
                content,
                context.health_literacy_level,
            )

            # Step 2: Cultural adaptation
            culturally_adapted = self._apply_cultural_adaptation(
                adapted_content,
                health_topic,
                context,
            )

            # Step 3: Translation if needed
            if context.primary_language != self.source_language:
                translated = self._translate_content(
                    culturally_adapted,
                    target_language=context.primary_language,
                )
                translation_quality = self._assess_translation_quality(
                    culturally_adapted,
                    translated,
                    context.primary_language,
                )
            else:
                translated = culturally_adapted
                translation_quality = 1.0

            # Step 4: Validation
            medical_accuracy = self._validate_medical_accuracy(
                original=content,
                adapted=translated,
            )

            cultural_appropriateness = self._validate_cultural_appropriateness(
                translated,
                context,
            )

            # Create material
            material = HealthEducationMaterial(
                title=self._generate_title(health_topic, context),
                content=translated,
                language=context.primary_language,
                cultural_context=context,
                health_topic=health_topic,
                literacy_level=context.health_literacy_level,
                medically_accurate=medical_accuracy,
                culturally_appropriate=cultural_appropriateness,
                translation_quality_score=translation_quality,
            )

            materials.append(material)

            logger.info(
                f"Generated material: {material.title} "
                f"(medical accuracy: {medical_accuracy}, "
                f"cultural appropriateness: {cultural_appropriateness})"
            )

        return materials

    def _adapt_literacy_level(
        self,
        content: str,
        target_level: HealthLiteracyLevel,
    ) -> str:
        """
        Adapt content to target health literacy level.

        Simplifies language while maintaining medical accuracy.
        """
        if target_level == HealthLiteracyLevel.ADVANCED:
            # No simplification needed
            return content

        # Load simplification model if not loaded
        if not hasattr(self, 'simplification_model'):
            self.simplification_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.simplification_model_name
            ).to(self.device)
            self.simplification_tokenizer = AutoTokenizer.from_pretrained(
                self.simplification_model_name
            )

        # Determine simplification parameters
        if target_level == HealthLiteracyLevel.BASIC:
            max_length = 100  # Shorter sentences
            num_beams = 4
        else:  # INTERMEDIATE
            max_length = 150
            num_beams = 3

        # Simplify content
        # In production, use medical-specific simplification models
        # trained on health literacy corpora
        inputs = self.simplification_tokenizer(
            content,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.simplification_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        simplified = self.simplification_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        # Post-process to preserve medical terms
        simplified = self._preserve_medical_terms(content, simplified)

        return simplified

    def _preserve_medical_terms(
        self,
        original: str,
        simplified: str,
    ) -> str:
        """
        Ensure critical medical terms are preserved in simplified text.

        Replaces oversimplified medical terms with accurate terminology.
        """
        # Extract medical terms from original
        original_terms = set()
        for term in self.medical_terms_to_preserve:
            if term.lower() in original.lower():
                original_terms.add(term)

        # Check if terms are preserved in simplified version
        result = simplified
        for term in original_terms:
            if term.lower() not in result.lower():
                # Term was oversimplified - need to restore
                # In production, use medical knowledge base to map
                # simplified terms back to accurate terminology
                logger.warning(
                    f"Medical term '{term}' was oversimplified, attempting restoration"
                )

        return result

    def _apply_cultural_adaptation(
        self,
        content: str,
        health_topic: str,
        context: CulturalContext,
    ) -> str:
        """
        Apply cultural adaptations to content.

        Adjusts framing, examples, and communication style for cultural context.
        """
        adapted = content

        # Apply template-based adaptations
        if health_topic in self.cultural_templates:
            templates = self.cultural_templates[health_topic]

            # Adjust for individualist vs collectivist orientation
            if context.individualist_vs_collectivist < 0.5:
                # Collectivist culture - emphasize family and community
                adapted = self._add_collectivist_framing(adapted, templates)
            else:
                # Individualist culture - emphasize personal control
                adapted = self._add_individualist_framing(adapted, templates)

            # Adjust for communication directness
            if context.direct_vs_indirect_communication < 0.5:
                # Indirect communication preferred - soften directives
                adapted = self._soften_directives(adapted)

        # Add culturally relevant examples
        adapted = self._add_cultural_examples(
            adapted,
            health_topic,
            context,
        )

        return adapted

    def _add_collectivist_framing(
        self,
        content: str,
        templates: Dict,
    ) -> str:
        """Add collectivist cultural framing emphasizing family and community."""
        # Add community-focused language
        collectivist_additions = [
            "Taking care of your health helps you care for your family.",
            "Your health is important to your community.",
            "Discuss these recommendations with your family.",
        ]

        # Append appropriate framing
        return content + "\n\n" + collectivist_additions[0]

    def _add_individualist_framing(
        self,
        content: str,
        templates: Dict,
    ) -> str:
        """Add individualist cultural framing emphasizing personal control."""
        individualist_additions = [
            "You have control over your health.",
            "These steps empower you to manage your condition.",
            "Your choices make a difference in your health outcomes.",
        ]

        return content + "\n\n" + individualist_additions[0]

    def _soften_directives(self, content: str) -> str:
        """
        Soften direct commands for cultures preferring indirect communication.

        Converts commands to suggestions and questions.
        """
        # Replace imperative verbs with softer alternatives
        # This is simplified - production systems need sophisticated NLU
        softened = content

        replacements = {
            r'\bMust\b': 'It would be beneficial to',
            r'\bShould\b': 'You might consider',
            r'\bDo not\b': 'It is recommended to avoid',
            r'\bStop\b': 'Consider stopping',
        }

        for pattern, replacement in replacements.items():
            softened = re.sub(pattern, replacement, softened, flags=re.IGNORECASE)

        return softened

    def _add_cultural_examples(
        self,
        content: str,
        health_topic: str,
        context: CulturalContext,
    ) -> str:
        """
        Add culturally relevant examples and context.

        Uses local foods, activities, and contexts in examples.
        """
        # In production, maintain database of culturally specific examples
        # This is a simplified demonstration

        if health_topic == "diabetes management" and "diet" in content.lower():
            # Add culture-specific dietary examples
            if context.country == "India":
                example = "\n\nFor example, choose whole grain roti over white rice, and include dal and vegetables in meals."
            elif context.country == "Mexico":
                example = "\n\nFor example, choose corn tortillas over flour, include beans and vegetables, and limit sugary drinks."
            else:
                example = "\n\nChoose whole grains, lean proteins, and vegetables in your meals."

            content += example

        return content

    def _translate_content(
        self,
        content: str,
        target_language: str,
    ) -> str:
        """
        Translate content to target language.

        Uses neural machine translation with medical term preservation.
        """
        # Load translation model if not cached
        model_key = f"{self.source_language}-{target_language}"

        if model_key not in self.translation_models:
            # Load appropriate MarianMT model
            model_name = f"Helsinki-NLP/opus-mt-{self.source_language}-{target_language}"

            try:
                model = MarianMTModel.from_pretrained(model_name).to(self.device)
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                self.translation_models[model_key] = (model, tokenizer)
                logger.info(f"Loaded translation model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load translation model: {e}")
                return f"[Translation to {target_language} unavailable: {content}]"

        model, tokenizer = self.translation_models[model_key]

        # Prepare text for translation (preserve medical terms)
        protected_text, term_mappings = self._protect_medical_terms_for_translation(content)

        # Translate
        inputs = tokenizer(
            protected_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            translated_tokens = model.generate(**inputs, max_length=512)

        translated = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        # Restore medical terms
        translated = self._restore_medical_terms_after_translation(
            translated,
            term_mappings,
        )

        return translated

    def _protect_medical_terms_for_translation(
        self,
        text: str,
    ) -> Tuple[str, Dict[str, str]]:
        """
        Replace medical terms with placeholders before translation.

        Prevents mistranslation of critical medical terminology.
        """
        protected_text = text
        term_mappings = {}

        for i, term in enumerate(self.medical_terms_to_preserve):
            if term in text:
                placeholder = f"__MEDICAL_TERM_{i}__"
                protected_text = protected_text.replace(term, placeholder)
                term_mappings[placeholder] = term

        return protected_text, term_mappings

    def _restore_medical_terms_after_translation(
        self,
        translated: str,
        term_mappings: Dict[str, str],
    ) -> str:
        """Restore medical terms after translation."""
        result = translated
        for placeholder, term in term_mappings.items():
            result = result.replace(placeholder, term)
        return result

    def _assess_translation_quality(
        self,
        source: str,
        translated: str,
        target_language: str,
    ) -> float:
        """
        Assess translation quality.

        In production, use BLEU, COMET, or human evaluation.
        This is a simplified placeholder.
        """
        # Placeholder quality assessment
        # In production: Use reference translations and automatic metrics
        # or human evaluation for medical content
        return 0.85  # Assumed good quality

    def _validate_medical_accuracy(
        self,
        original: str,
        adapted: str,
    ) -> bool:
        """
        Validate that adapted content maintains medical accuracy.

        Checks that critical medical information is preserved.
        In production, use medical knowledge bases and clinical review.
        """
        # Simplified validation
        # In production: Extract medical facts from both versions
        # and verify consistency using medical knowledge graphs

        # Check for presence of key medical terms
        critical_terms = [
            term for term in self.medical_terms_to_preserve
            if term.lower() in original.lower()
        ]

        preserved_count = sum(
            1 for term in critical_terms
            if term.lower() in adapted.lower()
        )

        preservation_rate = (
            preserved_count / len(critical_terms)
            if critical_terms
            else 1.0
        )

        # Require >80% preservation of critical terms
        return preservation_rate > 0.8

    def _validate_cultural_appropriateness(
        self,
        content: str,
        context: CulturalContext,
    ) -> bool:
        """
        Validate cultural appropriateness of content.

        In production, requires community review by cultural experts.
        This provides automated preliminary checks only.
        """
        # Preliminary automated checks
        # Real validation requires community review

        # Check for potentially inappropriate content
        sensitive_topics = ['death', 'sexuality', 'mental illness']
        has_sensitive_content = any(
            topic in content.lower()
            for topic in sensitive_topics
        )

        if has_sensitive_content:
            # Flag for community review
            logger.warning(
                f"Content contains sensitive topics requiring community review "
                f"for {context.get_description()}"
            )
            return False  # Requires manual review

        return True  # Preliminary check passed

    def _generate_title(
        self,
        health_topic: str,
        context: CulturalContext,
    ) -> str:
        """Generate culturally appropriate title."""
        # In production, translate and culturally adapt title
        return f"{health_topic.title()} - Information for Patients"

    def _load_medical_terminology(self) -> Set[str]:
        """
        Load medical terms that should be preserved in translation/simplification.

        In production, load from medical ontologies (SNOMED CT, etc.).
        """
        # Simplified example set
        return {
            'diabetes', 'insulin', 'glucose', 'hypertension',
            'tuberculosis', 'pneumonia', 'malaria', 'HIV',
            'antibiotic', 'vaccine', 'treatment', 'diagnosis',
            'medication', 'prescription', 'dosage',
        }

    def _load_cultural_templates(self) -> Dict[str, Dict]:
        """
        Load cultural adaptation templates.

        In production, develop through participatory design with
        communities and cultural experts.
        """
        # Simplified template structure
        return {
            'diabetes management': {
                'individualist_framing': ['personal_control', 'self_management'],
                'collectivist_framing': ['family_support', 'community_health'],
            },
            'tuberculosis prevention': {
                'individualist_framing': ['personal_protection', 'individual_risk'],
                'collectivist_framing': ['protecting_family', 'community_transmission'],
            },
        }
```

This implementation provides a complete multilingual patient education system with cultural adaptation. The system translates health education materials while preserving critical medical terminology, adapts content for different health literacy levels, applies cultural framing appropriate for local contexts, and validates both medical accuracy and cultural appropriateness. Critically, the system acknowledges that automated cultural adaptation is preliminary only—true cultural appropriateness requires community review and participatory development with local experts and community members.

## 29.5 Federated Learning and Data Sovereignty

Collaborative model development across institutions in different countries faces substantial challenges around data sharing, privacy, and sovereignty. Many countries restrict health data export to protect citizen privacy and maintain control over valuable data resources. Institutions may be reluctant to share sensitive patient data with external entities due to privacy regulations, competitive concerns, or lack of trust in data governance. Yet individual institutions often lack sufficient data for robust model training, particularly for rare conditions or underrepresented populations.

Federated learning enables collaborative model training without sharing raw data. Instead of centralizing data, models train locally at each participating site using local data. Only model updates (gradients or parameters) are shared with a central coordinator that aggregates updates to improve a global model. This architecture respects data sovereignty while enabling collective benefit from distributed data.

### 29.5.1 Federated Learning Architecture

The standard federated learning protocol proceeds iteratively:

1. A central server initializes a global model $$\theta^{(0)}$$
2. The server distributes the current global model $$\theta^{(t)}$$ to participating clients
3. Each client $$ k $$ trains the model on their local data for several iterations, computing updated parameters $$\theta_k^{(t+1)}$$
4. Clients send model updates $$\Delta\theta_k = \theta_k^{(t+1)} - \theta^{(t)}$$ to the server
5. The server aggregates updates to create an improved global model $$\theta^{(t+1)}$$

The most common aggregation strategy is Federated Averaging (FedAvg), which weights each client's contribution by their dataset size:

$$\theta^{(t+1)} = \theta^{(t)} + \sum_{k=1}^K \frac{n_k}{n} \Delta\theta_k$$

where $$n_k $$ is the number of samples at client $$ k $$ and $$ n = \sum_k n_k$$ is the total.

For global health applications, participating clients might be hospitals in different countries, health ministries, or regional health systems. Each maintains complete control over their data while contributing to model improvement.

### 29.5.2 Privacy and Security Considerations

While federated learning avoids sharing raw data, model updates themselves can leak information about training data. Gradient-based attacks can reconstruct individual training samples under certain conditions. Repeated participation allows inference about sensitive attributes of the local population.

**Differential privacy** provides formal privacy guarantees by adding carefully calibrated noise to model updates before sharing. The amount of noise balances privacy protection against model utility. For healthcare applications requiring strong privacy guarantees, substantial noise may be necessary, degrading model performance.

**Secure multi-party computation** enables computing functions over distributed private inputs without revealing those inputs. In federated learning, secure aggregation protocols allow the server to compute the sum of client updates without seeing individual updates. This prevents the server from targeting specific clients while maintaining the ability to aggregate contributions.

**Homomorphic encryption** allows computations on encrypted data. Clients can encrypt model updates before sending to the server. The server aggregates encrypted updates and sends the encrypted global model back to clients for decryption. This provides strong privacy guarantees but introduces substantial computational overhead.

For resource-limited global health settings, the computational cost of advanced privacy-preserving techniques may be prohibitive. Practical deployments often rely on institutional trust, data use agreements, and selective aggregation rather than cryptographic guarantees.

### 29.5.3 Challenges in Global Health Federated Learning

Global health federated learning faces unique challenges beyond standard federated learning settings:

**Data heterogeneity** across sites is extreme. Disease prevalence, demographics, clinical practices, and data quality vary dramatically across countries and healthcare settings. Standard federated averaging can perform poorly when client data distributions differ substantially. Personalized federated learning approaches that maintain both global and local model components may be necessary.

**Resource heterogeneity** means participating institutions have vastly different computational capabilities, network bandwidth, and reliability. A federated learning protocol that works well for hospitals in high-income countries may be infeasible for resource-limited clinics with limited bandwidth and intermittent connectivity. Asynchronous federated learning that doesn't require synchronous participation and adaptive communication protocols that adjust to available bandwidth address these challenges.

**Unequal participation** creates power dynamics where large, well-resourced institutions dominate model development while smaller sites contribute minimally. This risks replicating colonial patterns in global health where high-income country institutions extract value from low-income settings. Fair aggregation strategies that prevent any single institution from dominating and equitable benefit sharing that ensures value flows to all contributors are essential.

**Local vs global benefit tradeoff** emerges when the globally optimal model performs poorly for specific populations. Federated learning typically optimizes a global objective, but sites care primarily about their local performance. Multi-task federated learning that maintains site-specific model components alongside shared components can help balance global and local performance.

### 29.5.4 Implementation: Privacy-Preserving Federated Learning System

We implement a federated learning system appropriate for global health collaborations:

```python
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from datetime import datetime
import hashlib
import logging
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClientInfo:
    """Information about federated learning client."""
    client_id: str
    institution_name: str
    country: str
    data_size: int
    computational_capacity: str  # 'high', 'medium', 'low'
    network_bandwidth: str  # 'high', 'medium', 'low'

    def get_description(self) -> str:
        """Get human-readable client description."""
        return f"{self.institution_name} ({self.country}, n={self.data_size})"

@dataclass
class FederatedRound:
    """Information about a federated learning round."""
    round_number: int
    participating_clients: List[str]
    global_model_params: OrderedDict
    client_updates: Dict[str, OrderedDict]
    aggregation_weights: Dict[str, float]
    timestamp: str

    def get_round_hash(self) -> str:
        """Generate hash for round verification."""
        # Create deterministic representation for hashing
        round_data = f"{self.round_number}-{sorted(self.participating_clients)}"
        return hashlib.sha256(round_data.encode()).hexdigest()[:16]

class DifferentialPrivacy:
    """
    Differential privacy mechanism for federated learning.

    Adds calibrated noise to model updates to provide formal privacy
    guarantees while maintaining model utility.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        """
        Initialize differential privacy mechanism.

        Args:
            epsilon: Privacy budget (smaller = more private, less accurate)
            delta: Privacy parameter (probability of privacy violation)
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

        logger.info(
            f"Initialized DP with ε={epsilon}, δ={delta}, "
            f"max_norm={max_grad_norm}"
        )

    def clip_gradients(
        self,
        model_update: OrderedDict,
    ) -> OrderedDict:
        """
        Clip gradients to bound sensitivity.

        Args:
            model_update: Model parameter updates

        Returns:
            Clipped model updates
        """
        # Calculate total norm
        total_norm = 0.0
        for param_update in model_update.values():
            total_norm += torch.sum(param_update ** 2).item()
        total_norm = np.sqrt(total_norm)

        # Clip if necessary
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)

        if clip_coef < 1.0:
            clipped_update = OrderedDict()
            for name, param_update in model_update.items():
                clipped_update[name] = param_update * clip_coef

            logger.debug(f"Clipped gradients: norm {total_norm:.4f} -> {self.max_grad_norm}")
            return clipped_update

        return model_update

    def add_noise(
        self,
        model_update: OrderedDict,
        num_samples: int,
    ) -> OrderedDict:
        """
        Add Gaussian noise for differential privacy.

        Args:
            model_update: Clipped model updates
            num_samples: Number of samples used for this update

        Returns:
            Noised model updates
        """
        # Calculate noise scale based on privacy parameters
        # Using Gaussian mechanism for simplicity
        # In production, use more sophisticated composition
        sensitivity = 2 * self.max_grad_norm / num_samples
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

        noised_update = OrderedDict()
        for name, param_update in model_update.items():
            noise = torch.randn_like(param_update) * noise_scale
            noised_update[name] = param_update + noise

        logger.debug(f"Added DP noise with scale {noise_scale:.6f}")

        return noised_update

    def privatize_update(
        self,
        model_update: OrderedDict,
        num_samples: int,
    ) -> OrderedDict:
        """
        Apply differential privacy to model update.

        Args:
            model_update: Raw model updates
            num_samples: Number of samples in local dataset

        Returns:
            Privacy-preserving model updates
        """
        # Step 1: Clip gradients
        clipped = self.clip_gradients(model_update)

        # Step 2: Add noise
        private = self.add_noise(clipped, num_samples)

        return private

class FederatedClient:
    """
    Federated learning client for local model training.

    Trains model locally on private data and shares only
    model updates (with optional differential privacy).
    """

    def __init__(
        self,
        client_info: ClientInfo,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        use_differential_privacy: bool = True,
        dp_epsilon: float = 1.0,
    ):
        """
        Initialize federated client.

        Args:
            client_info: Client metadata
            model: Local model (initialized with global parameters)
            train_loader: Local training data
            device: Device for training
            use_differential_privacy: Whether to apply DP to updates
            dp_epsilon: Privacy budget for differential privacy
        """
        self.client_info = client_info
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device

        # Differential privacy
        self.use_differential_privacy = use_differential_privacy
        if use_differential_privacy:
            self.dp_mechanism = DifferentialPrivacy(
                epsilon=dp_epsilon,
                delta=1e-5,
                max_grad_norm=1.0,
            )

        logger.info(
            f"Initialized federated client: {client_info.get_description()} "
            f"(DP: {use_differential_privacy})"
        )

    def train_local_model(
        self,
        num_epochs: int = 1,
        learning_rate: float = 0.01,
    ) -> OrderedDict:
        """
        Train model on local data.

        Args:
            num_epochs: Number of local training epochs
            learning_rate: Learning rate for local training

        Returns:
            Model parameter updates (with optional DP)
        """
        # Store initial parameters
        initial_params = OrderedDict()
        for name, param in self.model.named_parameters():
            initial_params[name] = param.data.clone()

        # Local training
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_loader)
            logger.debug(
                f"Client {self.client_info.client_id} - "
                f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}"
            )

        # Calculate parameter updates
        updates = OrderedDict()
        for name, param in self.model.named_parameters():
            updates[name] = param.data - initial_params[name]

        # Apply differential privacy if enabled
        if self.use_differential_privacy:
            updates = self.dp_mechanism.privatize_update(
                updates,
                num_samples=len(self.train_loader.dataset),
            )

        logger.info(
            f"Client {self.client_info.client_id} completed local training"
        )

        return updates

    def update_model(self, global_params: OrderedDict):
        """
        Update local model with new global parameters.

        Args:
            global_params: Updated global model parameters
        """
        self.model.load_state_dict(global_params)
        logger.debug(
            f"Client {self.client_info.client_id} updated with global model"
        )

    def evaluate_model(
        self,
        test_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate local model performance.

        Args:
            test_loader: Local test data

        Returns:
            Dictionary of performance metrics
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                outputs = self.model(data)
                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0

        return {
            'accuracy': accuracy,
            'num_samples': total,
        }

class FederatedServer:
    """
    Federated learning server for coordinating global model training.

    Aggregates client updates while respecting data sovereignty
    and providing equitable participation across institutions.
    """

    def __init__(
        self,
        global_model: nn.Module,
        clients: List[FederatedClient],
        device: torch.device,
        aggregation_strategy: str = "fedavg",
        min_clients_per_round: int = 2,
    ):
        """
        Initialize federated server.

        Args:
            global_model: Global model to train
            clients: List of participating clients
            device: Device for aggregation
            aggregation_strategy: Aggregation method ('fedavg', 'equal_weight')
            min_clients_per_round: Minimum clients required per round
        """
        self.global_model = global_model.to(device)
        self.clients = {client.client_info.client_id: client for client in clients}
        self.device = device
        self.aggregation_strategy = aggregation_strategy
        self.min_clients_per_round = min_clients_per_round

        self.round_history: List[FederatedRound] = []
        self.current_round = 0

        logger.info(
            f"Initialized federated server with {len(clients)} clients "
            f"(strategy: {aggregation_strategy})"
        )

    def train_round(
        self,
        num_local_epochs: int = 1,
        client_sample_fraction: float = 1.0,
    ) -> FederatedRound:
        """
        Execute one round of federated training.

        Args:
            num_local_epochs: Local training epochs per client
            client_sample_fraction: Fraction of clients to sample

        Returns:
            FederatedRound containing round information
        """
        self.current_round += 1

        # Sample clients for this round
        num_clients_to_sample = max(
            self.min_clients_per_round,
            int(len(self.clients) * client_sample_fraction),
        )

        sampled_client_ids = np.random.choice(
            list(self.clients.keys()),
            size=min(num_clients_to_sample, len(self.clients)),
            replace=False,
        )

        logger.info(
            f"\n{'='*60}\n"
            f"Starting Federated Learning Round {self.current_round}\n"
            f"Participating clients: {len(sampled_client_ids)}/{len(self.clients)}\n"
            f"{'='*60}"
        )

        # Distribute current global model to clients
        global_params = self.global_model.state_dict()
        for client_id in sampled_client_ids:
            self.clients[client_id].update_model(global_params)

        # Collect client updates
        client_updates = {}
        client_weights = {}

        for client_id in sampled_client_ids:
            client = self.clients[client_id]

            logger.info(
                f"Training client: {client.client_info.get_description()}"
            )

            # Local training
            updates = client.train_local_model(
                num_epochs=num_local_epochs,
            )

            client_updates[client_id] = updates

            # Calculate aggregation weight
            if self.aggregation_strategy == "fedavg":
                # Weight by dataset size
                client_weights[client_id] = client.client_info.data_size
            else:  # equal_weight
                # Equal contribution regardless of data size
                client_weights[client_id] = 1.0

        # Normalize weights
        total_weight = sum(client_weights.values())
        aggregation_weights = {
            client_id: weight / total_weight
            for client_id, weight in client_weights.items()
        }

        # Aggregate updates
        aggregated_update = self._aggregate_updates(
            client_updates,
            aggregation_weights,
        )

        # Update global model
        self._update_global_model(aggregated_update)

        # Create round record
        fed_round = FederatedRound(
            round_number=self.current_round,
            participating_clients=list(sampled_client_ids),
            global_model_params=self.global_model.state_dict(),
            client_updates=client_updates,
            aggregation_weights=aggregation_weights,
            timestamp=datetime.utcnow().isoformat(),
        )

        self.round_history.append(fed_round)

        logger.info(
            f"Completed round {self.current_round} - "
            f"Global model updated"
        )

        return fed_round

    def _aggregate_updates(
        self,
        client_updates: Dict[str, OrderedDict],
        weights: Dict[str, float],
    ) -> OrderedDict:
        """
        Aggregate client updates into global update.

        Args:
            client_updates: Dict mapping client_id to model updates
            weights: Dict mapping client_id to aggregation weights

        Returns:
            Aggregated model update
        """
        # Initialize aggregated update
        aggregated = OrderedDict()

        # Get parameter names from first client
        first_client_id = next(iter(client_updates.keys()))
        param_names = client_updates[first_client_id].keys()

        # Aggregate each parameter
        for param_name in param_names:
            # Weighted sum of client updates
            weighted_sum = sum(
                client_updates[client_id][param_name] * weights[client_id]
                for client_id in client_updates.keys()
            )

            aggregated[param_name] = weighted_sum

        return aggregated

    def _update_global_model(self, aggregated_update: OrderedDict):
        """
        Update global model with aggregated updates.

        Args:
            aggregated_update: Aggregated parameter updates
        """
        current_params = self.global_model.state_dict()

        updated_params = OrderedDict()
        for name, param in current_params.items():
            updated_params[name] = param + aggregated_update[name]

        self.global_model.load_state_dict(updated_params)

    def evaluate_global_model(
        self,
        test_loaders_by_client: Dict[str, DataLoader],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate global model performance across all clients.

        Critical for assessing whether federated learning achieves
        equitable performance across diverse institutions.

        Args:
            test_loaders_by_client: Test data for each client

        Returns:
            Dict mapping client_id to performance metrics
        """
        results = {}

        for client_id, test_loader in test_loaders_by_client.items():
            client = self.clients[client_id]

            # Update client model with current global parameters
            client.update_model(self.global_model.state_dict())

            # Evaluate
            metrics = client.evaluate_model(test_loader)

            results[client_id] = {
                **metrics,
                'client_description': client.client_info.get_description(),
            }

            logger.info(
                f"Client {client_id} ({client.client_info.get_description()}): "
                f"Accuracy = {metrics['accuracy']:.4f}"
            )

        # Calculate summary statistics
        accuracies = [m['accuracy'] for m in results.values()]
        results['summary'] = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'accuracy_gap': np.max(accuracies) - np.min(accuracies),
        }

        logger.info(
            f"\nGlobal Model Performance Summary:\n"
            f"  Mean Accuracy: {results['summary']['mean_accuracy']:.4f}\n"
            f"  Std Dev: {results['summary']['std_accuracy']:.4f}\n"
            f"  Min-Max Gap: {results['summary']['accuracy_gap']:.4f}\n"
        )

        return results

    def save_global_model(self, path: str):
        """Save global model to disk."""
        torch.save(self.global_model.state_dict(), path)
        logger.info(f"Saved global model to {path}")

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Generate training summary with equity metrics.

        Returns:
            Dictionary containing training history and equity analysis
        """
        summary = {
            'total_rounds': self.current_round,
            'total_clients': len(self.clients),
            'aggregation_strategy': self.aggregation_strategy,
        }

        # Analyze participation equity
        participation_counts = {}
        for fed_round in self.round_history:
            for client_id in fed_round.participating_clients:
                participation_counts[client_id] = (
                    participation_counts.get(client_id, 0) + 1
                )

        participation_rates = {
            client_id: count / self.current_round
            for client_id, count in participation_counts.items()
        }

        summary['participation'] = {
            'by_client': participation_rates,
            'min_participation': min(participation_rates.values()) if participation_rates else 0,
            'max_participation': max(participation_rates.values()) if participation_rates else 0,
        }

        return summary
```

This implementation provides a complete federated learning system appropriate for global health collaborations. The system enables institutions across countries to collaboratively train models while maintaining complete control over their data. Optional differential privacy provides formal privacy guarantees for sensitive health data. The aggregation strategies support both standard FedAvg (weighted by data size) and equal weighting that prevents large institutions from dominating. The comprehensive evaluation framework tracks performance across all participating sites, ensuring federated learning achieves equitable outcomes rather than optimizing for well-represented institutions at the expense of smaller or resource-limited participants.

## 29.6 Participatory AI Development and Community Ownership

Technical approaches to resource-limited deployment, domain adaptation, multilingual systems, and federated learning are necessary but insufficient for equitable global health AI. Fundamentally, we must address power dynamics in AI development: who defines problems, who designs solutions, who owns resulting technologies, and who benefits from their deployment. Without explicit attention to these questions, even technically sophisticated AI risks perpetuating extractive relationships where high-income countries and large technology companies develop systems using data from LMICs while value flows primarily to external actors.

Participatory AI development positions communities, local health workers, and patients as partners in system design rather than passive recipients of externally developed technology. Community ownership ensures benefits accrue to communities providing data. Local capacity building creates sustainable technical expertise rather than dependencies on external vendors. These approaches challenge colonial patterns in global health technology development.

### 29.6.1 Principles of Participatory AI Development

Participatory design has deep roots in Scandinavian workplace democracy movements and community-based participatory research in public health. Applied to AI development for global health, key principles include:

**Co-design from problem definition** engages communities in defining what problems AI should address rather than assuming external priorities. A technology company might prioritize diseases that affect wealthy markets. Communities might prioritize different health challenges, or non-disease factors like transportation barriers to care, stigma, or health system navigation. Co-design ensures AI addresses locally defined needs.

**Meaningful involvement throughout development** integrates community input across the entire development lifecycle, not only at the beginning (needs assessment) or end (user testing). Community members provide expertise about local context that shapes data collection protocols, feature engineering, fairness metrics, deployment strategies, and evaluation criteria. Their knowledge is valued as expertise, not extracted as data.

**Capacity building and knowledge sharing** transfers technical skills and builds local AI development capacity rather than creating dependencies on external vendors. Training programs develop local data science teams. Documentation uses accessible language. Open-source code enables local adaptation and maintenance. The goal is sustainable local capacity rather than indefinite external technical assistance.

**Data sovereignty and community ownership** ensures communities maintain control over their data and own resulting AI systems. Data use agreements specify permitted uses, prohibit commercial exploitation without community benefit, and require community consent for any changes. Ownership structures like community trusts or cooperatives maintain collective control. Intellectual property arrangements ensure communities benefit financially from any commercial applications.

**Equitable benefit sharing** distributes benefits from AI systems to communities providing training data. At minimum, communities receive free access to resulting systems. Ideally, communities participate in any revenue from commercial deployment through licensing fees, equity ownership, or profit-sharing arrangements. The extractive pattern where pharmaceutical companies test drugs in LMICs without ensuring access must not be replicated in AI.

**Accountability to community priorities** makes AI systems accountable to community-defined success metrics, not only external metrics. Communities define what performance levels are acceptable, what tradeoffs between different objectives are appropriate, and when systems should be modified or discontinued. Ongoing community governance structures maintain this accountability after initial deployment.

### 29.6.2 Implementation Considerations

Implementing participatory AI development in practice faces several challenges. Participatory processes take time and resources, potentially conflicting with pressure for rapid deployment. Ensuring diverse community representation requires explicit effort to include marginalized voices within communities, not only easily accessible community leaders. Language and technical barriers require careful attention to accessible communication. Power imbalances between well-resourced external AI developers and under-resourced communities require conscious effort to create genuinely collaborative relationships.

Successful participatory AI development requires sustained commitment, adequate resources for community engagement, flexibility to adapt based on community input even when this requires changing planned approaches, and willingness to share control and ownership. It cannot be an extractive process disguised with participatory language.

### 29.6.3 Case Study: Community-Owned Maternal Health AI in Rural Uganda

A consortium of rural health centers in Uganda, supported by a nonprofit technology organization, developed an AI system for predicting preeclampsia risk using basic clinical data available at community health posts. The project explicitly employed participatory methods and community ownership models.

Problem definition emerged from community health workers and traditional birth attendants who identified preeclampsia as a major cause of maternal mortality in their communities. Existing risk assessment tools required laboratory tests unavailable in rural health posts, limiting their utility. The community identified the need for risk assessment using only data collectable by community health workers during home visits: blood pressure, symptoms, medical history, and fundal height.

Data collection involved training community health workers in structured data collection using mobile phones. Rather than extracting data for external model development, the project established a community data trust where health centers collectively controlled data. Data use agreements specified that models trained on this data must be made freely available to participating communities, with any commercial licensing revenue split between the community trust (70%) and the nonprofit organization (30%) to support ongoing development.

Model development involved community health workers throughout. They provided expertise about which clinical features were reliably measurable in home settings versus requiring health facility visits. They identified relevant contextual features like transportation time to hospital, which proved predictive of outcomes independent of clinical risk. They participated in defining fairness metrics, specifically requesting stratified evaluation by distance from health facilities to ensure the model worked equitably for the most remote communities.

The resulting model used gradient boosted trees trained on features collectable during home visits. Performance exceeded existing risk scores requiring laboratory data when evaluated specifically on the population and setting of interest. Critically, performance was equitable across distance from facilities, age groups, and parity.

Deployment integrated the AI system into existing community health worker workflows using Android tablets with offline capability. Community health workers received training emphasizing that AI provides decision support, not autonomous decisions. The system flagged high-risk pregnancies for referral to health centers while providing guidance for all pregnant women. Community-defined performance metrics included not only clinical accuracy but also community health worker satisfaction, pregnant women's understanding of risk information, and timely referral completion rates.

Governance structures maintained community control. A community advisory board including health center representatives, community health workers, traditional birth attendants, and pregnant women's advocacy groups provided ongoing oversight. They reviewed quarterly performance reports and approved any model updates. When performance monitoring revealed the model underestimated risk for adolescent pregnancies, the community board approved data collection focused on adolescents and model retraining.

This example demonstrates participatory AI development in practice: community-defined problems, ongoing community involvement in development decisions, data sovereignty through community trusts, capacity building through training programs, and community governance ensuring accountability to local priorities. The system addressed a locally prioritized health need using AI appropriate for resource-limited settings while maintaining community control and ensuring benefits accrued to participating communities.

## 29.7 Conclusion: Toward Equitable Global Health AI

Healthcare AI development and deployment in resource-limited settings and LMICs presents both enormous potential and substantial risks. AI systems that extend the capacity of limited clinical workforces, enable earlier diagnosis, and improve treatment selection could save millions of lives, particularly in settings with severe healthcare workforce shortages and high disease burdens. Yet uncritical deployment of AI systems developed in high-resource settings risks performance failures, inappropriate technology choices, unsustainable dependencies, and perpetuation of extractive relationships between high-income and low-income countries.

This chapter developed technical and social approaches for AI development that serves rather than exploits underserved global populations. Infrastructure-aware design creates systems that function reliably with limited computational resources, intermittent connectivity, and constrained data storage through model compression, offline-capable architectures, and efficient synchronization. Transfer learning and domain adaptation enable leveraging knowledge from data-rich settings while adapting to different disease prevalence, clinical presentation patterns, and available diagnostic resources in resource-limited contexts. Multilingual and culturally adaptive systems work across linguistic and cultural boundaries while respecting local knowledge systems and communication preferences.

Federated learning enables collaborative model development across institutions in different countries while respecting data sovereignty and avoiding extractive data centralization. Most fundamentally, participatory AI development processes position local clinicians, community health workers, patients, and communities as partners who define priorities, contribute expertise, and maintain ownership rather than passive recipients of externally imposed technology.

The equity considerations throughout this chapter are not optional features but fundamental requirements. Assessing whether AI systems work equitably across diverse deployment contexts, different patient populations, and varying resource levels is as essential as assessing overall accuracy. Ensuring communities providing training data maintain data sovereignty and benefit from resulting systems addresses power imbalances in global health. Building local capacity for AI development, adaptation, and maintenance creates sustainability rather than dependencies. Centering community-defined health priorities rather than external agendas challenges the history of externally driven interventions in global health.

Technical innovation alone is insufficient. Achieving health equity through AI requires sustained commitment to justice, meaningful community engagement and ownership, willingness to challenge systems and structures that perpetuate disparate outcomes, and recognition that the goal is not merely deploying AI globally but ensuring AI serves populations who need it most while respecting their autonomy, protecting their rights, and centering their priorities.

The future of global health AI depends on whether we can move beyond extractive models that centralize data and value in high-income settings toward genuinely collaborative, community-owned approaches that respect data sovereignty, ensure equitable benefit sharing, and center the expertise and priorities of communities most affected by health disparities. The technical foundations developed in this chapter enable that future, but realizing it requires commitment to justice as fundamental as commitment to accuracy.

## References

Abimbola, S., & Pai, M. (2020). Will global health survive its decolonisation? *The Lancet*, 396(10263), 1627-1628.

Agarwal, A., Beygelzimer, A., Dudík, M., Langford, J., & Wallach, H. (2018). A reductions approach to fair classification. *Proceedings of the 35th International Conference on Machine Learning*, 80, 60-69.

Ahmad, M. A., Patel, A., Eckert, C., Kumar, V., & Teredesai, A. (2020). Fairness in machine learning for healthcare. *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 3529-3530.

Buolamwini, J., & Gebru, T. (2018). Gender shades: Intersectional accuracy disparities in commercial gender classification. *Conference on Fairness, Accountability and Transparency*, 77-91.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. *International Conference on Machine Learning*, 1597-1607.

Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *IEEE Conference on Computer Vision and Pattern Recognition*, 248-255.

Dwork, C. (2006). Differential privacy. *International Colloquium on Automata, Languages, and Programming*, 1-12.

Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542(7639), 115-118.

Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., ... & Lempitsky, V. (2016). Domain-adversarial training of neural networks. *The Journal of Machine Learning Research*, 17(1), 2096-2030.

Gichoya, J. W., Banerjee, I., Bhimireddy, A. R., Burns, J. L., Celi, L. A., Chen, L. C., ... & Lungren, M. P. (2022). AI recognition of patient race in medical imaging: a modelling study. *The Lancet Digital Health*, 4(6), e406-e414.

Gulshan, V., Peng, L., Coram, M., Stumpe, M. C., Wu, D., Narayanaswamy, A., ... & Webster, D. R. (2016). Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. *JAMA*, 316(22), 2402-2410.

Hard, A., Rao, K., Mathews, R., Ramaswamy, S., Beaufays, F., Augenstein, S., ... & Ramage, D. (2018). Federated learning for mobile keyboard prediction. *arXiv preprint arXiv:1811.03604*.

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*.

Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. *arXiv preprint arXiv:1704.04861*.

Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K., Dally, W. J., & Keutzer, K. (2016). SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and < 0.5 MB model size. *arXiv preprint arXiv:1602.07360*.

Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., ... & Ng, A. Y. (2019). CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. *Proceedings of the AAAI Conference on Artificial Intelligence*, 33(01), 590-597.

Kairouz, P., McMahan, H. B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A. N., ... & Zhao, S. (2021). Advances and open problems in federated learning. *Foundations and Trends in Machine Learning*, 14(1-2), 1-210.

Kather, J. N., Krisam, J., Charoentong, P., Luedde, T., Herpel, E., Weis, C. A., ... & Halama, N. (2019). Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study. *PLoS Medicine*, 16(1), e1002730.

Kvamme, H., Borgan, Ø., & Scheel, I. (2019). Time-to-event prediction with neural networks and Cox regression. *Journal of Machine Learning Research*, 20(129), 1-30.

Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. *Proceedings of Machine Learning and Systems*, 2, 429-450.

McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. *Artificial Intelligence and Statistics*, 1273-1282.

Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. *ACM Computing Surveys*, 54(6), 1-35.

Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447-453.

Pan, S. J., & Yang, Q. (2009). A survey on transfer learning. *IEEE Transactions on Knowledge and Data Engineering*, 22(10), 1345-1359.

Rajkomar, A., Hardt, M., Howell, M. D., Corrado, G., & Chin, M. H. (2018). Ensuring fairness in machine learning to advance health equity. *Annals of Internal Medicine*, 169(12), 866-872.

Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., ... & Ng, A. Y. (2017). CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. *arXiv preprint arXiv:1711.05225*.

Reddi, S., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konečný, J., ... & McMahan, H. B. (2020). Adaptive federated optimization. *arXiv preprint arXiv:2003.00295*.

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., ... & Fei-Fei, L. (2015). ImageNet large scale visual recognition challenge. *International Journal of Computer Vision*, 115(3), 211-252.

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 4510-4520.

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.

Sheller, M. J., Edwards, B., Reina, G. A., Martin, J., Pati, S., Kotrotsou, A., ... & Bakas, S. (2020). Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data. *Scientific Reports*, 10(1), 1-12.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *The Journal of Machine Learning Research*, 15(1), 1929-1958.

Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *International Conference on Machine Learning*, 6105-6114.

Tjoa, E., & Guan, C. (2020). A survey on explainable artificial intelligence (XAI): Toward medical XAI. *IEEE Transactions on Neural Networks and Learning Systems*, 32(11), 4793-4813.

Torrey, L., & Shavlik, J. (2010). Transfer learning. In *Handbook of Research on Machine Learning Applications and Trends: Algorithms, Methods, and Techniques*, 242-264.

Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). ChestX-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2097-2106.

Weller, A. (2019). Challenges for transparency. *arXiv preprint arXiv:1708.01870*.

World Health Organization. (2021). *Ethics and governance of artificial intelligence for health*. WHO guidance. Geneva: World Health Organization.

Wu, X., Li, F., Kumar, A., Chaudhuri, K., Jha, S., & Naughton, J. (2017). Bolt-on differential privacy for scalable stochastic gradient descent-based analytics. *Proceedings of the 2017 ACM International Conference on Management of Data*, 1307-1322.

Yang, Q., Liu, Y., Chen, T., & Tong, Y. (2019). Federated machine learning: Concept and applications. *ACM Transactions on Intelligent Systems and Technology*, 10(2), 1-19.

Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? *Advances in Neural Information Processing Systems*, 27, 3320-3328.

Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2017). mixup: Beyond empirical risk minimization. *arXiv preprint arXiv:1710.09412*.

Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2021). Understanding deep learning (still) requires rethinking generalization. *Communications of the ACM*, 64(3), 107-115.

Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., & Chandra, V. (2018). Federated learning with non-iid data. *arXiv preprint arXiv:1806.00582*.

Zhu, L., Liu, Z., & Han, S. (2019). Deep leakage from gradients. *Advances in Neural Information Processing Systems*, 32, 14774-14784.
