# RESET
This is the code for **RESET: A Memory-Efficient Hash-Based Algorithm for Set Operations on GPUs**.

## Organization

1. **Preprocessing vs. kernel time**  
   Benchmarks that break down time between preprocessing and GPU kernels.

2. **Three set applications**  
   Triangle counting, information retrieval, and set similarity search, including comparisons against cuCollection and native baselines where applicable.

3. **Multi-GPU**  
   Tests and driver logic for multi-device setups.

4. **Intersection at different scales**  
   Set-intersection benchmarks over varying set sizes.

5. **SMOG subgraph matching**  
   Integration and application tests on the SMOG subgraph-matching pipeline.

6. **Hash-function comparisons**  
   Experiments comparing alternative hash functions.

7. **GPU-specific hyperparameter search**  
   Utilities to tune hyperparameters for a given GPU.

## Environment
Toolkit 12.2; g++ 12.2.0;
