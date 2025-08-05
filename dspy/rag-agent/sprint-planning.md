# Sprint Planning: DSPy RAG Agent Code Cleanup & Modularization

## 🎯 Sprint Objective
Clean up and modularize the `03-build-dspy-rag-agent.py` by extracting functions into dedicated modules (`optimizer.py`, `deploy.py`) while maintaining full functionality and ensuring proper MLflow integration for deployment.

## 📋 Current State Analysis

### Current File Structure
```
dspy/rag-agent/
├── 03-build-dspy-rag-agent.py (1127 lines) - MAIN TARGET FOR CLEANUP
├── agent.py (494 lines) - MLflow ChatAgent implementation
├── utils.py (175 lines) - Helper functions for retrieval & loading
├── metrics.py (344 lines) - Comprehensive evaluation metrics
├── test_agent.py (1034 lines) - Test suite for development
├── config.yaml (61 lines) - Configuration file
└── requirements.txt (13 lines) - Dependencies
```

### Key Problems Identified
1. **Monolithic main script**: `03-build-dspy-rag-agent.py` contains 1127 lines with mixed concerns
2. **Function coupling**: Optimization, deployment, and agent creation logic mixed together
3. **Hard to test**: Large functions make unit testing difficult
4. **Code reusability**: Functions locked inside main script can't be reused
5. **MLflow deployment**: Need to ensure new modules are included in deployment artifacts

## 🏗️ Proposed Target Structure

### New Modular Structure
```
dspy/rag-agent/
├── 01-dspy-data-preparation.py (unchanged)
├── 02-create-eval-dataset.py (unchanged)
├── 03-build-dspy-rag-agent.py (reduced to ~300 lines) - Main orchestration
├── agent.py (unchanged) - MLflow ChatAgent implementation
├── utils.py (enhanced) - General utilities + retrieval
├── optimizer.py (NEW) - DSPy optimization workflow
├── deploy.py (NEW) - Deployment and MLflow logging
├── metrics.py (unchanged) - Evaluation metrics
├── test_agent.py (enhanced) - Test suite + new module tests
├── config.yaml (unchanged) - Configuration
└── requirements.txt (unchanged) - Dependencies
```

## 📦 Module Breakdown Plan

### 1. `optimizer.py` (NEW MODULE)
**Purpose**: Handle all DSPy optimization logic

**Functions to Extract:**
- `prepare_training_data(spark, config)`
- `setup_evaluation_metric(judge_lm, config)`
- `run_optimization_workflow(lm, base_program, llm_config, model_config)`
- `run_bootstrap_optimization(program, training_examples, metric, config)`
- `run_miprov2_optimization(program, training_examples, metric, config)`
- `evaluate_program_performance(program, examples, metric)`
- `save_optimization_artifacts(program, metadata, mlflow_run)`

**Key Features:**
- Multi-stage optimization strategy support
- Comprehensive metrics integration
- MLflow logging for optimization results
- Configurable optimization parameters

### 2. `deploy.py` (NEW MODULE)
**Purpose**: Handle MLflow logging, model registration, and deployment

**Functions to Extract:**
- `create_final_agent(model_config, optimized_program, optimized_program_path)`
- `log_model_to_mlflow(final_config, llm_config, vector_search_config, optimized_program_path, optimization_results)`
- `test_logged_model(model_info)`
- `validate_model_deployment(model_info)`
- `deploy_model_to_serving(uc_full_model_name, model_info, llm_config, vector_search_config, optimized_program_path)`
- `register_model_in_unity_catalog(model_info, uc_model_name)`

**Key Features:**
- Full MLflow integration
- Model validation before deployment
- Unity Catalog registration
- Deployment to Model Serving endpoints

### 3. `utils.py` (ENHANCED)
**Functions to Add:**
- `setup_environment()` (moved from main script)
- `print_configuration(config)` (moved from main script)
- `load_config_safely(config_path)` (new utility)
- `validate_config_completeness(config)` (new utility)

## ⚡ Fast Optimization Configuration

Based on README.md analysis, update the optimization config for fast execution:

### Quick Iteration and Testing Mode (5-10 minutes)
```python
OPTIMIZATION_CONFIG = {
    "strategy": "bootstrap_only",           # Fast bootstrap-only strategy
    "auto_level": "light",                  # Light optimization level
    "num_threads": 4,                       # Parallel processing
    "training_examples_limit": 20,          # Reduced dataset for speed
    "evaluation_examples_limit": 5,         # Minimal evaluation set
    "bootstrap_config": {
        "max_bootstrapped_demos": 2,        # Minimal demos for speed
        "max_labeled_demos": 1,             # Single labeled example
        "metric_threshold": 0.3             # Standard threshold
    }
}
```

## 🧪 Testing Strategy

### 1. Pre-Refactoring Tests
```bash
# Baseline functionality test
python test_agent.py

# Current optimization test (with original code)
python 03-build-dspy-rag-agent.py # (set OPTIMIZE_AGENT = True, fast config, and DEPLOY_MODEL=False)
```

### 2. During Refactoring Tests
```bash
# Test individual modules as they're created
python -c "from optimizer import prepare_training_data; print('✅ optimizer import works')"
python -c "from deploy import create_final_agent; print('✅ deploy import works')"

# Test module functionality
python test_agent.py # (enhanced with new module tests)
```

### 3. Post-Refactoring Integration Tests
```bash
# Full workflow test with modular code
python 03-build-dspy-rag-agent.py

# Deployment artifact validation
python -c "import mlflow; model = mlflow.pyfunc.load_model('path/to/model'); print('✅ Deployment artifacts complete')"
```

### 4. New Test Functions to Add
- `test_optimizer_module()` - Test optimization functions
- `test_deploy_module()` - Test deployment functions  
- `test_mlflow_artifacts()` - Verify all modules included in MLflow logging
- `test_fast_optimization_config()` - Test fast config scenarios
- `test_module_imports()` - Test circular imports and dependencies

## 📋 Implementation Tasks

### Sprint Tasks (Priority Order)

#### Task 1: Create `optimizer.py` Module ⏱️ 3-4 hours
**Acceptance Criteria:**
- [ ] Extract all optimization functions from main script
- [ ] Maintain all existing functionality
- [ ] Add proper error handling and logging
- [ ] Include comprehensive docstrings
- [ ] Test with `test_agent.py`

**Sub-tasks:**
1. Create `optimizer.py` file structure
2. Extract and refactor `prepare_training_data()`
3. Extract and refactor `setup_evaluation_metric()`
4. Extract and refactor `run_optimization_workflow()`
5. Add module-level configuration handling
6. Update main script to import from optimizer module
7. Test optimization workflow end-to-end

#### Task 2: Create `deploy.py` Module ⏱️ 2-3 hours
**Acceptance Criteria:**
- [ ] Extract all deployment functions from main script
- [ ] Ensure MLflow artifacts include all new modules
- [ ] Maintain deployment functionality
- [ ] Add validation for deployment readiness
- [ ] Test deployment pipeline

**Sub-tasks:**
1. Create `deploy.py` file structure
2. Extract MLflow logging functions
3. Extract deployment functions
4. Update artifact inclusion to reference new modules
5. Update main script to import from deploy module
6. Test full deployment pipeline

#### Task 3: Update Main Script ⏱️ 1-2 hours
**Acceptance Criteria:**
- [ ] Reduce main script to orchestration logic only
- [ ] Maintain all workflow functionality
- [ ] Clean up imports and remove duplicate code
- [ ] Update fast optimization configuration
- [ ] Comprehensive testing

**Sub-tasks:**
1. Remove extracted functions from main script
2. Add imports for new modules
3. Update optimization configuration to fast mode
4. Clean up duplicate utilities
5. Update workflow orchestration
6. Test complete workflow

#### Task 4: Enhance Testing ⏱️ 2-3 hours
**Acceptance Criteria:**
- [ ] Add tests for new modules
- [ ] Test MLflow artifact inclusion
- [ ] Test fast optimization configurations
- [ ] Validate deployment artifacts
- [ ] Document testing procedures

**Sub-tasks:**
1. Add `test_optimizer_module()` to test suite
2. Add `test_deploy_module()` to test suite
3. Add `test_mlflow_artifacts()` validation
4. Add `test_fast_optimization()` scenarios
5. Update testing documentation
6. Run comprehensive test suite

#### Task 5: Documentation & Validation ⏱️ 1 hour
**Acceptance Criteria:**
- [ ] Update README.md with new structure
- [ ] Document new modules and their usage
- [ ] Validate all configurations work
- [ ] Performance benchmarking
- [ ] Final integration test

**Sub-tasks:**
1. Update README.md module documentation
2. Add docstrings and inline documentation
3. Run performance comparison (before/after)
4. Final integration test with deployment
5. Document fast optimization configurations

## 🚨 Risk Mitigation

### Critical Risks & Mitigation
1. **MLflow Artifact Missing**: Ensure `code_paths` includes new modules
2. **Import Circular Dependencies**: Design clear module hierarchy
3. **Configuration Coupling**: Pass config explicitly, avoid global state
4. **Optimization Regression**: Test optimization results before/after
5. **Deployment Failure**: Validate deployment artifacts thoroughly

### Rollback Plan
1. Keep backup of original `03-build-dspy-rag-agent.py`
2. Git branch for all changes with clear commits
3. Test suite must pass before any merges
4. Deployment validation before production use

## 📊 Success Metrics

### Quantitative Goals
- [ ] **Code Reduction**: Main script reduced from 1127 to ~300 lines (73% reduction)
- [ ] **Test Coverage**: All new modules have dedicated tests
- [ ] **Performance**: Optimization runs in 5-10 minutes (fast config)
- [ ] **Functionality**: 100% feature parity with original script
- [ ] **Deployment**: Successful MLflow deployment with all artifacts

### Qualitative Goals
- [ ] **Maintainability**: Clear separation of concerns
- [ ] **Reusability**: Modules can be imported independently
- [ ] **Testability**: Individual functions can be unit tested
- [ ] **Documentation**: Clear docstrings and usage examples
- [ ] **Reliability**: Robust error handling and validation

## 🎉 Definition of Done

### Sprint Completion Criteria
- [ ] All modules created and tested
- [ ] Main script refactored and functional
- [ ] Fast optimization configuration implemented
- [ ] Complete test suite passing
- [ ] MLflow deployment successful with all artifacts
- [ ] Documentation updated
- [ ] Performance benchmarks completed
- [ ] Code review and approval

### Ready for Production
- [ ] End-to-end workflow tested
- [ ] Deployment validation passed
- [ ] Performance meets or exceeds baseline
- [ ] All tests passing consistently
- [ ] Documentation complete and accurate

---

## 🚀 Next Steps

1. **Start with Task 1**: Create `optimizer.py` module
2. **Test Early and Often**: Run tests after each major change
3. **Incremental Approach**: One module at a time with full testing
4. **Continuous Validation**: Ensure MLflow artifacts complete throughout
5. **Performance Monitoring**: Track optimization times and accuracy

This sprint should result in a cleaner, more maintainable, and faster DSPy RAG agent framework while preserving all existing functionality and improving development velocity.