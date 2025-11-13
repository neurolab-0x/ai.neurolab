# Codebase Analysis: Inconsistencies and Duplications

## Executive Summary

The codebase has significant duplication between `/core` and `/preprocessing` directories, with multiple implementations of the same functionality. This analysis identifies all duplicates and provides a unification strategy.

## Critical Duplications Found

### 1. **load_data() Function**
- **Location 1**: `core/ml/processing.py` (lines 11-28)
- **Location 2**: `preprocessing/load_data.py` (lines 9-40)
- **Issue**: Two completely different implementations
  - `core/ml/processing.py`: Simple CSV loader, returns numpy array
  - `preprocessing/load_data.py`: Comprehensive loader supporting CSV, EDF, BDF, GDF, MAT formats
- **Recommendation**: **Keep `preprocessing/load_data.py`** (more comprehensive)

### 2. **extract_features() Function**
- **Location 1**: `core/ml/processing.py` (lines 67-85)
- **Location 2**: `preprocessing/features.py` (lines 312-400+)
- **Issue**: Two completely different implementations
  - `core/ml/processing.py`: Simple feature extraction to EEGFeatures objects
  - `preprocessing/features.py`: Comprehensive feature extraction with 100+ features
- **Recommendation**: **Keep `preprocessing/features.py`** (production-ready)

### 3. **label_eeg_states() Function**
- **Location 1**: `core/ml/processing.py` (lines 30-65)
- **Location 2**: `preprocessing/labeling.py` (lines 9-150+)
- **Issue**: Two different implementations
  - `core/ml/processing.py`: Simple ratio-based labeling
  - `preprocessing/labeling.py`: Multiple methods (frequency, threshold, clustering)
- **Recommendation**: **Keep `preprocessing/labeling.py`** (more flexible)

### 4. **preprocess_data() Function**
- **Location 1**: `core/ml/processing.py` (lines 87-105)
- **Location 2**: `preprocessing/preprocess.py` (lines 200-400+)
- **Issue**: Two completely different implementations
  - `core/ml/processing.py`: Simple normalization and splitting
  - `preprocessing/preprocess.py`: Comprehensive pipeline with validation, augmentation, balancing
- **Recommendation**: **Keep `preprocessing/preprocess.py`** (production-ready)

### 5. **load_calibrated_model() Function**
- **Location 1**: `core/ml/model.py` (lines 37-50)
- **Location 2**: `utils/model_loading.py` (lines 9-40)
- **Issue**: Two similar implementations with slight differences
  - `core/ml/model.py`: Returns Optional[tf.keras.Model], creates new model if not found
  - `utils/model_loading.py`: Returns keras.Model, raises FileNotFoundError if not found
- **Recommendation**: **Keep `core/ml/model.py`** (better error handling)

### 6. **create_model() Function**
- **Location 1**: `core/ml/model.py` (lines 9-35)
- **Location 2**: `train_model.py` (lines 17-35)
- **Issue**: Two similar implementations
  - Both create simple CNN models
  - Slightly different architectures
- **Recommendation**: **Keep `core/ml/model.py`** (more comprehensive with error handling)

## Import Inconsistencies

### Files importing from `core.ml`:
1. `utils/model_manager.py` - imports `load_calibrated_model`
2. `utils/ml_processor.py` - imports multiple functions
3. `train_model.py` - imports data processing functions

### Files importing from `preprocessing`:
1. `preprocessing/preprocess.py` - imports `extract_features`
2. `api/upload.py` - imports multiple functions
3. `api/training.py` - imports multiple functions
4. `api/real_time.py` - imports preprocessing functions

## Structural Issues

### 1. **Circular Dependencies Risk**
- `core/ml/processing.py` and `preprocessing/*` have overlapping functionality
- `utils/ml_processor.py` imports from both `core.ml` and `preprocessing`

### 2. **Inconsistent Module Organization**
- Data processing logic split between `core/ml/processing.py` and `preprocessing/`
- Model logic split between `core/ml/model.py`, `utils/model_loading.py`, and `models/model.py`

### 3. **Duplicate Functionality**
- Signal processing functions duplicated across multiple files
- Feature extraction logic scattered across codebase

## Recommended Unification Strategy

### Phase 1: Consolidate Data Processing (HIGH PRIORITY)
1. **Delete** `core/ml/processing.py` entirely
2. **Keep** all `preprocessing/*` modules
3. **Update** all imports to use `preprocessing` modules
4. **Create** `preprocessing/__init__.py` with clean exports

### Phase 2: Consolidate Model Management (HIGH PRIORITY)
1. **Keep** `core/ml/model.py` for model creation and loading
2. **Delete** redundant functions from `utils/model_loading.py`
3. **Keep** only utility functions in `utils/model_loading.py` (get_available_models, save_model, etc.)
4. **Update** `train_model.py` to import from `core/ml/model.py`

### Phase 3: Consolidate Feature Extraction (MEDIUM PRIORITY)
1. **Keep** `preprocessing/features.py` as the single source
2. **Remove** simple feature extraction from `core/ml/processing.py`
3. **Update** all imports

### Phase 4: Clean Up Utils (LOW PRIORITY)
1. Review `utils/` directory for any remaining duplicates
2. Ensure clear separation of concerns

## Impact Analysis

### Files Requiring Updates After Unification:

1. **utils/ml_processor.py** - Update imports from `core.ml.processing` to `preprocessing`
2. **train_model.py** - Update imports from `core.ml.processing` to `preprocessing` and `core.ml.model`
3. **utils/model_manager.py** - Keep as is (already uses `core.ml.model`)
4. **api/upload.py** - Keep as is (alreadyn
idatioting and valtes)
7. Final onsolidationion Ccture Extraeate 3 (F Phaslementte
6. Impll test sui Run fuon)
5.olidatiment Consage (Model Manse 2ent Phaem. Imple
4l test suit
3. Run fulidation)ing ConsolData Processase 1 (ment PhImplebranch
2. p e backueat. Cr Steps

1k

## Nexty rollbacllows eason control a*: Git versi Plan*ack
- **Rollbaseeach ph after ive testingehensomprgation**: C
- **Mitiiesar boundares with clete modulin separates are st duplicaeason**: Mo **RIUM
-LOW to MEDevel**: - **Risk Lsment

# Risk Assescode

#e plicatte duinaimSize**: Elundle *Reduced B
5. *esponsibilitiresus module e**: ObviorchitecturClearer A
4. ** of multipleadinsteon ementatimpl i: Test one**stingter Te3. **Betuse
ion to implementat which fusion about: No conistency**ved Cons**Improction
2. funr each th foource of trungle snance**: Sinte MaiReduced

1. **Unificationenefits of s

## Btion ~30 location calls:les
- Func ~15 fiements:Import stat Update:
-  of Code toated Linesim## Esttion)

#funcel _modedbratoad_calive lnes (remo: ~50 lig.py`model_loadins/ `utilion)
-odel functreate_me clines (removel.py`: ~20 od
- `train_mNTIRE FILE)s (DELETE E line.py`: ~400essingproc
- `core/ml/emove:Code to Rf  oted Lines
### Estimaessing`)
proc`prelready uses (aeep as is ** - K_time.pypi/real*a)
6. *ssing` `preproceuseseady lr as is (a Keep* -aining.py*api/trng`)
5. **essirocses `prep u