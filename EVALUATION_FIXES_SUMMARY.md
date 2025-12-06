# Evaluation and Plotting Fixes Summary

## Issues Found and Fixed

### 1. **Root Cause: Non-Stratified Data Splitting**
**Problem**: The `split_data()` function in `src/train_model.py` was using simple random shuffling without stratification. With only 20 samples total and a 10% test split (2 samples), it was very likely that both test samples would be from the same class, causing:
- Confusion matrix with only one row populated
- Precision/Recall/F1 = 0 for the missing class
- Misleading evaluation results

**Fix**: Implemented **stratified splitting** using `sklearn.model_selection.train_test_split` with `stratify=labels` parameter. This ensures both classes are represented in train/val/test splits proportionally.

**File Changed**: `src/train_model.py` - `split_data()` function

### 2. **Confusion Matrix Computation**
**Problem**: Confusion matrix was computed without explicit label ordering, which could lead to incorrect interpretation when one class is missing.

**Fix**: 
- Added explicit `labels=[0, 1]` parameter to all `confusion_matrix()` calls
- Ensured confusion matrix is always 2x2 even if one class is missing
- Added proper logging of confusion matrix values

**Files Changed**: 
- `src/evaluate_model.py` - `plot_confusion_matrix()` function
- `src/evaluate_model.py` - `evaluate_model()` function (Step 7)

### 3. **Metrics Computation**
**Problem**: Per-class metrics could return arrays of length 1 when one class was missing, causing index errors.

**Fix**:
- Added explicit `labels=[0, 1]` to all per-class metric computations
- Added padding to ensure arrays always have 2 values (pad with 0 if class missing)
- Ensured metrics dictionary always has both classes in per-class metrics

**File Changed**: `src/evaluate_model.py` - `compute_metrics()` function

### 4. **Sanity Checks**
**Problem**: No validation before computing metrics, leading to misleading results.

**Fix**: Added comprehensive sanity checks:
- Minimum test set size check (warns if < 10 samples)
- Both classes presence check (warns if missing)
- Prediction distribution logging
- Clear error messages when issues are detected

**File Changed**: `src/evaluate_model.py` - `evaluate_model()` function (Step 3)

### 5. **LaTeX Table Generation**
**Problem**: String formatting with `.format()` was failing due to LaTeX braces `{{` and `}}` being interpreted as format placeholders.

**Fix**: Changed from `.format()` to f-string formatting for all LaTeX table generation functions.

**File Changed**: `src/generate_latex_tables.py` - All three table generation functions

### 6. **Encoding Issues**
**Problem**: Writing summary file with emoji characters caused encoding errors on Windows.

**Fix**: Added `encoding='utf-8'` to file write operations.

**File Changed**: `src/evaluate_model.py` - `evaluate_model()` function (Step 8)

## Verification Results

### Test Run with Current Data (20 samples, 10 per class)
After fixes, with stratified splitting:
- **Test set**: 2 samples (1 class 0, 1 class 1) ✅
- **y_true**: [0, 1] ✅
- **y_pred**: [0, 1] (perfect predictions) ✅
- **Confusion Matrix**:
  ```
                Predicted
              Correct  Hallucination
  Actual Correct      1      0
        Hallucination  0      1
  ```
  - TN = 1, FP = 0, FN = 0, TP = 1 ✅

- **Metrics**:
  - Accuracy: 1.0 ✅
  - Precision (binary, pos=1): 1.0 ✅
  - Recall (binary, pos=1): 1.0 ✅
  - F1-Score (binary, pos=1): 1.0 ✅
  - Per-class metrics: Both classes have non-zero values ✅

### What the Correct Confusion Matrix Looks Like

With both classes present and perfect predictions:
```
                Predicted
              Correct  Hallucination
Actual Correct      1      0
      Hallucination  0      1
```

With both classes present but some errors:
```
                Predicted
              Correct  Hallucination
Actual Correct      TN      FP
      Hallucination  FN      TP
```

All four values (TN, FP, FN, TP) should be non-negative integers, and at least TN and TP should be > 0 if the model is working.

## Files Modified

1. **src/train_model.py**
   - Changed `split_data()` to use stratified splitting
   - Added class distribution logging

2. **src/evaluate_model.py**
   - Added comprehensive documentation header
   - Enhanced `compute_metrics()` with explicit labels and padding
   - Fixed `plot_confusion_matrix()` with explicit labels
   - Added sanity checks in `evaluate_model()`
   - Fixed confusion matrix computation with explicit labels
   - Added encoding='utf-8' to file writes
   - Enhanced logging throughout

3. **src/generate_latex_tables.py**
   - Fixed string formatting in all three table generation functions
   - Changed from `.format()` to f-strings to avoid LaTeX brace issues

## Label Convention (Confirmed)

- **0 = Correct / Non-Hallucination** (negative class)
- **1 = Hallucination** (positive class)

This convention is consistent across:
- Dataset loading
- Training
- Evaluation
- Plotting
- Metrics computation

## Current Status

✅ **All fixes applied and verified**
✅ **Stratified splitting ensures both classes in test set**
✅ **Confusion matrix correctly computed with explicit labels**
✅ **Metrics correctly computed for both classes**
✅ **LaTeX tables generating without errors**
✅ **All plots saving correctly**

## Next Steps (For Larger Dataset)

When using a larger dataset:
1. The stratified splitting will automatically ensure balanced test sets
2. Metrics will be more reliable with larger sample sizes
3. All sanity checks will help identify any remaining issues

## Testing

Run the following to verify:
```bash
python src/master_pipeline.py --config config.json
```

Expected output:
- Test set contains both classes (verified in logs)
- Confusion matrix shows 2x2 with all four values
- All metrics (precision, recall, F1) are non-zero for both classes
- LaTeX tables generate successfully
- All plots save correctly

