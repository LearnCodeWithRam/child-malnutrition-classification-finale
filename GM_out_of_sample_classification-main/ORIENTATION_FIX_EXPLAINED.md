# 🔧 Orientation Issue: What Was Wrong and How It Was Fixed

## The Problem

After running GPA (Generalized Procrustes Analysis), individual gorilla skull shapes were NOT following their group mean shape orientation. You could see:

- Female skulls with notches pointing in random directions
- Male skulls with notches pointing in random directions  
- Even though each group had a clear mean shape orientation, individual shapes were scattered

**Your key observation:** "at least other shapes should follow the notch direction of its mean shape"

## Root Cause Analysis

### Issue #1: Wrong Angle Calculation
```python
# ❌ OLD CODE (WRONG)
theta = np.arctan(alpha[1] / alpha[0])
```

**Problem:** `arctan(y/x)` only returns angles in range `[-π/2, π/2]` (only 2 quadrants). This means:
- If the notch points left (negative x), the angle is wrong
- If the notch points down (negative y), the angle is wrong
- Only works correctly for upper-right quadrant

### Issue #2: Function Not Returning Rotated Mean Shape
```python
# ❌ OLD CODE (WRONG)
def rotate_skulls(Proc_rot, ms):
    # ... rotation code ...
    return Rot  # Only returning rotated shapes, not rotated mean!
```

**Problem:** The mean shape itself wasn't being rotated, so plots couldn't show the rotated mean shape properly.

### Issue #3: Not Unpacking Both Return Values
```python
# ❌ OLD CODE (WRONG)
Proc_F_rotated = rotate_skulls(Proc_F['rotated'], Proc_F['mshape'])
# Only getting rotated shapes, losing the rotated mean shape
```

## The Solution

### Fix #1: Use `atan2()` for Correct Quadrant Detection
```python
# ✅ NEW CODE (CORRECT)
theta = np.arctan2(alpha[1], alpha[0])  # Handles all 4 quadrants correctly
```

**Why this works:** `atan2(y, x)` returns angles in full range `[-π, π]` (all 4 quadrants):
- Upper-right: 0 to π/2
- Upper-left: π/2 to π  
- Lower-left: -π to -π/2
- Lower-right: -π/2 to 0

### Fix #2: Return Both Rotated Shapes AND Rotated Mean
```python
# ✅ NEW CODE (CORRECT)
def rotate_skulls(Proc_rot, ms):
    """Rotate ALL shapes consistently so mean shape landmarks 0 and 2 are horizontal"""
    n = Proc_rot.shape[2]
    
    # Calculate rotation angle from MEAN SHAPE (landmarks 0 to 2)
    alpha = ms[2, :] - ms[0, :]
    theta = np.arctan2(alpha[1], alpha[0])  # Use atan2 for correct quadrant
    
    # Create rotation matrix (rotate by -theta to make horizontal)
    R = np.array([[np.cos(-theta), -np.sin(-theta)], 
                  [np.sin(-theta), np.cos(-theta)]])
    
    # Apply SAME rotation to ALL shapes
    Rot = np.zeros_like(Proc_rot)
    for i in range(n):
        Rot[:, :, i] = Proc_rot[:, :, i] @ R
    
    # Also rotate the mean shape
    ms_rotated = ms @ R
    
    return Rot, ms_rotated  # Return BOTH!
```

### Fix #3: Unpack Both Return Values
```python
# ✅ NEW CODE (CORRECT)
Proc_F_rotated, Proc_F_mshape_rotated = rotate_skulls(Proc_F['rotated'], Proc_F['mshape'])
Proc_M_rotated, Proc_M_mshape_rotated = rotate_skulls(Proc_M['rotated'], Proc_M['mshape'])
Proc_rotated, Proc_mshape_rotated = rotate_skulls(Proc['rotated'], Proc['mshape'])
```

## Why This Matters for Your Research

### Before GPA
- Both F and M gorilla skulls have notches pointing in the same direction (top)
- This is just how they were photographed/positioned

### After GPA (with old buggy code)
- Notches pointing in random directions ❌
- Can't see true shape differences
- Looks like chaos

### After GPA (with fixed code)  
- All individual shapes follow their group mean orientation ✅
- Female shapes consistently oriented with female mean
- Male shapes consistently oriented with male mean
- True shape differences clearly visible
- Sexual dimorphism apparent

## How to Apply the Fix

1. **Regenerate the notebook:**
   ```bash
   cd GM_out_of_sample_classification-main
   python create_notebook.py
   ```

2. **Clear Jupyter cache (CRITICAL!):**
   - Open the notebook in Jupyter
   - Click: `Kernel` → `Restart & Clear Output`
   - Click: `Cell` → `Run All`

3. **Verify the fix worked:**
   - Look for message: "📌 All individual shapes follow their group mean orientation!"
   - Check plots: individual shapes should be consistently oriented with their group mean
   - Female notches should all point in the same direction as female mean
   - Male notches should all point in the same direction as male mean

## Technical Summary

**What was wrong:**
- Used `arctan()` instead of `atan2()` → wrong angles in 2 quadrants
- Didn't return rotated mean shape → couldn't plot it correctly
- Didn't unpack both return values → lost the rotated mean shape

**How it was fixed:**
- Use `atan2(y, x)` for correct angle calculation in all quadrants
- Return tuple `(rotated_shapes, rotated_mean)` from function
- Unpack both values: `shapes, mean = rotate_skulls(...)`

**Result:**
- All shapes now consistently oriented with their group mean
- True shape differences clearly visible
- Ready for classification analysis

---

**Note:** The notebook file itself was outdated (had old code). Running `python create_notebook.py` regenerated it with the fixes. You must clear Jupyter's cache to see the changes!
